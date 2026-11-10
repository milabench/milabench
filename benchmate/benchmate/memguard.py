"""memguard: stop a benchmark before it runs the machine out of memory.

On unified-memory machines (DGX Spark / GB10, APUs) the GPU allocates from system RAM: a benchmark that
overcommits does not get a CUDA out-of-memory error, it pushes the whole machine into swap and reclaim until
it stops answering (the kernel's OOM killer acts far too late, or not at all). memguard watches the machine
and, when memory runs out, kills the running benchmark's processes (not milabench itself), so milabench
records that benchmark as failed and goes on with the next one.

    python -m benchmate.memguard --pid <milabench pid> [--min-available-gb 6] [--psi-full 20] [--log FILE]

Triggers (checked every --interval seconds):
  * MemAvailable below --min-available-gb (default: 5% of RAM, at least 4 GiB)
  * memory pressure: the "full" stall share of /proc/pressure/memory over 10 s above --psi-full percent
    (every task stalled on memory: what precedes a freeze)

Kills: every process below --pid (its descendants), process group by process group, except memguard's own
and the ones of --pid itself; SIGTERM, then SIGKILL after --grace seconds. One JSON line per event in --log.
Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time


def meminfo() -> dict[str, int]:
    out = {}
    with open("/proc/meminfo") as f:
        for line in f:
            key, value = line.split(":", 1)
            out[key] = int(value.split()[0]) * 1024
    return out


def psi_full_avg10() -> float | None:
    try:
        with open("/proc/pressure/memory") as f:
            for line in f:
                if line.startswith("full"):
                    return float(dict(kv.split("=") for kv in line.split()[1:])["avg10"])
    except (OSError, KeyError, ValueError):
        return None
    return None


def children() -> dict[int, list[int]]:
    tree: dict[int, list[int]] = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat") as f:
                stat = f.read()
        except OSError:
            continue
        # comm may contain spaces and parentheses: the fields after the last ')'
        ppid = int(stat.rsplit(")", 1)[1].split()[1])
        tree.setdefault(ppid, []).append(int(entry))
    return tree


def descendants(root: int) -> list[int]:
    tree = children()
    out, todo = [], list(tree.get(root, []))
    while todo:
        pid = todo.pop()
        out.append(pid)
        todo.extend(tree.get(pid, []))
    return out


def rss(pid: int) -> int:
    try:
        with open(f"/proc/{pid}/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except OSError:
        return 0


def cmdline(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return f.read().replace(b"\0", b" ").decode(errors="replace").strip()[:300]
    except OSError:
        return ""


def kill_benchmark(root: int, grace: float) -> list[dict]:
    """SIGTERM then SIGKILL the process groups below root, except root's and ours."""
    protected = {os.getpgid(root), os.getpgrp()}
    pids = descendants(root)
    victims = [{"pid": p, "rss_gb": round(rss(p) / 2**30, 2), "cmd": cmdline(p)} for p in pids]
    groups = set()
    for pid in pids:
        try:
            groups.add(os.getpgid(pid))
        except ProcessLookupError:
            pass
    groups -= protected
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pgid in groups:
            try:
                os.killpg(pgid, sig)
            except (ProcessLookupError, PermissionError):
                pass
        # and every process seen, one by one: root's own group (no setsid), and the children of a parent
        # killed first, which are reparented out of the tree but keep their pid
        for pid in pids:
            if pid == os.getpid():
                continue
            try:
                os.kill(pid, sig)
            except (ProcessLookupError, PermissionError):
                pass
        if sig == signal.SIGTERM:
            deadline = time.time() + grace
            while time.time() < deadline and any(os.path.exists(f"/proc/{p}") for p in pids):
                time.sleep(0.2)
    return sorted(victims, key=lambda v: -v["rss_gb"])


def main(argv=None):
    total = meminfo()["MemTotal"]
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--pid", type=int, required=True, help="milabench's pid: its descendants are the benchmark")
    parser.add_argument("--min-available-gb", type=float,
                        default=float(os.environ.get("MEMGUARD_MIN_AVAILABLE_GB", max(4.0, 0.05 * total / 2**30))))
    parser.add_argument("--psi-full", type=float, default=float(os.environ.get("MEMGUARD_PSI_FULL", 20)),
                        help="kill when every task stalled on memory this share (%%) of the last 10 s")
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--grace", type=float, default=5.0)
    parser.add_argument("--cooldown", type=float, default=30.0, help="seconds between two kills")
    parser.add_argument("--log", default=os.environ.get("MEMGUARD_LOG"))
    args = parser.parse_args(argv)

    def log(event: dict):
        line = json.dumps({"time": time.time(), **event})
        print("[memguard]", line, flush=True)
        if args.log:
            with open(args.log, "a") as f:
                f.write(line + "\n")

    log({"event": "start", "pid": args.pid, "min_available_gb": args.min_available_gb, "psi_full": args.psi_full,
         "mem_total_gb": round(total / 2**30, 1)})
    last_kill = 0.0
    while os.path.exists(f"/proc/{args.pid}"):
        info = meminfo()
        available = info.get("MemAvailable", 0) / 2**30
        psi = psi_full_avg10()
        reason = None
        if available < args.min_available_gb:
            reason = f"MemAvailable {available:.1f} GiB < {args.min_available_gb:.1f} GiB"
        elif psi is not None and psi > args.psi_full:
            reason = f"memory pressure full avg10 {psi:.0f}% > {args.psi_full:.0f}%"
        if reason and time.time() - last_kill > args.cooldown and descendants(args.pid):
            victims = kill_benchmark(args.pid, args.grace)
            last_kill = time.time()
            log({"event": "kill", "reason": reason, "available_gb": round(available, 2), "psi_full_avg10": psi,
                 "swap_used_gb": round((info.get("SwapTotal", 0) - info.get("SwapFree", 0)) / 2**30, 2),
                 "victims": victims[:8]})
        time.sleep(args.interval)
    log({"event": "stop"})


if __name__ == "__main__":
    sys.exit(main())
