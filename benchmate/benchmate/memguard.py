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
    (every task stalled on memory: what precedes a freeze); MEMGUARD_PSI_SCOPE=cgroup reads memguard's own
    cgroup instead, so containers thrashing against their own memory caps do not count (see psi_path)
  * the NVIDIA driver reporting a failed allocation or an Xid error in the kernel log (journalctl -k -f;
    --no-kernel-log to disable). On a GB10 the driver ran out of memory for its own allocations at a vLLM
    server's start with ~20 GiB still available, and the machine froze 10 minutes later: neither of the
    triggers above sees that coming. An out-of-memory message only kills while MemAvailable is below
    --driver-oom-gb (default 32, MEMGUARD_DRIVER_OOM_GB); above it, it is logged as a warning: llama.cpp hit
    one while loading and recovered (Step-3.5-Flash served fine), and one came at a load start with 105 GiB
    available (2026-11-25). An Xid error always kills.

Page cache (--evict-paths, default MILABENCH_EVICT_PAGE_CACHE): on a GB10 the driver can't reclaim the page
cache for its allocations, so MemAvailable (which counts the cache) overstates what the GPU can still get: the
machine froze with 89 GiB used and 26 GiB of model files cached. When MemFree drops below --min-free-gb,
memguard advises the kernel to drop the cached pages of those folders (posix_fadvise DONTNEED, no root
needed), at most every 15 s. Pages a process maps (mmapped weights) stay.

Kills: every process below --pid (its descendants), process group by process group, except memguard's own
and the ones of --pid itself; SIGTERM, then SIGKILL after --grace seconds. One JSON line per event in --log.
Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time


def meminfo() -> dict[str, int]:
    out = {}
    with open("/proc/meminfo") as f:
        for line in f:
            key, value = line.split(":", 1)
            out[key] = int(value.split()[0]) * 1024
    return out


def psi_path() -> str:
    """Where memory pressure is read. MEMGUARD_PSI_SCOPE=cgroup: memguard's own cgroup (cgroup v2), the processes
    it guards and not the machine. Containers run in cgroups of their own: Terminal-Bench tasks capped at 2 GB
    stall in reclaim against their own cap, and the machine-wide figure then reads 20% "full" with 55 GB still
    available, which killed healthy model servers (2026-11-25). MemAvailable and the driver's errors still
    watch the whole machine. Default (unset or "system"): /proc/pressure/memory, as before."""
    if os.environ.get("MEMGUARD_PSI_SCOPE", "system") == "cgroup":
        try:
            with open("/proc/self/cgroup") as f:
                path = next(line.split(":", 2)[2].strip() for line in f if line.startswith("0::"))
            candidate = f"/sys/fs/cgroup{path}/memory.pressure"
            if os.path.exists(candidate):
                return candidate
        except (OSError, StopIteration):
            pass
    return "/proc/pressure/memory"


PSI_PATH = psi_path()


def psi_full_avg10() -> float | None:
    try:
        with open(PSI_PATH) as f:
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


# NVIDIA driver messages after which the GPU, and on unified memory the whole machine, is in trouble
DRIVER_OOM = re.compile(r"NVRM:.*(NV_ERR_NO_MEMORY|Out of memory)")
KERNEL_PATTERNS = (DRIVER_OOM, re.compile(r"NVRM: Xid"))


class KernelLog:
    """The kernel log's new lines, read in a thread from ``journalctl -k -f``."""

    def __init__(self):
        self.lines: queue.Queue[str] = queue.Queue()
        self.proc = None
        if shutil.which("journalctl") is None:
            return
        try:
            self.proc = subprocess.Popen(["journalctl", "-k", "-f", "-n", "0", "-o", "cat"],
                                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        except OSError:
            self.proc = None
            return
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.proc.stdout:
            self.lines.put(line.rstrip())

    def alarms(self) -> list[str]:
        out = []
        while True:
            try:
                line = self.lines.get_nowait()
            except queue.Empty:
                return out
            if any(p.search(line) for p in KERNEL_PATTERNS):
                out.append(line)

    def close(self):
        if self.proc is not None:
            self.proc.terminate()


def evict_page_cache(paths: str) -> int:
    """posix_fadvise(DONTNEED) on every file under the ``:``-separated folders; returns the bytes advised."""
    advised = 0
    for root in filter(None, paths.split(":")):
        for folder, _, files in os.walk(root):
            for name in files:
                path = os.path.join(folder, name)
                if os.path.islink(path):
                    continue
                try:
                    fd = os.open(path, os.O_RDONLY)
                except OSError:
                    continue
                try:
                    advised += os.fstat(fd).st_size
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                except OSError:
                    pass
                finally:
                    os.close(fd)
    return advised


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
    parser.add_argument("--no-kernel-log", action="store_true", help="do not watch the NVIDIA driver's kernel messages")
    parser.add_argument("--evict-paths", default=os.environ.get("MILABENCH_EVICT_PAGE_CACHE", ""),
                        help="':'-separated folders whose page cache is dropped when MemFree runs low")
    parser.add_argument("--min-free-gb", type=float,
                        default=float(os.environ.get("MEMGUARD_MIN_FREE_GB", max(8.0, 0.07 * total / 2**30))))
    parser.add_argument("--driver-oom-gb", type=float, default=float(os.environ.get("MEMGUARD_DRIVER_OOM_GB", 32)),
                        help="a driver out-of-memory message kills only while MemAvailable is below this (GiB)")
    args = parser.parse_args(argv)

    def log(event: dict):
        line = json.dumps({"time": time.time(), **event})
        print("[memguard]", line, flush=True)
        if args.log:
            with open(args.log, "a") as f:
                f.write(line + "\n")

    log({"event": "start", "pid": args.pid, "min_available_gb": args.min_available_gb, "psi_full": args.psi_full,
         "psi_source": PSI_PATH, "driver_oom_gb": args.driver_oom_gb,
         "mem_total_gb": round(total / 2**30, 1)})
    kernel = None if args.no_kernel_log else KernelLog()
    if kernel is not None and kernel.proc is None:
        log({"event": "warning", "message": "kernel log unavailable (journalctl): driver errors are not watched"})
    last_kill = last_evict = 0.0
    while os.path.exists(f"/proc/{args.pid}"):
        info = meminfo()
        free = info.get("MemFree", 0) / 2**30
        if args.evict_paths and free < args.min_free_gb and time.time() - last_evict > 15:
            cached = info.get("Cached", 0) / 2**30
            evict_page_cache(args.evict_paths)
            last_evict = time.time()
            info = meminfo()
            log({"event": "evict", "free_gb": round(free, 2), "cached_gb": round(cached, 2),
                 "free_after_gb": round(info.get("MemFree", 0) / 2**30, 2),
                 "cached_after_gb": round(info.get("Cached", 0) / 2**30, 2)})
        available = info.get("MemAvailable", 0) / 2**30
        psi = psi_full_avg10()
        reason = None
        driver = kernel.alarms() if kernel is not None else []
        if driver and available >= args.driver_oom_gb and all(DRIVER_OOM.search(d) for d in driver):
            log({"event": "warning", "message": f"NVIDIA driver: {driver[0][:200]}", "count": len(driver),
                 "available_gb": round(available, 2), "action": f"none: MemAvailable >= {args.driver_oom_gb:g} GiB"})
            driver = []
        if driver:
            reason = f"NVIDIA driver: {driver[0][:200]}"
        elif available < args.min_available_gb:
            reason = f"MemAvailable {available:.1f} GiB < {args.min_available_gb:.1f} GiB"
        elif psi is not None and psi > args.psi_full:
            reason = f"memory pressure full avg10 {psi:.0f}% > {args.psi_full:.0f}%"
        if reason and (driver or time.time() - last_kill > args.cooldown) and descendants(args.pid):
            victims = kill_benchmark(args.pid, args.grace)
            last_kill = time.time()
            log({"event": "kill", "reason": reason, "available_gb": round(available, 2), "psi_full_avg10": psi,
                 "swap_used_gb": round((info.get("SwapTotal", 0) - info.get("SwapFree", 0)) / 2**30, 2),
                 "victims": victims[:8]})
        time.sleep(args.interval)
    if kernel is not None:
        kernel.close()
    log({"event": "stop"})


if __name__ == "__main__":
    sys.exit(main())
