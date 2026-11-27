"""Kill leftover milabench / benchmark / Ray processes on every system node.

``milabench tools warden`` reads ``$MILABENCH_SYSTEM`` and runs this module
with ``--node`` on each node (locally on the current one, over SSH on the
others). The node side finds the current user's leftovers from ``/proc`` and
the processes using a GPU through ``benchmate.warden.GPUProcessWarden``.

A first pass only lists the processes on every node; if a ``milabench
run/prepare/install`` is still alive somewhere, nothing is killed unless
``--force`` is given.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from argklass.command import Command

# Process titles rewritten with setproctitle (no executable path left to match)
TITLE_PREFIXES = ("ray::", "VLLM::", "VllmWorker", "EngineCore")
TITLE_NAMES = {"raylet", "gcs_server"}

ACTIVE_MILABENCH = re.compile(r"\bmilabench\s+(run|prepare|install)\b")


#
# Node side
#


def _cmdline(pid: int) -> list[str]:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            raw = f.read()
    except OSError:
        return []
    return [a.decode("utf8", "replace") for a in raw.split(b"\0") if a]


def _ppid(pid: int) -> int:
    try:
        with open(f"/proc/{pid}/stat") as f:
            # comm can contain spaces/parens: ppid is the 2nd field after the last ')'
            return int(f.read().rsplit(")", 1)[1].split()[1])
    except (OSError, IndexError, ValueError):
        return 0


def _uid(pid: int) -> int:
    try:
        return os.stat(f"/proc/{pid}").st_uid
    except OSError:
        return -1


def _alive(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat") as f:
            state = f.read().rsplit(")", 1)[1].split()[0]
        return state != "Z"
    except (OSError, IndexError):
        return False


def _ancestors(pid: int) -> set[int]:
    seen = set()
    while pid > 1 and pid not in seen:
        seen.add(pid)
        pid = _ppid(pid)
    return seen


def _leftover_reason(argv: list[str], roots: list[str]) -> Optional[str]:
    """Why this command line belongs to milabench, or None"""
    if not argv:
        return None

    first = argv[0]
    if first.startswith(TITLE_PREFIXES) or first.split()[0] in TITLE_NAMES:
        return "title"

    # Only the executable and the script are matched, so that unrelated
    # commands which merely mention a path (tail -f <run>.data) are spared
    for arg in argv[:2]:
        if any(arg.startswith(root) for root in roots):
            return "path"
        if os.path.basename(arg) == "milabench":
            return "milabench"

    # ssh launched by `milabench slurm srun`
    if os.path.basename(first) == "ssh":
        joined = " ".join(argv)
        if any(root in joined for root in roots):
            return "ssh"

    return None


def _gpu_processes() -> dict[int, str]:
    from benchmate.warden import GPUProcessWarden

    warden = GPUProcessWarden(kill_on_start=False, kill_on_end=False)
    found = {}
    for proc in warden.fetch_processes():
        try:
            found[int(proc.pid)] = proc.process_name or ""
        except (TypeError, ValueError):
            pass
    return found


def find_leftovers(roots: list[str], gpu: bool = True) -> list[dict]:
    me = os.getuid()
    spared = _ancestors(os.getpid())
    gpu_pids = _gpu_processes() if gpu else {}

    found = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue

        pid = int(entry)
        if pid in spared or not _alive(pid):
            continue

        argv = _cmdline(pid)
        reason = None
        if _uid(pid) == me:
            reason = _leftover_reason(argv, roots)
        if reason is None and pid in gpu_pids:
            reason = "gpu"
        if reason is None:
            continue

        found.append({
            "pid": pid,
            "uid": _uid(pid),
            "reason": reason,
            "cmd": " ".join(argv) or gpu_pids.get(pid, ""),
        })
    return found


def kill_processes(pids: list[int], timeout: float) -> list[int]:
    """SIGTERM, wait up to ``timeout``, then SIGKILL. Returns the survivors"""

    def send(pids, sig):
        for pid in pids:
            try:
                os.kill(pid, sig)
            except (ProcessLookupError, PermissionError):
                pass

    def wait(pids, delay):
        deadline = time.time() + delay
        while (pids := [p for p in pids if _alive(p)]) and time.time() < deadline:
            time.sleep(0.2)
        return pids

    send(pids, signal.SIGTERM)
    remaining = wait(pids, timeout)
    send(remaining, signal.SIGKILL)
    return wait(remaining, 5)


def node_main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Find/kill milabench leftovers on this node")
    parser.add_argument("--node", action="store_true")
    parser.add_argument("--root", action="append", default=[], help="Path prefix of milabench processes")
    parser.add_argument("--kill", action="store_true")
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--timeout", type=float, default=10)
    args = parser.parse_args(argv)

    processes = find_leftovers(args.root, gpu=not args.no_gpu)
    survivors = []
    if args.kill and processes:
        survivors = kill_processes([p["pid"] for p in processes], args.timeout)

    print(json.dumps({"processes": processes, "survivors": survivors}))


#
# Coordinator side
#


def _roots() -> list[str]:
    roots = []
    base = os.environ.get("MILABENCH_BASE")
    if base:
        roots.append(str(Path(base).absolute()))

    # milabench checkout: benchmarks/*/main.py, atom_server_entry.py, ...
    code = Path(__file__).resolve().parents[3] / "benchmarks"
    roots.append(str(code) + "/")
    return roots


def _node_command(roots, kill, gpu, timeout) -> list[str]:
    # The source is sent on stdin (see _run_on_node) so that workers whose
    # milabench checkout is stale still run this version
    cmd = [sys.executable, "-", "--node", "--timeout", str(timeout)]
    for root in roots:
        cmd += ["--root", root]
    if kill:
        cmd.append("--kill")
    if not gpu:
        cmd.append("--no-gpu")
    return cmd


def _run_on_node(node, command, sshkey) -> dict:
    from ..slurm.srun import _ssh_argv

    label = node.get("name") or node["ip"]
    argv = command if node.get("local") else _ssh_argv(node, command, sshkey=sshkey)

    try:
        proc = subprocess.run(
            argv, input=Path(__file__).read_text(), capture_output=True, text=True, timeout=300
        )
    except subprocess.TimeoutExpired:
        return {"node": label, "error": "timed out"}

    # GPU backends may print to stdout; the report is the last line
    lines = proc.stdout.strip().splitlines()
    try:
        report = json.loads(lines[-1])
    except (IndexError, json.JSONDecodeError):
        err = (proc.stderr.strip().splitlines() or [f"exit code {proc.returncode}"])[-1]
        return {"node": label, "error": err}

    report["node"] = label
    return report


def _run_everywhere(nodes, command, sshkey) -> list[dict]:
    with ThreadPoolExecutor(max_workers=max(1, len(nodes))) as pool:
        return list(pool.map(lambda n: _run_on_node(n, command, sshkey), nodes))


def _print_report(reports, verb):
    for report in reports:
        node = report["node"]
        if "error" in report:
            print(f"[{node}] error: {report['error']}")
            continue

        processes = report["processes"]
        if not processes:
            print(f"[{node}] clean")
            continue

        print(f"[{node}] {verb} {len(processes)} process(es)")
        for p in processes:
            print(f"  {p['pid']:>8}  {p['reason']:<9}  {p['cmd'][:160]}")

        for pid in report.get("survivors", []):
            print(f"  {pid:>8}  STILL ALIVE (other user, or unkillable)")


class Warden(Command):
    """Kill leftover milabench, benchmark and Ray processes on every system node."""

    name = "warden"

    # fmt: off
    @dataclass
    class Arguments:
        """Kill leftover milabench/benchmark/Ray processes on all the nodes of $MILABENCH_SYSTEM."""
        dry_run : bool          = False                      # Only list the processes
        force   : bool          = False                      # Kill even if a milabench run/prepare/install is still active
        no_gpu  : bool          = False                      # Do not kill processes using the GPUs (benchmate warden)
        timeout : float         = 10                         # Seconds between SIGTERM and SIGKILL
        system  : Optional[str] = None                       # System config with the node list (default: $MILABENCH_SYSTEM)
    # fmt: on

    @staticmethod
    def execute(args):
        from ...system import build_system_config

        path = args.system or os.getenv("MILABENCH_SYSTEM")
        if not path:
            print("error: pass --system or set MILABENCH_SYSTEM", file=sys.stderr)
            return 2

        system = build_system_config(path, gpu=False).get("system") or {}
        nodes = system.get("nodes") or []
        if not nodes:
            print(f"error: no nodes defined in system config: {path}", file=sys.stderr)
            return 2

        roots = _roots()
        sshkey = system.get("sshkey")
        gpu = not args.no_gpu

        listing = _run_everywhere(nodes, _node_command(roots, False, gpu, args.timeout), sshkey)
        _print_report(listing, "found")

        found = sum(len(r.get("processes", [])) for r in listing)
        errors = sum("error" in r for r in listing)

        if args.dry_run or found == 0:
            return 1 if errors else 0

        active = [
            (r["node"], p)
            for r in listing
            for p in r.get("processes", [])
            if ACTIVE_MILABENCH.search(p["cmd"])
        ]
        if active and not args.force:
            print("\nA milabench command is still running, use --force to kill it anyway:")
            for node, p in active:
                print(f"  [{node}] {p['pid']}  {p['cmd'][:160]}")
            return 3

        print()
        killed = _run_everywhere(nodes, _node_command(roots, True, gpu, args.timeout), sshkey)
        _print_report(killed, "killed")

        survivors = sum(len(r.get("survivors", [])) for r in killed)
        errors += sum("error" in r for r in killed)
        return 1 if survivors or errors else 0


COMMANDS = Warden


if __name__ == "__main__":
    node_main()
