"""Engine-only inference server on a worker node of a multi-node data parallel run.

Started on every node but the main one (``milabench slurm srun -x <main>``) while milabench runs the
coordinator server on the main node. The command line is built by milabench (milabench.commands.serving),
this script only knows the node's place in the group: it finds its own index by the address it uses to
reach the coordinator, like benchmate.ray_node (``SLURM_NODEID`` depends on the srun flavour), and
replaces ``@NODE_INDEX@`` / ``@DP_RANK@`` in the command.

    python -m benchmate.server_worker --nodes ip0,ip1 --dp-local 1 --pidfile F -- <server command>
    python -m benchmate.server_worker --stop --pidfile F

The server runs in its own session and its process group id goes to ``--pidfile``: ``--stop`` kills
exactly that group (the ssh fallback of srun does not forward signals to the remote command).
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import tempfile
import time

from benchmate.ray_node import ip_towards


def _alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


def _is_server(pid: int, marker: str) -> bool:
    """The pid in a stale pidfile may have been reused by an unrelated process."""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            cmdline = f.read().decode(errors="replace")
    except OSError:
        return False
    return marker in cmdline


def _marker(command: list[str]) -> str:
    """What identifies the server in its cmdline: the script or module, not the python interpreter"""
    for arg in command:
        if arg == "-m" or os.path.basename(arg).startswith("python"):
            continue
        return arg
    return command[0]


def stop(pidfile: str, grace: float = 30.0):
    try:
        with open(pidfile) as f:
            pgid, marker = f.read().split("\n", 1)
            pgid = int(pgid)
    except (OSError, ValueError):
        return
    if _alive(pgid) and _is_server(pgid, marker.strip()):
        for sig, wait in ((signal.SIGTERM, grace), (signal.SIGKILL, 10.0)):
            try:
                os.killpg(pgid, sig)
            except ProcessLookupError:
                break
            deadline = time.time() + wait
            while _alive(pgid) and time.time() < deadline:
                time.sleep(0.5)
            if not _alive(pgid):
                break
        print(f"[server_worker] stopped process group {pgid}", flush=True)
    try:
        os.unlink(pidfile)
    except OSError:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", help="comma separated node addresses, the coordinator first")
    parser.add_argument("--dp-local", type=int, default=1)
    parser.add_argument("--pidfile", required=True)
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    # a server left over by an interrupted run would hold the GPUs and the ports
    stop(args.pidfile)
    if args.stop:
        return 0

    nodes = args.nodes.split(",")
    me = ip_towards(nodes[0])
    if me not in nodes:
        raise SystemExit(f"[server_worker] this node ({me}) is not in {nodes}")
    index = nodes.index(me)

    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    placeholders = {"@NODE_INDEX@": str(index), "@DP_RANK@": str(index * args.dp_local)}
    for key, value in placeholders.items():
        command = [arg.replace(key, value) for arg in command]

    # ATOM's engine-only path (DP rank > 0) starts its Prometheus server without creating the multiprocess
    # directory it needs (only the coordinator path does); a value forwarded from the main node names a
    # directory of that node. Give the server a node-local one.
    if not os.path.isdir(os.environ.get("PROMETHEUS_MULTIPROC_DIR", "")):
        os.environ["PROMETHEUS_MULTIPROC_DIR"] = tempfile.mkdtemp(prefix="server-metrics-")

    print(f"[server_worker] node {index} ({me}):", " ".join(command), flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.pidfile)), exist_ok=True)
    proc = subprocess.Popen(command, start_new_session=True)
    with open(args.pidfile, "w") as f:
        f.write(f"{proc.pid}\n{_marker(command)}")

    def forward(signum, _frame):
        try:
            os.killpg(proc.pid, signum)
        except ProcessLookupError:
            pass

    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)
    return proc.wait()


if __name__ == "__main__":
    sys.exit(main())
