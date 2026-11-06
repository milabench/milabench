#!/usr/bin/env python
"""Engine-only inference server on a worker node of a multi-node data parallel run.

Started on every node but the main one (``milabench slurm srun -x <main>``) while main.py runs the
coordinator server and the client on the main node. The node finds its own index by the address it
uses to reach the coordinator, like benchmate.ray_node (``SLURM_NODEID`` depends on the srun flavour).

    server_worker.py --backend atom --nodes ip0,ip1 --dp-local 1 --pidfile F -- <server argv>
    server_worker.py --stop --pidfile F

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
from server_backends import build_server_command, multinode_dp_flags


def _alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


def _is_server(pid: int) -> bool:
    """The pid in a stale pidfile may have been reused by an unrelated process."""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            cmdline = f.read().decode(errors="replace")
    except OSError:
        return False
    return "atom_server_entry" in cmdline or "vllm" in cmdline


def stop(pidfile: str, grace: float = 30.0):
    try:
        with open(pidfile) as f:
            pgid = int(f.read().strip())
    except (OSError, ValueError):
        return
    if _alive(pgid) and _is_server(pgid):
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
    parser.add_argument("--backend", default="atom")
    parser.add_argument("--nodes", help="comma separated node addresses, the coordinator first")
    parser.add_argument("--dp-local", type=int, default=1)
    parser.add_argument("--pidfile", required=True)
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("server_argv", nargs=argparse.REMAINDER)
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

    server_argv = args.server_argv
    if server_argv and server_argv[0] == "--":
        server_argv = server_argv[1:]
    server_argv = [
        *server_argv,
        *multinode_dp_flags(
            args.backend, nnodes=len(nodes), dp_local=args.dp_local, node_index=index, master_ip=nodes[0]
        ),
    ]
    # ATOM's engine-only path (DP rank > 0) starts its Prometheus server without creating the multiprocess
    # directory it needs (only the coordinator path does); a value forwarded from the main node names a
    # directory of that node. Give the server a node-local one.
    if not os.path.isdir(os.environ.get("PROMETHEUS_MULTIPROC_DIR", "")):
        os.environ["PROMETHEUS_MULTIPROC_DIR"] = tempfile.mkdtemp(prefix="atom-metrics-")

    command = build_server_command(server_argv, backend=args.backend)
    print(f"[server_worker] node {index} ({me}):", " ".join(command), flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.pidfile)), exist_ok=True)
    proc = subprocess.Popen(command, start_new_session=True)
    with open(args.pidfile, "w") as f:
        f.write(str(proc.pid))

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
