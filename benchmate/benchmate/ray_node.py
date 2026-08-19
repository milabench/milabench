#!/usr/bin/env python3
"""rayrun: bring up this node's role (head or worker) in a symmetric Ray
cluster and block for the cluster's lifetime.

Every node (head and workers alike) runs this via a single symmetric
``milabench slurm srun -- rayrun ...`` call -- head and workers are tasks of
the *same* srun step. An earlier design started the head as a bare local
background process outside any srun step and only dispatched workers via a
separate ``srun -x main``; that asymmetric split was found to make
freshly-joined worker nodes get marked dead by Ray's GCS health check within
seconds of joining, reproducibly, across many runs (see milabench's
ray_smoke diagnostic benchmark). This symmetric version, validated
standalone, did not reproduce that failure.

Role is decided by comparing this node's own hostname against the given
head hostname -- not by ``$SLURM_NODEID``, which is not guaranteed to put
the designated main node at rank 0, and not by IP-matching against
``hostname -I`` (tried first; unreliable in practice -- on a multi-homed
node it can miss the interface milabench actually recorded, and silently
swallowing subprocess errors made every node fall back to "worker", head
included, with no diagnostic). A single hostname string comparison has
none of that ambiguity.

Both roles run ``ray start ... --block`` as *this process itself* (blocked
on directly on the head, execve'd into on workers) -- deliberately not
backgrounded/detached. This is meant to stay alive for the whole cluster's
lifetime: the caller runs it concurrently with a wait/workload/``ray stop``
sequence, so a plain ``ray stop`` (run separately, once per node, as the
sequence's last step) kills the local raylet/GCS this process is blocked
on, which makes it return on its own -- no signal-forwarding or
process-handle bookkeeping needed on the caller's side.
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time


def is_head(head_hostname: str) -> bool:
    return socket.gethostname() == head_hostname


def wait_ray_ready(ray_bin: str, address: str, timeout_s: int) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        result = subprocess.run(
            [ray_bin, "status", "--address", address],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode == 0:
            return True
        time.sleep(2)
    return False


def run_head(ray_bin: str, head_ip: str, port: int, marker: str, timeout_s: int) -> int:
    print(f"[rayrun] this node ({head_ip}) is head, starting ray --block", flush=True)
    if os.path.exists(marker):
        os.remove(marker)

    cmd = [ray_bin, "start", "--head", f"--node-ip-address={head_ip}", f"--port={port}", "--block"]
    print("[rayrun] +", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd)

    if not wait_ray_ready(ray_bin, f"{head_ip}:{port}", timeout_s):
        print(f"[rayrun] ERROR: head never became reachable after {timeout_s}s", file=sys.stderr)
        proc.terminate()
        return 1

    os.makedirs(os.path.dirname(marker), exist_ok=True)
    with open(marker, "w", encoding="utf-8") as f:
        f.write(f"{head_ip}:{port}\n")
    print(f"[rayrun] head ready (pid {proc.pid}), wrote {marker}", flush=True)

    # Block here for the cluster's whole lifetime. `ray stop` (run later,
    # once per node, as its own separate step) kills the raylet/GCS this is
    # waiting on, so this returns on its own once the cluster is torn down.
    return proc.wait()


def run_worker(ray_bin: str, head_ip: str, port: int, marker: str, timeout_s: int) -> int:
    print(f"[rayrun] waiting for head marker at {marker}", flush=True)
    waited = 0
    while not os.path.exists(marker):
        if waited >= timeout_s:
            print(f"[rayrun] ERROR: head marker never appeared after {timeout_s}s", file=sys.stderr)
            return 1
        time.sleep(1)
        waited += 1

    print(f"[rayrun] head marker seen after {waited}s, joining {head_ip}:{port} --block", flush=True)
    cmd = [ray_bin, "start", f"--address={head_ip}:{port}", "--block"]
    print("[rayrun] +", " ".join(cmd), flush=True)
    # Nothing runs after this, so replace this process with ray's instead of
    # leaving a python wrapper sitting on top of it -- same effect as
    # `proc.wait()` on the head (blocks for the cluster's lifetime, returns
    # once locally killed by `ray stop`), one less layer.
    os.execvp(cmd[0], cmd)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ray-bin", required=True, help="Path to the ray executable")
    parser.add_argument("--head-ip", required=True, help="Head node IP address (for ray's own bind/connect)")
    parser.add_argument("--head-hostname", required=True, help="Head node hostname (for role detection)")
    parser.add_argument("--port", type=int, required=True, help="Ray GCS port")
    parser.add_argument(
        "--ready-marker", required=True, help="Shared-filesystem path signaling head readiness"
    )
    parser.add_argument(
        "--timeout", type=int, default=600, help="Seconds to wait for readiness (default: 600)"
    )
    args = parser.parse_args(argv)

    if is_head(args.head_hostname):
        return run_head(args.ray_bin, args.head_ip, args.port, args.ready_marker, args.timeout)
    return run_worker(args.ray_bin, args.head_ip, args.port, args.ready_marker, args.timeout)


if __name__ == "__main__":
    sys.exit(main())
