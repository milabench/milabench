#!/usr/bin/env python
"""GPU and NCCL checks used by tt_healthcheck.sh (runs in the bench venv).

    tt_health.py matmul                  # per local GPU: bf16 matmul TFLOPS
    torchrun ... tt_health.py allreduce  # all-reduce bus bandwidth (GB/s)

Prints one parsable line per result: "[tt-health] <key>=<value> ...".
"""

import os
import socket
import sys
import time

import torch


def matmul(n: int = 8192, iters: int = 50) -> None:
    host = socket.gethostname().split(".")[0]
    for dev in range(torch.cuda.device_count()):
        torch.cuda.set_device(dev)
        a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
        for _ in range(5):
            a @ b
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            a @ b
        torch.cuda.synchronize()
        tflops = 2 * n**3 * iters / (time.perf_counter() - start) / 1e12
        print(f"[tt-health] matmul host={host} gpu={dev} tflops={tflops:.0f}", flush=True)


def allreduce(size_mb: int = 1024, iters: int = 20) -> None:
    import torch.distributed as dist

    dist.init_process_group("nccl")
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    world = dist.get_world_size()
    x = torch.ones(size_mb * 2**20 // 2, dtype=torch.bfloat16, device="cuda")
    for _ in range(5):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / iters
    # NCCL-tests convention: bus bandwidth = algbw * 2 (n - 1) / n.
    busbw = x.numel() * x.element_size() / elapsed * 2 * (world - 1) / world / 1e9
    if dist.get_rank() == 0:
        nodes = int(os.environ.get("TT_HEALTH_NODES", "1"))
        print(
            f"[tt-health] allreduce scope={os.environ.get('TT_HEALTH_SCOPE', '?')} "
            f"nodes={nodes} ranks={world} size_mb={size_mb} busbw_gbs={busbw:.1f}",
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    {"matmul": matmul, "allreduce": allreduce}[sys.argv[1]]()
