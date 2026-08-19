#!/usr/bin/env python
"""torchrun + srun smoke test: all-reduce latency / bandwidth across nodes.

Uses torchcompat for the current device (CPU via plugin fallback, or GPU).
Runs timed ``all_reduce`` iterations and logs latency + bandwidth so rank 0
still feeds metrics to milabench.
"""

from __future__ import annotations

import argparse
import os
import socket
import time

import torch
import torch.distributed as dist
import torchcompat.core as accelerator
from benchmate.monitor import setupvoir


GIB = 1024**3


def bus_bandwidth_GBps(nbytes: int, elapsed_s: float, world: int) -> float:
    """NVIDIA-style bus bandwidth for all_reduce (ring factor ``2*(n-1)/n``)."""
    if elapsed_s <= 0 or world <= 1:
        return 0.0
    factor = 2.0 * (world - 1) / world
    return (nbytes * factor) / elapsed_s / GIB


def measure_allreduce(tensor: torch.Tensor, repeats: int, warmup: int):
    """Time ``repeats`` all_reduce calls; return per-iter elapsed seconds."""
    for _ in range(warmup):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        accelerator.synchronize()

    times = []
    for _ in range(repeats):
        accelerator.synchronize()
        t0 = time.perf_counter()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        accelerator.synchronize()
        times.append(time.perf_counter() - t0)
    return times


def main():
    parser = argparse.ArgumentParser(description="torchsrun all-reduce bandwidth/latency")
    parser.add_argument("--repeats", type=int, default=30, help="Timed all_reduce iterations")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument(
        "--size-mb",
        type=float,
        default=64.0,
        help="Payload size for bandwidth all_reduce (MiB)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="Distributed backend (default: torchcompat device CCL)",
    )
    args = parser.parse_args()

    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = accelerator.fetch_device(local_rank)
    backend = args.backend or accelerator.ccl

    accelerator.init_process_group(backend=backend)

    rank = dist.get_rank()
    world = dist.get_world_size()
    host = socket.gethostname()

    print(
        f"[torchsrun] host={host} rank={rank}/{world} local_rank={local_rank} "
        f"device={device} backend={backend} SLURM_NODEID={os.getenv('SLURM_NODEID', '?')}",
        flush=True,
    )

    latency_tensor = torch.ones(1, dtype=torch.float32, device=device)
    latency_times = measure_allreduce(
        latency_tensor, repeats=args.repeats, warmup=args.warmup
    )

    nelem = max(int(args.size_mb * 1024 * 1024 // 4), 1)
    bw_tensor = torch.ones(nelem, dtype=torch.float32, device=device)
    nbytes = bw_tensor.nelement() * bw_tensor.element_size()
    bw_times = measure_allreduce(bw_tensor, repeats=args.repeats, warmup=args.warmup)

    if rank == 0:
        log, monitor = setupvoir(monogpu=False, interval=0.5)
        print(
            f"[torchsrun] measuring {args.repeats} iters, "
            f"payload={nbytes / (1024 * 1024):.1f} MiB",
            flush=True,
        )

        for i, (lat_s, bw_s) in enumerate(zip(latency_times, bw_times)):
            latency_us = lat_s * 1e6
            bandwidth = bus_bandwidth_GBps(nbytes, bw_s, world)
            print(
                f"[torchsrun] iter={i:02d} latency={latency_us:.1f} us "
                f"bandwidth={bandwidth:.3f} GiB/s",
                flush=True,
            )
            log(
                {
                    "task": "train",
                    "rate": bandwidth,
                    "units": "GiB/s",
                    "latency_us": latency_us,
                    "size_bytes": nbytes,
                    "world_size": world,
                    "device": str(device),
                    "iter": i,
                    "time": time.time(),
                }
            )

        monitor.stop()
        lat_avg = sum(latency_times) / len(latency_times) * 1e6
        bw_avg = sum(bus_bandwidth_GBps(nbytes, t, world) for t in bw_times) / len(bw_times)
        print(
            f"[torchsrun] done avg_latency={lat_avg:.1f} us "
            f"avg_bandwidth={bw_avg:.3f} GiB/s world={world} device={device}",
            flush=True,
        )

    dist.barrier()
    accelerator.destroy_process_group()


if __name__ == "__main__":
    main()
