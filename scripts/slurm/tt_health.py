#!/usr/bin/env python
"""GPU and NCCL checks used by tt_healthcheck.sh (runs in the bench venv).

    tt_health.py matmul                  # per local GPU: bf16 matmul TFLOPS
    tt_health.py soak [SECONDS]          # all local GPUs at once, sustained: TFLOPS,
                                         # SM clock, power, temperature, throttle
                                         # reasons per 10 s window
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


def soak(seconds: float = 600, window: float = 10, n: int = 8192) -> None:
    """Sustained load on every local GPU at once (as in training), so power and
    thermal limits show up: the short matmul check loads one GPU for ~0.1 s."""
    import subprocess
    import threading

    host = socket.gethostname().split(".")[0]
    devices = range(torch.cuda.device_count())
    done = {dev: 0 for dev in devices}  # matmuls finished, per GPU
    stop = threading.Event()

    def work(dev: int) -> None:
        torch.cuda.set_device(dev)
        a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
        while not stop.is_set():
            for _ in range(10):
                a @ b
            torch.cuda.synchronize()
            done[dev] += 10

    threads = [threading.Thread(target=work, args=(dev,), daemon=True) for dev in devices]
    for thread in threads:
        thread.start()
    query = "index,clocks.sm,power.draw,temperature.gpu,clocks_event_reasons.active"
    start = last = time.perf_counter()
    before = dict(done)
    while last - start < seconds:
        time.sleep(window)
        now = time.perf_counter()
        smi = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True,
        ).stdout.splitlines()
        for dev in devices:
            tflops = 2 * n**3 * (done[dev] - before[dev]) / (now - last) / 1e12
            idx, sm, power, temp, reasons = (f.strip() for f in smi[dev].split(","))
            print(
                f"[tt-health] soak host={host} gpu={dev} t={now - start:.0f} tflops={tflops:.0f} "
                f"sm_mhz={sm} power_w={power} temp_c={temp} reasons={reasons}",
                flush=True,
            )
        before, last = dict(done), now
    stop.set()
    for thread in threads:
        thread.join()


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
    command, *args = sys.argv[1:]
    {"matmul": matmul, "allreduce": allreduce, "soak": soak}[command](*map(float, args))
