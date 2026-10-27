"""CPU-only stand-in for the per-rank wandb runs of the GPU benchmark (no model, no GPU).

Each Slurm task is one rank. It names its run as the benchmark does
(tt_native.instrument.set_wandb_env, per-rank mode), opens TorchTitan's own
WandBLogger and logs, every ``log_freq`` steps, the keys a real run logs
(TorchTitan's and ours) with synthetic values, then the same config (worker
identity, run block) and summary keys. It also writes the per-rank step files
(metrics.<rank>.jsonl). Two anomalies to find in the dashboards:

    DUMMY_SLOW_DATA_RANK   rank with data-loader stalls (default 5)
    DUMMY_HEAVY_NODE       node index using ~2 GiB more memory (default 2)

Step times are the same on every rank (FSDP: every rank waits for the
slowest), and stalls on the slow rank lengthen every rank's step.
Rank 0 prints init, barrier and finish times per rank.

    srun --ntasks-per-node=4 python tt_wandb_ranks_test.py [steps] [pause_s]
"""

import json
import math
import os
import random
import socket
import sys
import time

import torch.distributed as dist

SKIP = 30  # warm-up steps, as the window configs
LOG_FREQ = 10
TOKENS_PER_GPU = 3 * 4096
FLOPS_PER_TOKEN = 30.7e9  # ~ 16-layer Qwen3.8-27B, gives ~38 % MFU at 1.47 s
PEAK = 989e12


def step_times(steps: int, slow_rank_waits: list[float]) -> list[float]:
    """Shared step times: compile on step 1, small noise, plus the slow rank's stalls."""
    rng = random.Random(1234)
    times = []
    for step in range(1, steps + 1):
        base = 95.0 if step == 1 else 1.47 + rng.gauss(0, 0.004)
        times.append(base + max(slow_rank_waits[step - 1] - 0.05, 0.0))
    return times


def data_waits(rank: int, slow_rank: int, steps: int) -> list[float]:
    rng = random.Random(rank)
    waits = []
    for step in range(1, steps + 1):
        wait = abs(rng.gauss(0.03, 0.01))
        if rank == slow_rank and (step % 17 == 0 or 120 <= step < 140):
            wait = rng.uniform(0.4, 1.4)
        waits.append(wait)
    return waits


def main(steps: int = 230, pause: float = 0.2) -> None:
    rank = int(os.environ["SLURM_PROCID"])
    local_rank = int(os.environ["SLURM_LOCALID"])
    os.environ.setdefault("RANK", str(rank))
    os.environ.setdefault("LOCAL_RANK", str(local_rank))
    os.environ["BENCHMATE_WANDB_ALL_RANKS"] = "1"
    host = socket.gethostname().split(".")[0]
    world = int(os.environ["SLURM_NTASKS"])
    node_index = int(os.environ.get("SLURM_NODEID", 0))
    slow_rank = int(os.environ.get("DUMMY_SLOW_DATA_RANK", 5))
    heavy_node = int(os.environ.get("DUMMY_HEAVY_NODE", 2))

    from tt_native import instrument
    from torchtitan.observability.metrics import WandBLogger

    instrument.set_wandb_env()
    dist.init_process_group(
        "gloo", init_method=f"tcp://{os.environ['MASTER_ADDR']}:29511", rank=rank, world_size=world
    )

    worker = {
        "rank": rank,
        "local_rank": local_rank,
        "host": host,
        "cuda_device": local_rank,
        "gpu_uuid": f"GPU-dummy-{host}-{local_rank}",
        "gpu_pci_bus_id": f"00000000:{0x1b + 0x2b * local_rank:02x}:00.0",
    }
    config = {"worker": worker, "milabench": {"run": instrument.run_block(), "dummy": True}}

    log_dir = os.path.join(os.environ.get("TT_WANDB_TEST_DIR", "./outputs"), f"rank_{rank}")
    t0 = time.time()
    start = time.perf_counter()
    logger = WandBLogger(log_dir, config_dict=config)
    init_s = time.perf_counter() - start
    start = time.perf_counter()
    dist.barrier()
    barrier_s = time.perf_counter() - start

    slow_waits = data_waits(slow_rank, slow_rank, steps)
    shared = step_times(steps, slow_waits)
    waits = data_waits(rank, slow_rank, steps)
    mem_extra = 2.0 if node_index == heavy_node else 0.0
    rng = random.Random(100 + rank)

    rank_dir = os.environ.get("TT_RANK_METRICS_DIR")
    rank_file = None
    if rank_dir:
        os.makedirs(rank_dir, exist_ok=True)
        rank_file = open(os.path.join(rank_dir, f"metrics.{rank}.jsonl"), "w")
        rank_file.write(json.dumps({"rank": rank, "kind": "worker", "time": time.time(), **worker}) + "\n")

    clock = t0
    last_logged = 0
    for step in range(1, steps + 1):
        clock += shared[step - 1]
        measuring = step > SKIP
        if rank_file:
            rank_file.write(json.dumps({
                "rank": rank, "kind": "step", "step": step, "time": clock,
                "data_wait_s": waits[step - 1], "local_tokens": TOKENS_PER_GPU,
                "mem_allocated_gib": 32.16 + mem_extra + rng.uniform(-0.01, 0.01),
                "measuring": measuring, "step_s": shared[step - 1],
                "step_s_source": "cuda_events" if measuring else "wall",
            }) + "\n")
        if not (step == 1 or step % LOG_FREQ == 0):
            continue
        window = range(last_logged, step)
        last_logged = step
        step_time = sorted(shared[i] for i in window)[len(window) // 2]
        window_waits = [waits[i] for i in window]
        tps = TOKENS_PER_GPU / (sum(shared[i] for i in window) / len(window))
        loss = 12.9 * math.exp(-step / 40) + 2.2 + rng.gauss(0, 0.02)
        remaining = (steps - step) * step_time
        metrics = {
            "loss_metrics/global_avg_loss": loss,
            "loss_metrics/global_max_loss": loss + 0.01,
            "grad_norm": 1.2 * math.exp(-step / 60) + 0.1,
            "lr/AdamW": 3e-4 * min(step / 10, 1.0),
            "throughput(tps)": tps,
            "tflops": tps * FLOPS_PER_TOKEN / 1e12,
            "mfu(%)": 100 * tps * FLOPS_PER_TOKEN / PEAK,
            "time_metrics/end_to_end(s)": sum(shared[i] for i in window) / len(window),
            "time_metrics/data_loading(s)": sum(window_waits) / len(window_waits),
            "time_metrics/data_loading(%)": 100 * sum(window_waits) / sum(shared[i] for i in window),
            "memory/max_active(GiB)": 41.40 + mem_extra,
            "memory/max_active(%)": 100 * (41.40 + mem_extra) / 79.2,
            "memory/max_reserved(GiB)": 56.26 + mem_extra,
            "memory/max_reserved(%)": 100 * (56.26 + mem_extra) / 79.2,
            "memory/num_alloc_retries": 0,
            "memory/num_ooms": 0,
            "n_tokens_seen": step * TOKENS_PER_GPU * world,
            "loss": loss,
            "measuring": int(step > SKIP),
            "step_time": step_time,
            "rate": TOKENS_PER_GPU * world / step_time,
            "time_remaining": remaining,
            "dataloader_wait": sorted(window_waits)[len(window_waits) // 2],
            "dataloader_wait_max": max(window_waits),
        }
        if step == 1:
            metrics["time_to_first_step"] = 110.0 + rng.uniform(-1, 1)
        logger.log(metrics, step)
        time.sleep(pause)

    measured = shared[SKIP:]
    my_waits = waits[SKIP:]
    run = logger.wandb.run
    run.summary.update({
        "rank/step_time_median": sorted(measured)[len(measured) // 2],
        "rank/step_time_p90": sorted(measured)[int(0.9 * (len(measured) - 1))],
        "rank/data_wait_median_s": sorted(my_waits)[len(my_waits) // 2],
        "rank/data_wait_max_s": max(my_waits),
        "rank/peak_allocated_gib": 41.40 + mem_extra,
        "rank/peak_reserved_gib": 56.26 + mem_extra,
        "rank/measured_steps": len(measured),
        "startup/wandb_init_s": init_s,
    })
    if rank == 0:
        median = sorted(measured)[len(measured) // 2]
        run.summary.update({
            "official/rate": TOKENS_PER_GPU * world / median,
            "official/tokens_per_s_per_gpu": TOKENS_PER_GPU / median,
            "official/step_time_median": median,
            "official/measured_steps": len(measured),
            "official/warmup_steps": SKIP,
        })
    if rank_file:
        rank_file.close()

    start = time.perf_counter()
    logger.close()
    finish_s = time.perf_counter() - start

    rows = [None] * world
    dist.all_gather_object(rows, (rank, host, local_rank, t0, init_s, barrier_s, finish_s))
    if rank == 0:
        first = min(r[3] for r in rows)
        print("[wandb-ranks] rank host local start_s init_s barrier_s finish_s", flush=True)
        for r, h, lr, t, i, b, f in sorted(rows):
            print(f"[wandb-ranks] {r} {h} {lr} {t - first:.2f} {i:.2f} {b:.2f} {f:.2f}", flush=True)
        inits = sorted(r[4] for r in rows)
        print(f"[wandb-ranks] init_s min={inits[0]:.2f} median={inits[len(inits) // 2]:.2f} "
              f"max={inits[-1]:.2f} ranks={world} slow_data_rank={slow_rank} heavy_node={heavy_node}",
              flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main(*(float(a) if i else int(a) for i, a in enumerate(sys.argv[1:])))
