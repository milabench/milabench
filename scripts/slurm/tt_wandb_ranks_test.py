"""CPU-only check of per-rank wandb logging across nodes (no model, no GPU).

Each Slurm task is one rank: it names its run as the GPU benchmark does
(tt_native.instrument.set_wandb_env, per-rank mode), opens TorchTitan's own
WandBLogger, waits at a gloo barrier (how long the slowest init holds the
others), logs a few steps, then finishes. Rank 0 prints one line per rank with
init, barrier and finish times.

    srun --ntasks-per-node=4 python tt_wandb_ranks_test.py [steps] [interval_s]

Needs the bench venv (torch, torchtitan, wandb) and WANDB_ENTITY / mode in
the environment; TT_RUN_TAG names the runs.
"""

import os
import socket
import sys
import time

import torch.distributed as dist


def main(steps: int = 20, interval: float = 1.0) -> None:
    rank = int(os.environ["SLURM_PROCID"])
    os.environ.setdefault("RANK", str(rank))
    os.environ.setdefault("LOCAL_RANK", os.environ["SLURM_LOCALID"])
    os.environ["BENCHMATE_WANDB_ALL_RANKS"] = "1"
    host = socket.gethostname().split(".")[0]
    world = int(os.environ["SLURM_NTASKS"])

    from tt_native import instrument
    from torchtitan.observability.metrics import WandBLogger

    instrument.set_wandb_env()
    dist.init_process_group(
        "gloo", init_method=f"tcp://{os.environ['MASTER_ADDR']}:29511", rank=rank, world_size=world
    )

    log_dir = os.path.join(os.environ.get("TT_WANDB_TEST_DIR", "./outputs"), f"rank_{rank}")
    t0 = time.time()
    start = time.perf_counter()
    logger = WandBLogger(log_dir, config_dict={"worker": {"rank": rank, "host": host,
                                                           "local_rank": os.environ["LOCAL_RANK"]}})
    init_s = time.perf_counter() - start
    start = time.perf_counter()
    dist.barrier()
    barrier_s = time.perf_counter() - start

    for step in range(1, steps + 1):
        logger.log({"step_time": interval, "rank_metric": rank + step / 100}, step)
        time.sleep(interval)

    start = time.perf_counter()
    logger.close()
    finish_s = time.perf_counter() - start

    rows = [None] * world
    dist.all_gather_object(rows, (rank, host, os.environ["LOCAL_RANK"], t0, init_s, barrier_s, finish_s))
    if rank == 0:
        first = min(r[3] for r in rows)
        print("[wandb-ranks] rank host local start_s init_s barrier_s finish_s", flush=True)
        for r, h, lr, t, i, b, f in sorted(rows):
            print(f"[wandb-ranks] {r} {h} {lr} {t - first:.2f} {i:.2f} {b:.2f} {f:.2f}", flush=True)
        inits = sorted(r[4] for r in rows)
        print(f"[wandb-ranks] init_s min={inits[0]:.2f} median={inits[len(inits) // 2]:.2f} "
              f"max={inits[-1]:.2f} ranks={world}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main(*(float(a) if i else int(a) for i, a in enumerate(sys.argv[1:])))
