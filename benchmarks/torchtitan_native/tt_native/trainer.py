"""Trainer with a fixed benchmark structure: warm-up, then measurement.

* Step time is the wall time between consecutive step ends, measured after a
  device synchronize, so it covers data loading, forward/backward, optimizer
  and any collective the step waits on.
* Warm-up steps (compilation, CUDA graph capture, allocator growth) are
  excluded from every reported number.
* The measurement window is a step count, or a time budget converted to a
  step count after warm-up (agreed on by all ranks, so they stop together).
* Rank 0 writes one JSON results record; every rank's step-time summary goes
  to a compact JSONL file next to it so stragglers can be found.
"""

from __future__ import annotations

import json
import logging
import math
import os
import socket
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
import torch.distributed as dist

from torchtitan.trainer import Trainer

logger = logging.getLogger(__name__)

SCHEMA = "milabench.torchtitan_native.v1"

# Set by main.py as early as possible (before heavy imports).
PROCESS_START: float | None = None

# Optional callback installed by the voirfile: fn(step, step_time_s, tokens, loss).
STEP_HOOK: Callable[..., None] | None = None


def _process_start() -> float:
    if PROCESS_START is not None:
        return PROCESS_START
    try:
        import psutil

        return psutil.Process().create_time()
    except Exception:
        return time.time()


def _torchtitan_commit() -> str | None:
    try:
        from importlib.metadata import distribution

        direct = distribution("torchtitan").read_text("direct_url.json")
        if direct:
            return json.loads(direct).get("vcs_info", {}).get("commit_id")
    except Exception:
        pass
    return None


def _compile_stats() -> dict[str, Any]:
    """Inductor cache hits/misses and compile time so far (this rank)."""
    stats: dict[str, Any] = {}
    try:
        from torch._dynamo.utils import counters

        inductor = counters.get("inductor", {})
        for key in (
            "fxgraph_cache_hit",
            "fxgraph_cache_miss",
            "fxgraph_cache_bypass",
            "autotune_local_cache_hit",
            "autotune_local_cache_miss",
        ):
            if key in inductor:
                stats[key] = int(inductor[key])
        stats["frames_compiled"] = int(counters.get("stats", {}).get("unique_graphs", 0))
    except Exception as exc:  # stats only
        stats["error"] = repr(exc)
    try:
        from torch._dynamo.utils import compile_times

        per_phase = compile_times(repr="raw", aggregate=True)
        stats["compile_seconds"] = {
            k: round(float(v), 2) for k, v in per_phase.items() if float(v) >= 0.5
        }
    except Exception as exc:
        stats["compile_seconds_error"] = repr(exc)
    return stats


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    k = (len(ordered) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


@dataclass(kw_only=True, slots=True)
class BenchConfig:
    warmup_steps: int = 40
    measure_steps: int = 100
    measure_seconds: float | None = None
    """If set, overrides ``measure_steps`` with a time budget for the window."""
    max_measure_steps: int = 100_000
    results_dir: str = "./results"
    run_name: str = "run"
    description: dict[str, Any] = field(default_factory=dict)
    """Free-form config recorded verbatim in the results record."""


class BenchTrainer(Trainer):
    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        bench: BenchConfig = field(default_factory=BenchConfig)

    def __init__(self, config: Config):
        self._t_trainer_init_start = time.time()
        super().__init__(config)
        self._t_model_initialized = time.time()

        bench = config.bench
        self._warmup_steps = bench.warmup_steps
        self._total_steps = bench.warmup_steps + bench.measure_steps
        self._step_times: list[float] = []
        self._last_step_end: float | None = None
        self._t_first_step_done: float | None = None
        self._t_measure_start: float | None = None
        self._first_step_s: float | None = None
        self._compile_stats: dict[str, Any] = {}

        ctx = self.engine.parallelism_context
        self._world_size = ctx.world_size
        self._dp_degree = ctx.world_size // (ctx.tp * ctx.cp * ctx.pp)
        self._tokens_per_step = (
            config.training.num_tokens_per_microbatch_per_dp_rank
            * self.num_pp_microbatches
            * self.gradient_accumulation_steps
            * self._dp_degree
        )

    # -- training loop -----------------------------------------------------

    def should_continue_training(self) -> bool:
        return self.engine.num_completed_steps < self._total_steps

    def train_step(self, data_iterator):
        if self._last_step_end is None:
            torch.cuda.synchronize()
            self._last_step_end = time.perf_counter()

        super().train_step(data_iterator)

        torch.cuda.synchronize()
        now = time.perf_counter()
        step_time = now - self._last_step_end
        self._last_step_end = now
        self._step_times.append(step_time)

        step = self.engine.num_completed_steps
        if step == 1:
            self._t_first_step_done = time.time()
            self._first_step_s = step_time
            self._compile_stats = _compile_stats()
            logger.info(f"[bench] first step {step_time:.1f}s, compile: {self._compile_stats}")
        if step == self._warmup_steps:
            self._t_measure_start = time.time()
            self._plan_measurement()

        if STEP_HOOK is not None and step > self._warmup_steps:
            loss = getattr(self.engine, "loss", None)
            STEP_HOOK(step, step_time, self._tokens_per_step, loss)

    def _plan_measurement(self) -> None:
        """Turn a time budget into a step count every rank agrees on."""
        seconds = self.config.bench.measure_seconds
        if not seconds:
            return
        recent = self._step_times[-min(10, len(self._step_times)) :]
        estimate = torch.tensor([statistics.median(recent)], device=self.engine.device)
        if dist.is_initialized():
            dist.all_reduce(estimate, op=dist.ReduceOp.MAX)
        steps = max(1, math.ceil(seconds / float(estimate.item())))
        steps = min(steps, self.config.bench.max_measure_steps)
        self._total_steps = self._warmup_steps + steps
        logger.info(
            f"[bench] measuring {steps} steps (~{seconds:.0f}s at "
            f"{float(estimate.item()):.3f}s/step)"
        )

    def train(self):
        super().train()
        self._write_results()

    # -- results -----------------------------------------------------------

    def _gather_floats(self, values: list[float]) -> list[list[float]]:
        """All-gather a same-length float list from every rank (NCCL)."""
        local = torch.tensor(values, dtype=torch.float64, device=self.engine.device)
        if not dist.is_initialized():
            return [values]
        out = [torch.empty_like(local) for _ in range(dist.get_world_size())]
        dist.all_gather(out, local)
        return [t.tolist() for t in out]

    def _write_results(self) -> None:
        measured = self._step_times[self._warmup_steps :]
        if not measured:
            logger.warning("[bench] no measured steps; not writing results")
            return

        rank = dist.get_rank() if dist.is_initialized() else 0
        stats = torch.cuda.memory_stats()
        gib = 1024**3
        process_start = _process_start()
        scalars = [
            stats.get("allocated_bytes.all.peak", 0) / gib,
            stats.get("reserved_bytes.all.peak", 0) / gib,
            process_start,
            self._t_trainer_init_start,
            self._t_model_initialized,
            self._t_first_step_done or float("nan"),
            self._t_measure_start or float("nan"),
        ]
        all_scalars = self._gather_floats(scalars)
        all_steps = self._gather_floats(measured)
        if dist.is_initialized():
            hostnames: list[str] = [None] * dist.get_world_size()  # type: ignore
            dist.all_gather_object(hostnames, socket.gethostname())
        else:
            hostnames = [socket.gethostname()]

        if rank != 0:
            return

        # A synchronous step is as slow as its slowest rank.
        step_max = [max(col) for col in zip(*all_steps)]
        median = statistics.median(step_max)
        mean = statistics.fmean(step_max)
        tokens = self._tokens_per_step
        world = self._world_size
        flops_per_token = self.engine.num_flops_per_token
        peak = self.metrics_processor.gpu_peak_flops
        tps = tokens / median
        mfu = 100.0 * flops_per_token * tps / (peak * world) if peak else None

        def spread(index: int) -> dict[str, float]:
            col = [row[index] for row in all_scalars]
            return {"min": min(col), "max": max(col)}

        def since_start(index: int) -> dict[str, float]:
            t0 = min(row[2] for row in all_scalars)
            col = [row[index] - t0 for row in all_scalars]
            return {"min_s": min(col), "max_s": max(col)}

        bench = self.config.bench
        training = self.config.training
        record = {
            "schema": SCHEMA,
            "run_name": bench.run_name,
            "timestamp": time.time(),
            "config": {
                **bench.description,
                "gpus": world,
                "nodes": len(set(hostnames)),
                "parallelism": {
                    "dp_replicate": self.engine.parallelism_context.dp_replicate,
                    "dp_shard": self.engine.parallelism_context.dp_shard,
                    "tp": self.engine.parallelism_context.tp,
                    "pp": self.engine.parallelism_context.pp,
                },
                "seq_len": training.max_context_length,
                "tokens_per_microbatch_per_dp_rank": (
                    training.num_tokens_per_microbatch_per_dp_rank
                ),
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "global_batch_tokens": tokens,
                "global_batch_sequences": tokens // training.max_context_length,
                "torchtitan_commit": _torchtitan_commit(),
                "torch_version": torch.__version__,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "run_tag": os.environ.get("TT_RUN_TAG"),
                "hosts": sorted(set(hostnames)),
            },
            "measurement": {
                "warmup_steps": self._warmup_steps,
                "measured_steps": len(measured),
                "measure_seconds_budget": bench.measure_seconds,
                "step_time_definition": (
                    "wall time between consecutive step ends (after "
                    "cuda.synchronize), max over ranks per step"
                ),
            },
            "throughput": {
                "global_tokens_per_s": tps,
                "tokens_per_s_per_gpu": tps / world,
                "global_tokens_per_s_mean_step": tokens / mean,
                "step_time_s": {
                    "median": median,
                    "mean": mean,
                    "stdev": statistics.pstdev(step_max),
                    "min": min(step_max),
                    "p10": _percentile(step_max, 0.10),
                    "p90": _percentile(step_max, 0.90),
                    "max": max(step_max),
                },
            },
            "mfu": {
                "mfu_percent": mfu,
                "flops_per_token": flops_per_token,
                "peak_flops_per_gpu": peak,
                "formula": (
                    "TorchTitan Qwen35Model.get_nparams_and_flops: "
                    "6*active_params (embeddings excluded, lm_head included) "
                    "+ per full-attention layer 6*n_heads*2*head_dim*seq_len "
                    "(no causal halving) + per Gated DeltaNet layer "
                    "6*3*n_value_heads*key_head_dim*value_head_dim "
                    "(recurrent state update; projections in active_params). "
                    "mfu = flops_per_token * global_tokens_per_s / (peak * gpus)"
                ),
            },
            "memory_gib": {
                "peak_allocated": spread(0),
                "peak_reserved": spread(1),
            },
            "startup": {
                "process_start_epoch": spread(2),
                "trainer_init_start": since_start(3),
                "model_initialized": since_start(4),
                "first_step_done": since_start(5),
                "measurement_start": since_start(6),
                "first_step_s_rank0": self._first_step_s,
                "compile_rank0": self._compile_stats,
                "note": "seconds since the earliest rank's process start",
            },
        }

        out_dir = Path(bench.results_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        if job_id := os.environ.get("SLURM_JOB_ID"):
            # Concurrent runs (one per leaf switch) share a run name.
            stamp = f"{stamp}-j{job_id}"
        record_path = out_dir / f"{bench.run_name}.{stamp}.json"
        per_rank_path = out_dir / f"{bench.run_name}.{stamp}.ranks.jsonl"
        record["per_rank_file"] = str(per_rank_path)

        with open(per_rank_path, "w") as fp:
            for r, (steps, scalars_r) in enumerate(zip(all_steps, all_scalars)):
                fp.write(
                    json.dumps(
                        {
                            "rank": r,
                            "host": hostnames[r],
                            "step_median": round(statistics.median(steps), 5),
                            "step_p90": round(_percentile(steps, 0.9), 5),
                            "step_max": round(max(steps), 5),
                            "peak_alloc_gib": round(scalars_r[0], 2),
                            "peak_reserved_gib": round(scalars_r[1], 2),
                            "first_step_done_s": round(scalars_r[5] - scalars_r[2], 2),
                        }
                    )
                    + "\n"
                )

        with open(record_path, "w") as fp:
            json.dump(record, fp, indent=2)

        logger.info(f"[bench] results: {record_path}")
        print("[bench-result] " + json.dumps(record), flush=True)
