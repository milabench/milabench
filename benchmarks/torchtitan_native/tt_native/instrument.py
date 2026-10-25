"""voir/benchmate instrumentation of TorchTitan's Trainer (installed by voirfile.py).

Same pattern as the other milabench packs: TorchTitan's trainer is unchanged,
its methods are wrapped from the voirfile, and benchmate does the timing.

Measurement window
    ``skip`` warm-up steps (compilation, allocator growth) are not timed.
    Then benchmate's ``ManualTimedIterator`` times every step with device
    events (no host sync per step) and pushes one ``rate`` per step: global
    tokens per second (TorchTitan's local valid tokens, summed over the
    data-parallel ranks by benchmate). It stops after ``stop`` observations,
    or after a ``stop_seconds`` budget converted to an observation count all
    ranks agree on. The stop is applied on every rank through
    ``should_continue_training``; voir's rank-0 stop signal is not used,
    since stopping rank 0 alone would leave the other ranks in collectives.

Summary
    After training, per-rank memory, step times, data-loading time and
    startup timestamps are gathered and rank 0 pushes one ``summary`` data
    event (config, MFU inputs, memory, startup, ...). Everything ends up in
    milabench's run data; ``export_record.py`` builds the JSON results record
    from it. The official throughput is the median of the pushed rates.
"""

from __future__ import annotations

import json
import math
import os
import socket
import statistics
import time
from typing import Any

import torch
import torch.distributed as dist

SCHEMA = "milabench.torchtitan_native.v2"

# Set by main.py as early as possible (before heavy imports).
PROCESS_START: float | None = None

# Set by tt_native.configs: model / run settings recorded in the summary.
RUN_DESCRIPTION: dict[str, Any] = {}


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


def compile_stats() -> dict[str, Any]:
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

        # "name, seconds" lines after a two-line header.
        seconds = {}
        for line in (compile_times(repr="str", aggregate=True) or "").splitlines()[2:]:
            name, _, value = line.rpartition(",")
            try:
                if float(value) >= 0.5:
                    seconds[name.strip()] = round(float(value), 2)
            except ValueError:
                continue
        stats["compile_seconds"] = seconds
    except Exception as exc:
        stats["compile_seconds_error"] = repr(exc)
    return stats


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    k = (len(ordered) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


class TrainerInstrument:
    """Per-process state; methods are called from the wrapped Trainer methods."""

    def __init__(self, *, skip: int, stop: int, stop_seconds: float | None, push):
        self.skip = skip
        self.stop = stop
        self.stop_seconds = stop_seconds
        self.push = push
        self.timer = None
        self.trainer = None
        self.dp_divisor = 1
        self.t_init_start: float | None = None
        self.t_model_initialized: float | None = None
        self.t_first_step_done: float | None = None
        self.t_measure_start: float | None = None
        self.first_step_s: float | None = None
        self.compile: dict[str, Any] = {}
        self._warmup_walls: list[float] = []
        self._last_wall: float | None = None
        self._step_ends: list[Any] = []  # device events: measurement start, then each step end
        self._data_time_acc = 0.0
        self._data_times: list[float] = []
        self._local_tokens = 0
        self._loss = None

    # -- wrapped methods ---------------------------------------------------

    def on_init_start(self) -> None:
        self.t_init_start = time.time()

    def on_init_done(self, trainer) -> None:
        self.t_model_initialized = time.time()
        self.trainer = trainer
        ctx = trainer.engine.parallelism_context
        # Ranks of a TP/CP group share the same tokens: count them once.
        self.dp_divisor = ctx.tp * ctx.cp

    def on_tokens(self, local_tokens: int) -> None:
        self._local_tokens = int(local_tokens)

    def on_loss(self, loss) -> None:
        self._loss = loss

    def on_data_time(self, seconds: float) -> None:
        self._data_time_acc += seconds

    def before_step(self) -> None:
        if self._last_wall is None:
            self._last_wall = time.perf_counter()
        step = self.trainer.engine.num_completed_steps + 1
        if step == self.skip + 1:
            self._start_measurement()

    def after_step(self) -> None:
        step = self.trainer.engine.num_completed_steps
        now = time.perf_counter()
        wall = now - self._last_wall
        self._last_wall = now
        if step == 1:
            self.t_first_step_done = time.time()
            self.first_step_s = wall
            self.compile = compile_stats()
        if self.timer is None:
            self._warmup_walls.append(wall)
            self._data_time_acc = 0.0
            return
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        self._step_ends.append(end)
        self._data_times.append(self._data_time_acc)
        self._data_time_acc = 0.0
        # benchmate sums this over all ranks at push time.
        self.timer.step(batch_override=self._local_tokens // self.dp_divisor)
        # Push (and sync) on TorchTitan's logging steps only, which already
        # synchronize; every rank pushes at the same step (collective reduce).
        if self.timer.is_done() or step % 10 == 0:
            self.timer._push()
            if self._loss is not None and _rank() == 0:
                self.push(loss=float(self._loss), task="train")

    def should_stop(self) -> bool:
        return self.timer is not None and self.timer.is_done()

    # -- measurement -------------------------------------------------------

    def _start_measurement(self) -> None:
        from benchmate.metrics import ManualTimedIterator, agree_observation_count

        count = self.stop
        if self.stop_seconds:
            recent = self._warmup_walls[-10:] or [1.0]
            count = agree_observation_count(
                self.stop_seconds,
                statistics.fmean(recent),
                device=torch.device("cuda", torch.cuda.current_device()),
            )
        self.stop = count
        self.t_measure_start = time.time()
        self.timer = ManualTimedIterator(
            iter(()),
            event_fn=torch.cuda.Event,
            rank=_rank(),
            push=self.push,
            device=torch.device("cuda", torch.cuda.current_device()),
            earlystop=count,
            raise_stop_program=False,
        )
        # Manual stepping: set what TimedIterator.__iter__ would have set.
        self.timer.last_time = time.time()
        self.timer.start = torch.cuda.Event(enable_timing=True)
        self.timer.start.record()
        self._step_ends = [self.timer.start]

    # -- summary -----------------------------------------------------------

    def push_summary(self) -> None:
        if self.timer is None:
            return
        self.timer._push()
        torch.cuda.synchronize()
        ends = self._step_ends
        steps = [a.elapsed_time(b) / 1000 for a, b in zip(ends, ends[1:])]
        stats = torch.cuda.memory_stats()
        gib = 1024**3
        local = [
            stats.get("allocated_bytes.all.peak", 0) / gib,
            stats.get("reserved_bytes.all.peak", 0) / gib,
            _process_start(),
            self.t_init_start or float("nan"),
            self.t_model_initialized or float("nan"),
            self.t_first_step_done or float("nan"),
            self.t_measure_start or float("nan"),
            statistics.median(steps) if steps else float("nan"),
            percentile(steps, 0.9),
            max(steps) if steps else float("nan"),
            statistics.median(self._data_times) if self._data_times else 0.0,
            max(self._data_times) if self._data_times else 0.0,
        ]
        rows = _gather_floats(local)
        # Every rank measured the same number of steps (agreed count).
        data_rows = _gather_floats(self._data_times)
        hosts = _gather_hosts()
        if _rank() != 0:
            return

        trainer = self.trainer
        ctx = trainer.engine.parallelism_context
        training = trainer.config.training
        dp = ctx.world_size // (ctx.tp * ctx.cp * ctx.pp)
        tokens_per_step = (
            training.num_tokens_per_microbatch_per_dp_rank
            * trainer.num_pp_microbatches
            * trainer.gradient_accumulation_steps
            * dp
        )
        t0 = min(r[2] for r in rows)

        def spread(i):
            col = [r[i] for r in rows]
            return {"min": min(col), "max": max(col)}

        def since_start(i):
            col = [r[i] - t0 for r in rows]
            return {"min_s": min(col), "max_s": max(col)}

        summary = {
            "schema": SCHEMA,
            "config": {
                **RUN_DESCRIPTION,
                "gpus": ctx.world_size,
                "nodes": len(set(hosts)),
                "hosts": sorted(set(hosts)),
                "parallelism": {
                    "dp_replicate": ctx.dp_replicate,
                    "dp_shard": ctx.dp_shard,
                    "tp": ctx.tp,
                    "pp": ctx.pp,
                },
                "seq_len": training.max_context_length,
                "tokens_per_microbatch_per_dp_rank": (
                    training.num_tokens_per_microbatch_per_dp_rank
                ),
                "gradient_accumulation_steps": trainer.gradient_accumulation_steps,
                "global_batch_tokens": tokens_per_step,
                "global_batch_sequences": tokens_per_step // training.max_context_length,
                "torchtitan_commit": _torchtitan_commit(),
                "torch_version": torch.__version__,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "run_tag": os.environ.get("TT_RUN_TAG"),
            },
            "measurement": {
                "warmup_steps": self.skip,
                "measured_steps": len(steps),
                "measure_seconds_budget": self.stop_seconds,
                "rate_definition": (
                    "per step: global tokens / step time; step time from device "
                    "events on rank 0 (benchmate ManualTimedIterator); official "
                    "value = median of the per-step rates"
                ),
            },
            "mfu_inputs": {
                "flops_per_token": trainer.engine.num_flops_per_token,
                "peak_flops_per_gpu": trainer.metrics_processor.gpu_peak_flops,
                "formula": (
                    "TorchTitan Qwen35Model.get_nparams_and_flops: "
                    "6*active_params (embeddings excluded, lm_head included) "
                    "+ per full-attention layer 6*n_heads*2*head_dim*seq_len "
                    "(no causal halving) + per Gated DeltaNet layer "
                    "6*3*n_value_heads*key_head_dim*value_head_dim. "
                    "mfu = flops_per_token * tokens_per_s / (peak * gpus)"
                ),
            },
            "memory_gib": {"peak_allocated": spread(0), "peak_reserved": spread(1)},
            "data_loading": {
                "median_s_per_step": max(r[10] for r in rows),
                "max_s_per_step": max(r[11] for r in rows),
                "definition": (
                    "CPU time fetching a step's microbatches, measured steps; "
                    "median per rank, worst rank reported"
                ),
                "stragglers": _data_stragglers(data_rows, steps, hosts),
            },
            "startup": {
                "process_start_epoch": spread(2),
                "trainer_init_start": since_start(3),
                "model_initialized": since_start(4),
                "first_step_done": since_start(5),
                "measurement_start": since_start(6),
                "first_step_s_rank0": self.first_step_s,
                "compile_rank0": self.compile,
                "note": "seconds since the earliest rank's process start",
            },
            "per_rank": [
                {
                    "rank": r,
                    "host": hosts[r],
                    "step_median": round(row[7], 5),
                    "step_p90": round(row[8], 5),
                    "step_max": round(row[9], 5),
                    "peak_alloc_gib": round(row[0], 2),
                    "peak_reserved_gib": round(row[1], 2),
                    "data_wait_sum_s": round(sum(data_rows[r]), 4),
                    "data_wait_max_s": round(max(data_rows[r], default=0.0), 4),
                }
                for r, row in enumerate(rows)
            ],
        }
        self.push(summary=summary, task="train")


def _data_stragglers(
    data_rows: list[list[float]], steps: list[float], hosts: list[str]
) -> dict[str, Any]:
    """How much waiting for data on the slowest rank costs each step.

    With FSDP every rank waits for the slowest one, so the per-step cost is the
    maximum wait over ranks (``data_rows[rank][step]``), compared with the
    step time measured on rank 0.
    """
    n = min(len(steps), *(len(r) for r in data_rows)) if data_rows else 0
    if n == 0:
        return {}
    worst = [max(r[i] for r in data_rows) for i in range(n)]
    step = steps[:n]
    totals = sorted(
        ((sum(r[:n]), rank) for rank, r in enumerate(data_rows)), reverse=True
    )

    def mean_step(selected):
        return statistics.fmean(selected) if selected else None

    corr = None
    if n > 2 and statistics.pstdev(worst) > 0 and statistics.pstdev(step) > 0:
        corr = statistics.correlation(worst, step)
    return {
        "worst_rank_wait_s": {
            "mean": statistics.fmean(worst),
            "median": statistics.median(worst),
            "p90": percentile(worst, 0.9),
            "max": max(worst),
        },
        "exposed_fraction": sum(worst) / sum(step),
        "steps_over_0.1s": sum(w > 0.1 for w in worst),
        "steps_over_0.5s": sum(w > 0.5 for w in worst),
        "steps": n,
        "mean_step_s_wait_over_0.1s": mean_step([s for s, w in zip(step, worst) if w > 0.1]),
        "mean_step_s_wait_under_0.1s": mean_step([s for s, w in zip(step, worst) if w <= 0.1]),
        "corr_worst_wait_vs_step": corr,
        "top_ranks": [
            {"rank": rank, "host": hosts[rank], "wait_sum_s": total}
            for total, rank in totals[:3]
        ],
        "definition": (
            "per measured step: max over ranks of the data wait; step time from "
            "rank 0 device events; exposed_fraction = sum(max wait) / sum(step time)"
        ),
    }


def _rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def _gather_floats(values: list[float]) -> list[list[float]]:
    if not dist.is_initialized():
        return [values]
    local = torch.tensor(values, dtype=torch.float64, device="cuda")
    out = [torch.empty_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(out, local)
    return [t.tolist() for t in out]


def _gather_hosts() -> list[str]:
    if not dist.is_initialized():
        return [socket.gethostname()]
    hosts: list[str] = [None] * dist.get_world_size()  # type: ignore[list-item]
    dist.all_gather_object(hosts, socket.gethostname())
    return hosts


def install(instrument: TrainerInstrument) -> None:
    """Wrap TorchTitan's Trainer methods (idempotent)."""
    # The trainer calls `sl.log_trace_scalar` on the package, which re-exports
    # it from structured_logging: patch the package attribute.
    from torchtitan.observability import structured_logger as sl
    from torchtitan.trainer import Trainer

    if getattr(Trainer, "_tt_native_instrumented", False):
        return

    orig_init = Trainer.__init__
    orig_train_step = Trainer.train_step
    orig_train = Trainer.train
    orig_should_continue = Trainer.should_continue_training
    orig_microbatches = Trainer.microbatch_generator
    orig_scalar = sl.log_trace_scalar

    def __init__(self, config):
        instrument.on_init_start()
        orig_init(self, config)
        instrument.on_init_done(self)

    def train_step(self, data_iterator):
        instrument.before_step()
        orig_train_step(self, data_iterator)
        loss = getattr(self.engine, "loss", None)
        if loss is not None:
            instrument.on_loss(loss)
        instrument.after_step()

    def train(self):
        orig_train(self)
        instrument.push_summary()

    def should_continue_training(self):
        return orig_should_continue(self) and not instrument.should_stop()

    def microbatch_generator(self, data_iterable):
        inner = orig_microbatches(self, data_iterable)
        while True:
            start = time.perf_counter()
            try:
                microbatch = next(inner)
            except StopIteration:
                return
            instrument.on_data_time(time.perf_counter() - start)
            yield microbatch

    def log_trace_scalar(scalars, *args, **kwargs):
        if (local := scalars.get("local_valid_tokens")) is not None:
            instrument.on_tokens(local)
        return orig_scalar(scalars, *args, **kwargs)

    Trainer.__init__ = __init__
    Trainer.train_step = train_step
    Trainer.train = train
    Trainer.should_continue_training = should_continue_training
    Trainer.microbatch_generator = microbatch_generator
    sl.log_trace_scalar = log_trace_scalar
    Trainer._tt_native_instrumented = True
