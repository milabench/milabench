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

Run identity
    Rank 0 pushes a ``run_start`` data event when the trainer starts building
    (before distributed init), so a run that fails early still records its
    job, variant and nodes. Launcher variables (``TT_CAMPAIGN``, ``TT_VARIANT``,
    ``TT_REPETITION``, ``TT_FAMILY``, ``TT_HEALTH_DIR``, ``TT_TOPOLOGY``,
    ``TT_EXCLUSIVE_CLUSTER``) fill the ``run`` and ``placement`` blocks.

Summary
    After training, per-rank memory, step times, data-loading time and
    startup timestamps are gathered and rank 0 pushes one ``summary`` data
    event (schema ``milabench.pretrain.v1``, scripts/analysis/FORMAT.md).
    Everything ends up in milabench's run data; ``export_record.py`` builds
    the JSON results record from it. The official throughput is the median of
    the pushed rates.

Per-rank step rows (level 2, live)
    Every rank appends one JSONL row per step to its own file (no collective),
    flushed on TorchTitan's logging steps, after the sync they already do:
    step time (wall clock during warm-up, device events once measuring), data
    wait, local tokens, allocated memory. The first row identifies the worker
    (host, GPU index, UUID, PCI bus). Directory: ``TT_RANK_METRICS_DIR``, else
    ``$MILABENCH_DIR_RUNS/rank-metrics/<job>-<tag>``; files ``metrics.<rank>.jsonl``
    (rows carry ``rank`` and ``time``, as benchmate's rankstore).

wandb (level-1 metrics, rank 0)
    TorchTitan's MetricsProcessor logs to wandb when enabled (``wandb`` option
    or ``BENCHMATE_WANDB=1``), every ``log_freq`` steps: loss, grad norm, lr,
    tokens/s per GPU, TFLOPS, MFU, data-loading time, peak memory. On the same
    log call this module adds the names shared with the other workloads
    (``loss``, ``rate``: global tokens/s, ``step_time``: median since the last
    log, from device events once measuring), ``measuring``,
    ``time_remaining`` and, at step 1, ``time_to_first_step``. Run identity and
    placement go to the run config, the official results to the run summary.
    Run naming follows benchmate's ``wandb_push``: project
    ``BENCHMATE_WANDB_PROJECT`` (default milabench), group = milabench run
    name, job type = bench name. The entity comes from ``WANDB_ENTITY``.
"""

from __future__ import annotations

import json
import math
import os
import socket
import statistics
import subprocess
import sys
import time
from typing import Any

import torch
import torch.distributed as dist

SCHEMA = "milabench.pretrain.v1"

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


# -- run identity (FORMAT.md "Summary schema", "What the launcher sets") ------


def _env_int(name: str) -> int | None:
    value = os.environ.get(name)
    return int(value) if value else None


def run_block() -> dict[str, Any]:
    return {
        "cluster": os.environ.get("CC_CLUSTER") or os.environ.get("SLURM_CLUSTER_NAME"),
        "campaign": os.environ.get("TT_CAMPAIGN"),
        "variant": os.environ.get("TT_VARIANT") or os.environ.get("TT_RUN_TAG"),
        "repetition": _env_int("TT_REPETITION"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
        "slurm_reservation": os.environ.get("SLURM_JOB_RESERVATION") or None,
        "exclusive_cluster": os.environ.get("TT_EXCLUSIVE_CLUSTER", "0") == "1",
        "run_tag": os.environ.get("TT_RUN_TAG"),
    }


def workload_block() -> dict[str, Any]:
    """Model and batch settings known before the trainer is built."""
    d = RUN_DESCRIPTION
    nccl = None
    try:
        nccl = ".".join(str(v) for v in torch.cuda.nccl.version())
    except Exception:
        pass
    return {
        "model": f"{d.get('model', '?')}-{d.get('flavor', '')}".strip("-"),
        "framework": "torchtitan",
        "framework_commit": _torchtitan_commit(),
        "torch_version": torch.__version__,
        "nccl_version": nccl,
        "n_layers": d.get("n_layers"),
        "n_layers_full": d.get("n_layers_full"),
        "params_trainable_fraction": 1.0,
        "seq_len": d.get("seq_len"),
        "microbatch_sequences": d.get("microbatch_sequences_per_gpu"),
        "grad_accum": d.get("grad_accum", 1),
        "precision": "bf16",
        "optimizer": "AdamW",
        "optimizer_state_dtype": d.get("adam_moments"),
        "activation_checkpointing": d.get("activation_checkpointing"),
        "compile": d.get("compile"),
        "moe": None,
    }


def _hostnames(spec: str) -> list[str]:
    if "[" not in spec:
        return [h for h in spec.replace(",", " ").split() if h]
    try:
        out = subprocess.run(
            ["scontrol", "show", "hostnames", spec], capture_output=True, text=True, check=True
        )
        return out.stdout.split()
    except Exception:
        return []


def _leaves(hosts: list[str]) -> tuple[dict[str, list[str]], float | None]:
    """Leaf switch -> hosts of this run, from the launcher's topology snapshot."""
    path = os.environ.get("TT_TOPOLOGY")
    if not path or not os.path.exists(path):
        return {}, None
    wanted, leaves = set(hosts), {}
    for line in open(path).read().splitlines():
        fields = dict(f.split("=", 1) for f in line.split() if "=" in f)
        if fields.get("Level") == "0" and "Nodes" in fields:
            mine = sorted(wanted.intersection(_hostnames(fields["Nodes"])))
            if mine:
                leaves[fields["SwitchName"]] = mine
    return leaves, os.path.getmtime(path)


def _excluded_nodes() -> dict[str, str]:
    """Nodes the launcher's health check excluded (``excluded.txt``: host, reason)."""
    health = os.environ.get("TT_HEALTH_DIR")
    path = os.path.join(health, "excluded.txt") if health else None
    if not path or not os.path.exists(path):
        return {}
    excluded = {}
    for line in open(path).read().splitlines():
        host, _, reason = line.partition("\t")
        if host:
            excluded[host] = reason
    return excluded


def placement_block(hosts: list[str]) -> dict[str, Any]:
    leaves, topology_epoch = _leaves(hosts)
    return {
        "hosts": hosts,
        "family": os.environ.get("TT_FAMILY"),
        "ep_groups": None,
        "leaves": leaves,
        "excluded_nodes": _excluded_nodes(),
        "health_check": os.environ.get("TT_HEALTH_DIR"),
        "topology_snapshot_epoch": topology_epoch,
    }


def deviations() -> list[dict[str, str]]:
    d = RUN_DESCRIPTION
    result = [
        {"id": "mtp", "text": "MTP off"},
        {"id": "text_only", "text": "text only (vision encoder removed)"},
    ]
    if d.get("n_layers") and d.get("n_layers_full") and d["n_layers"] < d["n_layers_full"]:
        result.append({"id": "truncated", "text": f"{d['n_layers']} of {d['n_layers_full']} layers"})
    if d.get("options", {}).get("dataset") == "random":
        result.append({"id": "data", "text": "random tokens"})
    return result


def _launch_hosts() -> list[str]:
    """Nodes of this run as the launcher gave them (before distributed init)."""
    spec = os.environ.get("TT_NODES") or os.environ.get("SLURM_JOB_NODELIST") or ""
    return sorted(set(_hostnames(spec))) or [socket.gethostname()]


def _compile_cache_state(compile_mode: str | None, stats: dict[str, Any]) -> str:
    if compile_mode in (None, "eager"):
        return "none"
    if stats.get("fxgraph_cache_miss", 0) == 0 and stats.get("fxgraph_cache_hit", 0) > 0:
        return "warm"
    return "cold"


def set_wandb_env() -> None:
    """Name the run for TorchTitan's WandBLogger (reads WANDB_* at init).

    Values already set in the environment win.
    """
    try:
        config = json.loads(os.environ.get("MILABENCH_CONFIG") or "{}")
    except ValueError:
        config = {}
    run = run_block()
    name = run["run_tag"] or run["variant"]
    tags = [run["cluster"], os.environ.get("TT_FAMILY"), run["campaign"]]
    defaults = {
        "WANDB_PROJECT": os.environ.get("BENCHMATE_WANDB_PROJECT", "milabench"),
        "WANDB_RUN_GROUP": config.get("run_name"),
        "WANDB_RUN_JOB_TYPE": config.get("name"),
        "WANDB_RUN_NAME": name,
        "WANDB_RUN_TAGS": ",".join(t for t in tags if t),
    }
    if wandb_all_ranks():
        # One run per rank: name it after its worker. Only local rank 0 samples the
        # node's system metrics (every run would otherwise sample all its GPUs).
        local_rank = os.environ.get("LOCAL_RANK", "0")
        host = socket.gethostname().split(".")[0]
        defaults["WANDB_RUN_NAME"] = f"{name or 'run'}-{host}-gpu{local_rank}"
        defaults["WANDB_RUN_TAGS"] += f",{host}"
        if local_rank != "0":
            defaults["WANDB__DISABLE_STATS"] = "true"
    for name, value in defaults.items():
        if value:
            os.environ.setdefault(name, str(value))


def wandb_all_ranks() -> bool:
    """Per-rank wandb runs (TorchTitan ``save_for_all_ranks``), as tt_native.configs decides."""
    if os.environ.get("BENCHMATE_WANDB_ALL_RANKS") == "1":
        return True
    try:
        return bool(json.loads(os.environ.get("TT_NATIVE_OPTIONS") or "{}").get("wandb_all_ranks"))
    except ValueError:
        return False


def _wandb_run():
    """The wandb run TorchTitan opened on this rank, if any (no import otherwise)."""
    wandb = sys.modules.get("wandb")
    return wandb.run if wandb is not None else None


def _local_gpu() -> tuple[int, str | None]:
    local_rank = int(os.environ.get("LOCAL_RANK", torch.cuda.current_device()))
    try:
        uuid = str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid)
        uuid = uuid if uuid.startswith("GPU-") else f"GPU-{uuid}"
    except Exception:
        uuid = None
    return local_rank, uuid


def _worker() -> dict[str, Any]:
    """This rank's identity: node, GPU (index, UUID, PCI address)."""
    local_rank, uuid = _local_gpu()
    pci = None
    try:
        p = torch.cuda.get_device_properties(torch.cuda.current_device())
        pci = f"{p.pci_domain_id:08x}:{p.pci_bus_id:02x}:{p.pci_device_id:02x}.0"
    except Exception:
        pass
    return {
        "rank": _rank(),
        "local_rank": local_rank,
        "host": socket.gethostname().split(".")[0],
        "cuda_device": torch.cuda.current_device(),
        "gpu_uuid": uuid,
        "gpu_pci_bus_id": pci,
    }


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
        self.run_start: dict[str, Any] | None = None
        self.wandb_init_s: float | None = None  # this rank's wandb.init, when it ran
        self._pending_rows: list[dict[str, Any]] = []  # steps not yet written to the rank file
        self._rank_file = None
        self.rank_metrics_dir: str | None = None
        self.global_tokens_per_step = 0
        self._walls_logged = 0  # warm-up wall times / step ends already in a wandb log
        self._ends_logged = 0

    # -- wrapped methods ---------------------------------------------------

    def on_init_start(self) -> None:
        self.t_init_start = time.time()
        set_wandb_env()
        if int(os.environ.get("RANK", "0")) == 0:
            self.run_start = {
                "schema": SCHEMA,
                "run": run_block(),
                "workload": workload_block(),
                "placement": placement_block(_launch_hosts()),
                "process_start_epoch": _process_start(),
            }
            self.push(run_start=self.run_start, task="train")

    def on_init_done(self, trainer) -> None:
        self.t_model_initialized = time.time()
        self.trainer = trainer
        ctx = trainer.engine.parallelism_context
        # Ranks of a TP/CP group share the same tokens: count them once.
        self.dp_divisor = ctx.tp * ctx.cp
        training = trainer.config.training
        self.global_tokens_per_step = (
            training.num_tokens_per_microbatch_per_dp_rank
            * trainer.num_pp_microbatches
            * trainer.gradient_accumulation_steps
            * ctx.world_size
            // (ctx.tp * ctx.cp * ctx.pp)
        )
        run = _wandb_run()
        if run is not None:
            config: dict[str, Any] = {"worker": _worker()}
            if self.run_start is not None:
                config["milabench"] = self.run_start
            else:
                config["milabench"] = {"run": run_block(), "workload": workload_block()}
            run.config.update(config, allow_val_change=True)
            run.summary["startup/wandb_init_s"] = self.wandb_init_s

        processor = trainer.metrics_processor
        orig_log = processor.log

        def log(step, global_avg_loss, global_max_loss, grad_norm, extra_metrics=None):
            extra = dict(extra_metrics or {})
            extra.update(self.wandb_metrics(step, global_avg_loss))
            extra.update(self.flush_rank_rows())
            return orig_log(step, global_avg_loss, global_max_loss, grad_norm, extra_metrics=extra)

        processor.log = log

    def wandb_metrics(self, step: int, loss: float) -> dict[str, float]:
        """Metrics added to TorchTitan's log call (its logging steps are synchronized,
        so the device events of the previous steps have completed)."""
        metrics: dict[str, float] = {"loss": loss, "measuring": int(self.timer is not None)}
        if step == 1:
            metrics["time_to_first_step"] = time.time() - _process_start()
        if self.timer is None:
            times = self._warmup_walls[self._walls_logged :]
            self._walls_logged = len(self._warmup_walls)
        else:
            ends = self._step_ends[self._ends_logged :]
            times = [a.elapsed_time(b) / 1000 for a, b in zip(ends, ends[1:])]
            self._ends_logged = max(len(self._step_ends) - 1, 0)
        if not times:
            return metrics
        step_time = statistics.median(times)
        metrics["step_time"] = step_time
        metrics["rate"] = self.global_tokens_per_step / step_time
        if self.timer is None:
            measure = self.stop_seconds or self.stop * step_time
            metrics["time_remaining"] = max(self.skip - step, 0) * step_time + measure
        else:
            measured = len(self._step_ends) - 1
            metrics["time_remaining"] = max(self.stop - measured, 0) * step_time
        return metrics

    def flush_rank_rows(self) -> dict[str, float]:
        """Write the pending step rows to this rank's file. Call where the device
        events of those steps have completed (logging steps, end of run).
        Returns this window's data-wait metrics for the wandb log call."""
        rows, self._pending_rows = self._pending_rows, []
        if not rows:
            return {}
        for row in rows:
            index = row.pop("_end_index", None)
            if index is not None:
                ends = self._step_ends
                row["step_s"] = ends[index - 1].elapsed_time(ends[index]) / 1000
                row["step_s_source"] = "cuda_events"
        try:
            f = self._open_rank_file()
            f.write("".join(json.dumps({"rank": _rank(), **r}) + "\n" for r in rows))
            f.flush()
        except OSError as exc:  # never stop training for a metrics file
            print(f"[tt-native] rank metrics write failed: {exc!r}", flush=True)
        waits = [r["data_wait_s"] for r in rows]
        return {"dataloader_wait": statistics.median(waits), "dataloader_wait_max": max(waits)}

    def _open_rank_file(self):
        if self._rank_file is None:
            run = run_block()
            directory = os.environ.get("TT_RANK_METRICS_DIR") or os.path.join(
                os.environ.get("MILABENCH_DIR_RUNS", "."),
                "rank-metrics",
                f"{run['slurm_job_id'] or 'local'}-{run['run_tag'] or run['variant'] or 'run'}",
            )
            os.makedirs(directory, exist_ok=True)
            self.rank_metrics_dir = directory
            self._rank_file = open(os.path.join(directory, f"metrics.{_rank()}.jsonl"), "a")
            header = {"rank": _rank(), "kind": "worker", "time": time.time(), **_worker(), "run": run}
            self._rank_file.write(json.dumps(header) + "\n")
        return self._rank_file

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
        row = {
            "kind": "step",
            "step": step,
            "time": time.time(),
            "data_wait_s": self._data_time_acc,
            "local_tokens": self._local_tokens,
            "mem_allocated_gib": torch.cuda.memory_allocated() / 1024**3,
            "measuring": self.timer is not None,
        }
        self._pending_rows.append(row)
        if self.timer is None:
            self._warmup_walls.append(wall)
            row["step_s"], row["step_s_source"] = wall, "wall"
            self._data_time_acc = 0.0
            return
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        self._step_ends.append(end)
        row["_end_index"] = len(self._step_ends) - 1
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
            self.flush_rank_rows()
            return
        self.timer._push()
        torch.cuda.synchronize()
        self.flush_rank_rows()
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
            self.wandb_init_s if self.wandb_init_s is not None else float("nan"),
        ]
        self._wandb_rank_summary(local, steps)
        rows = _gather_floats(local)
        # Every rank measured the same number of steps (agreed count).
        data_rows = _gather_floats(self._data_times)
        ranks = _gather_objects((socket.gethostname(), *_local_gpu()))
        hosts = [r[0] for r in ranks]
        if _rank() != 0:
            return

        trainer = self.trainer
        ctx = trainer.engine.parallelism_context
        training = trainer.config.training
        dp = ctx.world_size // (ctx.tp * ctx.cp * ctx.pp)
        tokens_per_gpu = (
            training.num_tokens_per_microbatch_per_dp_rank
            * trainer.num_pp_microbatches
            * trainer.gradient_accumulation_steps
        )
        tokens_per_step = tokens_per_gpu * dp
        t0 = min(r[2] for r in rows)

        def spread(i):
            col = [r[i] for r in rows]
            return {"min": min(col), "max": max(col)}

        def since_start(i):
            col = [r[i] - t0 for r in rows]
            return {"min_s": min(col), "max_s": max(col)}

        params = getattr(trainer.engine, "model_param_count", None)
        workload = {
            **workload_block(),
            "params_total": params,
            "params_active": params,
            "seq_len": training.max_context_length,
            "grad_accum": trainer.gradient_accumulation_steps,
            "tokens_per_gpu_per_step": tokens_per_gpu,
            "global_batch_tokens": tokens_per_step,
            "global_batch_sequences": tokens_per_step // training.max_context_length,
        }
        summary = {
            "schema": SCHEMA,
            "run": run_block(),
            "workload": workload,
            "parallelism": {
                "gpus": ctx.world_size,
                "nodes": len(set(hosts)),
                "dp_replicate": ctx.dp_replicate,
                "dp_shard": ctx.dp_shard,
                "tp": ctx.tp,
                "pp": ctx.pp,
                "cp": ctx.cp,
                "ep": 1,
            },
            "placement": placement_block(sorted(set(hosts))),
            "framework_config": RUN_DESCRIPTION,
            "measurement": {
                "warmup_steps": self.skip,
                "measured_steps": len(steps),
                "measure_seconds_budget": self.stop_seconds,
                "measurement_start_epoch": self.t_measure_start,
                "measurement_end_epoch": time.time(),
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
                "wandb_init_s": _finite_spread([r[12] for r in rows]),
                "compile_cache": _compile_cache_state(
                    RUN_DESCRIPTION.get("compile"), self.compile
                ),
                "compile_rank0": self.compile,
                "note": "seconds since the earliest rank's process start",
            },
            "per_rank": [
                {
                    "rank": r,
                    "host": ranks[r][0],
                    "local_rank": ranks[r][1],
                    "gpu_uuid": ranks[r][2],
                    "step_median": round(row[7], 5),
                    "step_p90": round(row[8], 5),
                    "step_max": round(row[9], 5),
                    "peak_alloc_gib": round(row[0], 2),
                    "peak_reserved_gib": round(row[1], 2),
                    "data_wait_sum_s": round(sum(data_rows[r]), 4),
                    "data_wait_max_s": round(max(data_rows[r], default=0.0), 4),
                    "wandb_init_s": None if math.isnan(row[12]) else round(row[12], 3),
                }
                for r, row in enumerate(rows)
            ],
            "rank_metrics": {
                "dir": self.rank_metrics_dir,
                "files": "metrics.<rank>.jsonl",
                "rows": "worker header, then one row per step (step_s, data_wait_s, local_tokens, mem_allocated_gib)",
            },
            "deviations": deviations(),
            "estimates": [],
        }
        self.push(summary=summary, task="train")
        self._wandb_summary(summary, steps, tokens_per_step)

    def _wandb_rank_summary(self, local: list[float], steps: list[float]) -> None:
        """This rank's own numbers in its wandb run (per-rank runs; rank 0 too)."""
        run = _wandb_run()
        if run is None:
            return
        run.summary.update(
            {
                "rank/step_time_median": local[7],
                "rank/step_time_p90": local[8],
                "rank/data_wait_median_s": local[10],
                "rank/data_wait_max_s": local[11],
                "rank/peak_allocated_gib": local[0],
                "rank/peak_reserved_gib": local[1],
                "rank/measured_steps": len(steps),
            }
        )

    def _wandb_summary(self, summary: dict[str, Any], steps: list[float], tokens: int) -> None:
        run = _wandb_run()
        if run is None or not steps:
            return
        rates = [tokens / s for s in steps]
        gpus = summary["parallelism"]["gpus"]
        startup = summary["startup"]
        run.summary.update(
            {
                "official/rate": statistics.median(rates),
                "official/tokens_per_s_per_gpu": statistics.median(rates) / gpus,
                "official/step_time_p10": percentile(steps, 0.1),
                "official/step_time_median": statistics.median(steps),
                "official/step_time_p90": percentile(steps, 0.9),
                "official/measured_steps": len(steps),
                "official/warmup_steps": self.skip,
                "startup/model_initialized_s": startup["model_initialized"]["max_s"],
                "startup/first_step_done_s": startup["first_step_done"]["max_s"],
                "startup/compile_cache": startup["compile_cache"],
                "memory/peak_reserved_gib_max": summary["memory_gib"]["peak_reserved"]["max"],
            }
        )


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


def _finite_spread(values: list[float]) -> dict[str, float] | None:
    values = [v for v in values if not math.isnan(v)]
    if not values:
        return None
    return {"min": min(values), "median": statistics.median(values), "max": max(values), "ranks": len(values)}


def _rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def _gather_floats(values: list[float]) -> list[list[float]]:
    if not dist.is_initialized():
        return [values]
    local = torch.tensor(values, dtype=torch.float64, device="cuda")
    out = [torch.empty_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(out, local)
    return [t.tolist() for t in out]


def _gather_objects(value: Any) -> list[Any]:
    if not dist.is_initialized():
        return [value]
    values: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(values, value)
    return values


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

    # wandb.init on every logging rank: timed, and logged before and after, so a
    # slow or hung init (known at scale) shows which rank and how long.
    from torchtitan.observability import metrics as tt_metrics

    orig_wandb_init = tt_metrics.WandBLogger.__init__

    def wandb_init(self, *args, **kwargs):
        rank = os.environ.get("RANK", "0")
        print(f"[tt-native] wandb_init_start rank={rank} t={time.time():.3f}", flush=True)
        start = time.perf_counter()
        try:
            orig_wandb_init(self, *args, **kwargs)
        finally:
            instrument.wandb_init_s = time.perf_counter() - start
            print(
                f"[tt-native] wandb_init_done rank={rank} s={instrument.wandb_init_s:.3f}",
                flush=True,
            )

    tt_metrics.WandBLogger.__init__ = wandb_init

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
