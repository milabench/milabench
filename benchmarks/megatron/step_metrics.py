"""benchmate's TimedIterator on Megatron's training loop, and the per-rank step rows.

Megatron owns the loop, so the timer is attached from two sides:
  - the train dataloader is wrapped (``wrap_loader``): data wait, laps and the CUDA-event clock come for free;
  - one ``step(tokens)`` call per optimizer step closes the measurement (``rate``).
The throughput timer runs on rank 0 only. Every rank, pipeline middle stages included, writes its own step rows
(``RankRows``, scripts/analysis/FORMAT.md section 4): their clock is ``train_step``, not the loader.
"""
import functools
import json
import os
import socket
import time

import torch

from benchmate.metrics import ManualTimedIterator, default_device, default_event
from benchmate.monitor import get_rank
from benchmate.rankstore import get_store, ranked_push


class LocalBatchTimer(ManualTimedIterator):
    """No all-reduce of the batch size: Megatron builds the train loader on some ranks only (first / last pipeline stage),
    a collective here would wait forever on the others. Rank 0 always has a loader and reports the global tokens."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # normally set by __iter__ / wrapped(), which only run if Megatron iterates the loader before the first step
        self.last_time = time.time()
        self.start = None

    def batch_size(self, bs):
        return bs


class SlowLoader:
    """Fake dataloading stall, to see what a starved GPU looks like in the timers: ``BENCHMATE_FAKE_DATA_LATENCY``
    seconds of sleep before every ``BENCHMATE_FAKE_DATA_EVERY``-th batch (default 5), 0 = off."""

    def __init__(self, loader, latency, every):
        self.loader, self.latency, self.every = loader, latency, max(every, 1)

    def __iter__(self):
        for i, batch in enumerate(self.loader):
            if i % self.every == self.every - 1:
                time.sleep(self.latency)
            yield batch

    def __len__(self):
        return len(self.loader)

    def __getattr__(self, item):
        return getattr(self.loader, item)


class DataWait:
    """Host time blocked in the loader's ``next``, added to the step row (``next``). Only the first and last pipeline
    stages have a loader; on the first one the batch is read inside the forward, so this time is also in ``fwd_ms``."""

    def __init__(self, loader, rows):
        self.loader, self.rows = loader, rows
        rows.has_loader = True

    def __iter__(self):
        iterator = iter(self.loader)
        while True:
            start = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                return
            self.rows.data_wait += time.perf_counter() - start
            yield batch

    def __len__(self):
        return len(self.loader)

    def __getattr__(self, item):
        return getattr(self.loader, item)


def _event():
    event = torch.cuda.Event(enable_timing=True)
    event.record()
    return event


class RankRows:
    """One row per step on every rank, in the rankstore file (``metrics.<bench>.<host>.<rank>.jsonl``), with the
    field names the TorchTitan pack writes (``batch_id``, ``elapsed``, ``next``, ``rate``, ...).

    ``elapsed`` is ``train_step`` on the device (CUDA events, start to end of the call, so the metric reports run
    between steps are left out); ``wall`` is the host time from the end of the previous step to the end of this one.
    ``fwd_ms`` / ``bwd_ms`` sum Megatron's per-microbatch ``forward_step`` / ``backward_step`` on the device, exposed
    waits included (pipeline receives are outside them). With the combined 1F1B schedule (EP all-to-all overlap) the
    two are interleaved and only ``fwd_bwd_ms`` is written. ``optimizer_ms`` is ``optimizer.step()`` (gradient norm
    and its all-reduce included).

    The events are never synchronized during the run: a row is written once its events have completed, at the next
    flush (every ``push_every`` steps), and the last rows at ``flush(final=True)``."""

    def __init__(self, push_every):
        self.push_every = push_every
        self.pending = []
        self.current = None  # the step in progress: its events, None outside train_step (evaluation is not counted)
        self.has_loader = False
        self.data_wait = 0.0
        self.last_end = None
        self.header_done = False
        self.written = []  # the rows written so far, for the run summary
        self.path = None  # this rank's rankstore file, once written

    def timed(self, key, fn):
        """``fn`` with CUDA events around each call, added to the current step under ``key``."""

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            current = self.current
            if current is None:
                return fn(*args, **kwargs)
            start = _event()
            result = fn(*args, **kwargs)
            current[key].append((start, _event()))
            return result

        return wrapper

    def begin(self):
        if not self.header_done:
            self.header_done = True
            for row in header_rows():
                ranked_push(**row)
        self.current = {"start": _event(), "fwd": [], "bwd": [], "fwd_bwd": [], "opt": []}

    def time_optimizer(self, optimizer):
        """Time this optimizer's ``step`` (the instance's, once: a ChainedOptimizer's inner ones are not counted again)."""
        if optimizer is not None and not getattr(optimizer, "_rank_rows_timed", False):
            optimizer.step = self.timed("opt", optimizer.step)
            optimizer._rank_rows_timed = True

    def end(self, step, tokens):
        current, self.current = self.current, None
        if current is None:
            return
        current["end"] = _event()
        now = time.perf_counter()
        current.update(
            step=step,
            tokens=tokens,
            time=time.time(),
            wall=now - self.last_end if self.last_end is not None else None,
            next=self.data_wait if self.has_loader else None,
            mem_allocated_gib=torch.cuda.memory_allocated() / 2**30,
            mem_alloc_peak_gib=torch.cuda.max_memory_allocated() / 2**30,
            mem_reserved_peak_gib=torch.cuda.max_memory_reserved() / 2**30,
        )
        self.last_end = now
        self.data_wait = 0.0
        self.pending.append(current)
        if step % self.push_every == 0:
            self.flush()

    def flush(self, final=False):
        if final and self.pending:
            torch.cuda.synchronize()
        while self.pending and all(e.query() for e in _events(self.pending[0])):
            try:
                row = _row(self.pending.pop(0))
                self.written.append(row)
                ranked_push(**row)
                self.path = get_store().path
            except Exception as exc:  # never stop training for a metrics row
                print(f"[megatron-glm5] rank row failed: {exc!r}", flush=True)


def _events(step):
    pairs = step["fwd"] + step["bwd"] + step["fwd_bwd"] + step["opt"]
    return [step["start"], step["end"], *(e for pair in pairs for e in pair)]


def _row(step):
    def ms(pairs):
        return sum(start.elapsed_time(end) for start, end in pairs)

    elapsed = step["start"].elapsed_time(step["end"]) / 1000
    row = {
        "kind": "step",
        "batch_id": step["step"],
        "task": "train",
        "time": step["time"],
        "elapsed": elapsed,
        "elapsed_source": "cuda_events",
        "wall": step["wall"],
        "local_tokens": step["tokens"],
        "rate": step["tokens"] / elapsed if elapsed > 0 else None,
        "next": step["next"],
        "mem_allocated_gib": step["mem_allocated_gib"],
        "mem_alloc_peak_gib": step["mem_alloc_peak_gib"],
        "mem_reserved_peak_gib": step["mem_reserved_peak_gib"],
    }
    # an empty cell means unavailable, never zero: no call, no column
    if step["fwd_bwd"]:
        row["fwd_bwd_ms"] = ms(step["fwd"] + step["bwd"] + step["fwd_bwd"])
    else:
        if step["fwd"]:
            row["fwd_ms"] = ms(step["fwd"])
        if step["bwd"]:
            row["bwd_ms"] = ms(step["bwd"])
    if step["opt"]:
        row["optimizer_ms"] = ms(step["opt"])
    return {k: v for k, v in row.items() if v is not None}


def header_rows():
    """This rank's identity (``worker``) and the group's ``config`` (``benchmate.rankmetrics.v1``, the same on every
    rank but ``pp_stage``). For MoE both data-parallel sizes are recorded: dense layers and experts."""
    from megatron.core import parallel_state as ps

    now = time.time()
    worker = {"kind": "worker", "time": now, "local_rank": int(os.environ.get("LOCAL_RANK", 0)),
              "host": socket.gethostname().split(".")[0], "cuda_device": torch.cuda.current_device()}
    try:
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        worker["gpu_uuid"] = str(props.uuid)
        worker["gpu_pci_bus_id"] = f"{props.pci_domain_id:08x}:{props.pci_bus_id:02x}:{props.pci_device_id:02x}.0"
    except Exception:
        pass
    try:
        run_name = json.loads(os.environ["MILABENCH_CONFIG"]).get("run_name")
    except Exception:
        run_name = None
    config = {
        "kind": "config",
        "schema": "benchmate.rankmetrics.v1",
        "time": now,
        "run_name": run_name,
        "world_size": torch.distributed.get_world_size(),
        "parallelism": {
            "dp": ps.get_data_parallel_world_size(),
            "expert_dp": ps.get_expert_data_parallel_world_size(),
            "tp": ps.get_tensor_model_parallel_world_size(),
            "pp": ps.get_pipeline_model_parallel_world_size(),
            "cp": ps.get_context_parallel_world_size(),
            "ep": ps.get_expert_model_parallel_world_size(),
        },
        "pp_stage": ps.get_pipeline_model_parallel_rank(),
    }
    return [worker, config]


class StepTimer:
    """``push_every``: flush the queued step events every N steps (each flush waits for the GPU) and at ``flush()``."""

    def __init__(self, log, push_every=None):
        self.log = log
        self.timer = None
        self.push_every = max(int(push_every or os.getenv("BENCHMATE_PUSH_EVERY", 10)), 1)
        self.steps = 0
        self.rows = RankRows(self.push_every)

    def _push(self, **event):
        event.setdefault("task", "train")
        self.log({k: v for k, v in event.items() if v is not None})

    def wrap_loader(self, loader):
        """The wrapped loader is a drop-in replacement: Megatron only iterates it."""
        latency = float(os.getenv("BENCHMATE_FAKE_DATA_LATENCY", 0))
        if latency > 0:  # inside the timer: the stall is part of ``next``
            loader = SlowLoader(loader, latency, int(os.getenv("BENCHMATE_FAKE_DATA_EVERY", 5)))
        loader = DataWait(loader, self.rows)
        if get_rank() > 0:  # the rate is rank 0's; the other ranks' steps are their rows
            return loader
        self.timer = LocalBatchTimer(
            loader,
            event_fn=default_event(),
            rank=max(get_rank(), 0),
            push=self._push,
            device=default_device(),
            earlystop=10**9,
            raise_stop_program=False,
            batch_size_fn=lambda microbatch: 0,  # counted per step instead (see step)
        )
        return self.timer

    def step(self, global_tokens):
        """Call on every rank after each optimizer step (ranks without a train loader have no timer)."""
        if self.timer is not None:
            if self.timer.start is None:  # the loader was never iterated: the clock starts now
                self.timer.start = self.timer.event_fn(enable_timing=True)
                self.timer.start.record()
            self.timer.step(batch_override=global_tokens)
            self.steps += 1
            if self.steps % self.push_every == 0:
                self.timer._push()

    def flush(self):
        """Push what is still queued (end of the run: the loader's own end-of-iteration flush never runs under Megatron)."""
        if self.timer is not None and self.timer.events:
            self.timer._push()
        self.rows.flush(final=True)

    def restart_clock(self):
        """Restart the step clock now: whatever ran since the last ``step`` (e.g. a metrics report) is not counted."""
        if self.timer is not None and self.timer.start is not None:
            self.timer.start = self.timer.event_fn(enable_timing=True)
            self.timer.start.record()


def install(log):
    """Wrap the train dataloader Megatron builds, time the schedules' forward / backward calls, return the StepTimer."""
    import megatron.training.training as training
    from megatron.core.pipeline_parallel import combined_1f1b, schedules

    steps = StepTimer(log)
    # module globals: every schedule (no pipelining, 1F1B, interleaved) calls them through these names
    schedules.forward_step = steps.rows.timed("fwd", schedules.forward_step)
    schedules.backward_step = steps.rows.timed("bwd", schedules.backward_step)
    combined_1f1b.combined_forward_backward_step = steps.rows.timed(
        "fwd_bwd", combined_1f1b.combined_forward_backward_step
    )
    build = training.build_train_valid_test_data_loaders

    def build_wrapped(*args, **kwargs):
        train, valid, test = build(*args, **kwargs)
        if train is not None:
            train = steps.wrap_loader(train)
        return train, valid, test

    training.build_train_valid_test_data_loaders = build_wrapped
    return steps
