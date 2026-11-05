"""Checkpoint save / load timing (Trillium F1), after the measurement.

The trainer's CheckpointManager is built with ``load_only=True`` (tt_native.configs),
so nothing is saved while training. Once the measurement is done, every rank runs
:func:`run_checkpoint`: ``checkpoint_repeats`` times, save through TorchTitan (DCP),
load it back, delete it. Events: scripts/trillium/FORMAT.md ("Checkpoint events").

- save ``blocked_s``: until ``save()`` returns, plus the device-to-host staging for
  ``async_with_pinned_mem``; ``total_s``: until the write has completed. The first
  ``async_pinned`` save also allocates the pinned buffers, so its ``blocked_s`` is
  an upper bound (``pinned_buffers_allocated``).
- Before each load, one rank per node drops the checkpoint from its page cache, or
  the load would time the client's cache, not the filesystem.

``content``: ``full`` = model + optimizer + LR scheduler + data loader + train state;
``weights`` = model only, bf16 (TorchTitan's last-step model-only save, sync only).
"""

from __future__ import annotations

import os
import shutil
import time
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist

from . import instrument
from .instrument import _gather_floats, _rank

MODES = {"disabled": "sync", "async": "async", "async_with_pinned_mem": "async_pinned"}


def _drop_page_cache(path: str) -> None:
    """Drop this node's cached pages of every file under ``path``."""
    for root, _, names in os.walk(path):
        for name in names:
            fd = os.open(os.path.join(root, name), os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)


def _size(path: str) -> int:
    return sum(os.path.getsize(os.path.join(root, name)) for root, _, names in os.walk(path) for name in names)


def _peak_gib() -> float:
    return torch.cuda.max_memory_allocated() / 2**30


def run_checkpoint(trainer, push, events: list[dict[str, Any]]) -> None:
    """Save / load ``checkpoint_repeats`` times; rank 0 pushes one ``checkpoint``
    event per operation and appends it to ``events``."""
    from torchtitan.components.checkpointer.base import MODEL

    opts = instrument.RUN_DESCRIPTION.get("options", {})
    content = "weights" if opts.get("checkpoint") == "model" else "full"
    mode = MODES[opts.get("checkpoint_async", "disabled")]
    repeats = int(opts.get("checkpoint_repeats", 1))
    engine = trainer.engine
    ckpt = engine.checkpointer
    step = engine.num_completed_steps
    path = ckpt._create_checkpoint_id(step)

    # Collectives here wait for whole saves / loads: the training timeout
    # (train_timeout_seconds) is far too short for them.
    from torchtitan.distributed import utils as dist_utils

    timeout = timedelta(seconds=int(opts.get("checkpoint_timeout_seconds", 1800)))
    dist_utils.set_pg_timeouts(timeout=timeout, parallelism_context=engine.parallelism_context)
    if getattr(ckpt, "pg", None) is not None:  # async saves' gloo group
        dist.set_timeout(timeout, ckpt.pg)

    def record(event: dict[str, Any], local: list[float]) -> None:
        # Every rank's [start, end_blocked, end_total, peak GiB] -> the slowest, on rank 0.
        rows = _gather_floats(local)
        if _rank() != 0:
            return
        event.update(
            {
                "step": step,
                "content": content,
                "format": "torch.dcp",
                "bytes": _size(path),
                "path": ckpt.folder,  # the step is in step-<step>/
                "start_epoch": min(r[0] for r in rows),
                "end_epoch": max(r[2] for r in rows),
                "blocked_s": max(r[1] - r[0] for r in rows),
                "total_s": max(r[2] - r[0] for r in rows),
                "params_total": getattr(engine, "model_param_count", None),
                "peak_allocated_gib": max(r[3] for r in rows),
            }
        )
        events.append(event)
        push(checkpoint=event, task="train")
        print(f"[tt-native] checkpoint {event}", flush=True)

    for rep in range(1, repeats + 1):
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        ckpt.load_only = False
        try:
            if content == "full":
                ckpt.save(step)
            else:
                ckpt.save(step, last_step=True)
        finally:
            ckpt.load_only = True
        if mode == "async_pinned":
            ckpt.maybe_wait_for_staging()
        t_blocked = time.time()
        ckpt.maybe_wait_for_saving()
        t_total = time.time()
        dist.barrier()
        event: dict[str, Any] = {"op": "save", "repetition": rep, "mode": mode}
        if mode == "async_pinned":
            event["pinned_buffers_allocated"] = rep == 1
        record(event, [t0, t_blocked, t_total, _peak_gib()])

        if int(os.environ.get("LOCAL_RANK", "0")) == 0:
            _drop_page_cache(path)
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        if content == "full":
            if not ckpt.load(step=step):
                raise RuntimeError(f"checkpoint load {rep}/{repeats} of step {step}: nothing loaded")
        else:
            ckpt._load_checkpoint(
                {MODEL: ckpt.states[MODEL]}, path, from_hf=False, from_quantized=False
            )
        torch.cuda.synchronize()
        t1 = time.time()
        peak = _peak_gib()
        dist.barrier()
        record({"op": "load", "repetition": rep, "mode": "sync"}, [t0, t1, t1, peak])

        if _rank() == 0:
            shutil.rmtree(path, ignore_errors=True)
            if not os.listdir(ckpt.folder):
                os.rmdir(ckpt.folder)
        dist.barrier()
