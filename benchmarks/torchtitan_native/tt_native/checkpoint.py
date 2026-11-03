"""Checkpoint save / load timing (Trillium F1), after the measurement.

The trainer's CheckpointManager is built with ``load_only=True`` (tt_native.configs),
so nothing is saved while training. Once the measurement is done, every rank
runs :func:`run_checkpoint`: ``checkpoint_repeats`` times, save the current state
through TorchTitan (DCP), time it, count what was written, overwrite the model
weights, load them back, check they are restored (load event ``restored``), and
delete the folder (the last one is kept with ``checkpoint_keep``). The summary,
pushed afterwards, lists the operations in ``summary.checkpoint``.

Events follow scripts/trillium/FORMAT.md ("Checkpoint events"). Times (wall
clock, every rank): ``blocked_s`` / ``total_s`` are the maximum over ranks,
``*_min`` the minimum.

- save ``blocked_s``: until ``save()`` returns, plus the device-to-host staging
  for ``async_with_pinned_mem`` (what training would wait for); ``total_s``:
  until the write has completed (the save future resolves).
- load: until ``load()`` returns (synchronous).

``content``: ``full`` = model + optimizer + LR scheduler + data loader + train
state (``CheckpointManager.save``); ``weights`` = model only, bf16
(TorchTitan's last-step model-only save, synchronous only), loaded back through
``_load_checkpoint`` with the model state alone.
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


def _fs_type(path: str) -> str | None:
    """Filesystem type of ``path`` (longest mount point prefix in /proc/mounts)."""
    best, fstype = "", None
    try:
        with open("/proc/mounts") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 3 and (path == parts[1] or path.startswith(parts[1].rstrip("/") + "/")):
                    if len(parts[1]) > len(best):
                        best, fstype = parts[1], parts[2]
    except OSError:
        pass
    return fstype


def _usage(path: str) -> tuple[int, int]:
    total, files = 0, 0
    for root, _, names in os.walk(path):
        for name in names:
            try:
                total += os.path.getsize(os.path.join(root, name))
                files += 1
            except OSError:
                pass
    return total, files


def _local(t: torch.Tensor) -> torch.Tensor:
    return t._local_tensor if hasattr(t, "_local_tensor") else t


def _checksum(model_parts) -> float:
    """Sum of squares of this rank's model weights (fp32)."""
    with torch.no_grad():
        total = torch.zeros((), dtype=torch.float64, device="cuda")
        for part in model_parts:
            for p in part.parameters():
                total += _local(p).double().square().sum()
    return float(total)


def _zero_weights(model_parts) -> None:
    with torch.no_grad():
        for part in model_parts:
            for p in part.parameters():
                _local(p).zero_()


def run_checkpoint(trainer, push, events: list[dict[str, Any]]) -> None:
    """Save / load ``checkpoint_repeats`` times; rank 0 pushes one ``checkpoint``
    event per operation and appends it to ``events``."""
    from torchtitan.components.checkpointer.base import MODEL

    opts = instrument.RUN_DESCRIPTION.get("options", {})
    content = "weights" if opts.get("checkpoint") == "model" else "full"
    mode = MODES[opts.get("checkpoint_async", "disabled")]
    repeats = int(opts.get("checkpoint_repeats", 1))
    keep = bool(opts.get("checkpoint_keep", False))
    engine = trainer.engine
    ckpt = engine.checkpointer
    step = engine.num_completed_steps
    path = ckpt._create_checkpoint_id(step)
    params = engine.model_parts
    fs_type = _fs_type(os.path.abspath(ckpt.folder))

    # Collectives here wait for whole saves / loads: the training timeout
    # (train_timeout_seconds) is far too short for them.
    from torchtitan.distributed import utils as dist_utils

    timeout = timedelta(seconds=int(opts.get("checkpoint_timeout_seconds", 1800)))
    dist_utils.set_pg_timeouts(timeout=timeout, parallelism_context=engine.parallelism_context)
    if getattr(ckpt, "pg", None) is not None:  # async saves' gloo group
        dist.set_timeout(timeout, ckpt.pg)

    def record(event: dict[str, Any], local: list[float]) -> None:
        # Every rank's [start, end_blocked, end_total, ok] -> spreads on rank 0.
        rows = _gather_floats(local)
        if _rank() != 0:
            return
        blocked = [r[1] - r[0] for r in rows]
        total = [r[2] - r[0] for r in rows]
        event.update(
            {
                "step": step,
                "content": content,
                "format": "torch.dcp",
                "path": ckpt.folder,  # the step is in step-<step>/
                "filesystem": fs_type,
                "ranks": len(rows),
                "start_epoch": min(r[0] for r in rows),
                "end_epoch": max(r[2] for r in rows),
                "blocked_s": max(blocked),
                "blocked_s_min": min(blocked),
                "total_s": max(total),
                "total_s_min": min(total),
            }
        )
        if local[3:]:
            event["restored"] = all(r[3] > 0 for r in rows)
        events.append(event)
        push(checkpoint=event, task="train")
        print(f"[tt-native] checkpoint {event}", flush=True)

    for rep in range(1, repeats + 1):
        torch.cuda.synchronize()
        dist.barrier()
        reference = _checksum(params)
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
        if _rank() == 0:
            event["bytes"], event["files"] = _usage(path)
        record(event, [t0, t_blocked, t_total])

        # Load into overwritten weights: the checksum shows they come back.
        _zero_weights(params)
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.time()
        if content == "full":
            ckpt.load(step=step)
        else:
            ckpt._load_checkpoint(
                {MODEL: ckpt.states[MODEL]}, path, from_hf=False, from_quantized=False
            )
        torch.cuda.synchronize()
        t1 = time.time()
        restored = _checksum(params)
        # weights are saved in bf16: compare at bf16 precision.
        tolerance = 1e-9 if content == "full" else 1e-2
        ok = abs(restored - reference) <= tolerance * max(abs(reference), 1.0)
        dist.barrier()
        event = {"op": "load", "repetition": rep, "mode": "sync"}
        if _rank() == 0:
            event["bytes"], event["files"] = _usage(path)
        record(event, [t0, t1, t1, float(ok)])

        if _rank() == 0 and not (keep and rep == repeats):
            shutil.rmtree(path, ignore_errors=True)
            if not os.listdir(ckpt.folder):
                os.rmdir(ckpt.folder)
        dist.barrier()
