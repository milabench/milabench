"""Checkpoint save / load timing (Trillium F1), after the measurement.

The trainer's CheckpointManager is built with ``load_only=True`` (tt_native.configs),
so nothing is saved while training. Once the measurement is done, every rank
runs :func:`run_checkpoint`: ``checkpoint_repeats`` times, save the current state
through TorchTitan (DCP), time it, count what was written, overwrite the model
weights, load them back, check they are restored (load event ``checksum_match``),
and delete the folder (the last one is kept with ``checkpoint_keep``). A load whose
check is false or missing fails the run, once its event is recorded and the folder
dealt with. The summary, pushed afterwards, even after that failure, lists the
operations in ``summary.checkpoint``.

Events follow scripts/trillium/FORMAT.md ("Checkpoint events"). Times (wall
clock, every rank): ``blocked_s`` / ``total_s`` are the maximum over ranks,
``*_min`` the minimum.

- save ``blocked_s``: until ``save()`` returns, plus the device-to-host staging
  for ``async_with_pinned_mem`` (what training would wait for); ``total_s``:
  until the write has completed (the save future resolves). The first
  ``async_pinned`` save of a process also allocates the pinned host buffers, so
  its ``blocked_s`` is an upper bound (event ``pinned_buffers_allocated``);
  later repetitions reuse them.
- load: until ``load()`` returns (synchronous).

``peak_allocated_gib``: peak GPU memory during the operation, maximum over ranks
(peak counters reset just before it; the summary's ``memory_gib``, gathered before
the first save, keeps the training peak). ``params_total``: parameters of the model.

Page cache: every rank reads back the shards it wrote seconds earlier, so a
load could be served from the client's page cache instead of the filesystem.
Before each load, one rank per node drops the step directory's pages
(``posix_fadvise(POSIX_FADV_DONTNEED)`` on every file; no root needed; the
writer has synced the files, so their pages are clean and droppable); the load
event records it as ``cache_drop``. Reads are not shifted to another node: with
the same world size DCP gives every rank its own shards back, and re-sharding
the load is not cheap, so the measurement relies on the cache drop.

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


def _filesystem(path: str) -> tuple[str | None, str | None]:
    """(filesystem, mount source) of ``path``: the longest mount point prefix of
    its real path in /proc/mounts. NFS from a VAST server (Trillium:
    ``trillium.vast.local:/trillium_scratch``) is ``vast-nfs``; other NFS mounts
    keep their type and name the server (``nfs4:<server>``)."""
    path = os.path.realpath(path)
    best, fstype, source = "", None, None
    try:
        with open("/proc/mounts") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 3 and (path == parts[1] or path.startswith(parts[1].rstrip("/") + "/")):
                    if len(parts[1]) >= len(best):
                        best, fstype, source = parts[1], parts[2], parts[0]
    except OSError:
        pass
    if fstype and fstype.startswith("nfs") and source:
        server = source.split(":", 1)[0]
        fstype = "vast-nfs" if "vast" in server.lower() else f"{fstype}:{server}"
    return fstype, source


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


CACHE_DROP = "posix_fadvise_dontneed"


def _drop_page_cache(path: str) -> int:
    """Drop this node's cached pages of every file under ``path``; returns the file count."""
    count = 0
    for root, _, names in os.walk(path):
        for name in names:
            try:
                fd = os.open(os.path.join(root, name), os.O_RDONLY)
            except OSError:
                continue
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                count += 1
            finally:
                os.close(fd)
    return count


def _local(t: torch.Tensor) -> torch.Tensor:
    return t._local_tensor if hasattr(t, "_local_tensor") else t


def _checksum(model_parts, dtype: torch.dtype | None = None) -> float:
    """Sum of squares of this rank's model weights, rounded to ``dtype`` first
    (bf16 for weights-only checkpoints), accumulated in fp64."""
    with torch.no_grad():
        total = torch.zeros((), dtype=torch.float64, device="cuda")
        for part in model_parts:
            for p in part.parameters():
                local = _local(p)
                if dtype is not None:
                    local = local.to(dtype)
                total += local.double().square().sum()
    return float(total)


def _zero_weights(model_parts) -> None:
    with torch.no_grad():
        for part in model_parts:
            for p in part.parameters():
                _local(p).zero_()


def _peak_gib() -> float:
    return torch.cuda.max_memory_allocated() / 2**30


def run_checkpoint(trainer, push, events: list[dict[str, Any]]) -> None:
    """Save / load ``checkpoint_repeats`` times; rank 0 pushes one ``checkpoint``
    event per operation and appends it to ``events``. Raises ``RuntimeError``, on every
    rank, when a load's checksum does not match or is missing."""
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
    # The folder does not exist yet: its parent's mount.
    fs_type, fs_source = _filesystem(os.path.dirname(os.path.abspath(ckpt.folder)))

    # Collectives here wait for whole saves / loads: the training timeout
    # (train_timeout_seconds) is far too short for them.
    from torchtitan.distributed import utils as dist_utils

    timeout = timedelta(seconds=int(opts.get("checkpoint_timeout_seconds", 1800)))
    dist_utils.set_pg_timeouts(timeout=timeout, parallelism_context=engine.parallelism_context)
    if getattr(ckpt, "pg", None) is not None:  # async saves' gloo group
        dist.set_timeout(timeout, ckpt.pg)

    def record(event: dict[str, Any], local: list[float]) -> bool | None:
        # Every rank's [start, end_blocked, end_total, peak GiB, ok] -> spreads on rank 0.
        # Returns, on every rank, whether all the ranks' checksums match (None: no check).
        rows = _gather_floats(local)
        matched = all(r[4] > 0 for r in rows) if local[4:] else None
        if _rank() != 0:
            return matched
        blocked = [r[1] - r[0] for r in rows]
        total = [r[2] - r[0] for r in rows]
        event.update(
            {
                "step": step,
                "content": content,
                "format": "torch.dcp",
                "path": ckpt.folder,  # the step is in step-<step>/
                "filesystem": fs_type,
                "filesystem_source": fs_source,
                "ranks": len(rows),
                "start_epoch": min(r[0] for r in rows),
                "end_epoch": max(r[2] for r in rows),
                "blocked_s": max(blocked),
                "blocked_s_min": min(blocked),
                "total_s": max(total),
                "total_s_min": min(total),
                "params_total": getattr(engine, "model_param_count", None),
                "peak_allocated_gib": max(r[3] for r in rows),
            }
        )
        if matched is not None:
            event["checksum_match"] = matched
        events.append(event)
        push(checkpoint=event, task="train")
        print(f"[tt-native] checkpoint {event}", flush=True)
        return matched

    for rep in range(1, repeats + 1):
        torch.cuda.synchronize()
        dist.barrier()
        # Weights-only checkpoints hold bf16 weights: the loaded fp32 weights
        # must equal the bf16-rounded ones (same values, same summation order).
        reference = _checksum(params, torch.bfloat16 if content == "weights" else None)
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
        if _rank() == 0:
            event["bytes"], event["files"] = _usage(path)
        record(event, [t0, t_blocked, t_total, _peak_gib()])

        # Load into overwritten weights: the checksum shows they come back.
        _zero_weights(params)
        dropped = 0
        if int(os.environ.get("LOCAL_RANK", "0")) == 0:
            dropped = _drop_page_cache(path)
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        if content == "full":
            ckpt.load(step=step)
        else:
            ckpt._load_checkpoint(
                {MODEL: ckpt.states[MODEL]}, path, from_hf=False, from_quantized=False
            )
        torch.cuda.synchronize()
        t1 = time.time()
        peak = _peak_gib()  # before the checksum's own allocations
        restored = _checksum(params)
        ok = abs(restored - reference) <= 1e-12 * max(abs(reference), 1.0)
        dist.barrier()
        event = {"op": "load", "repetition": rep, "mode": "sync", "cache_drop": CACHE_DROP}
        if _rank() == 0:
            event["cache_drop_files_rank0"] = dropped
        if _rank() == 0:
            event["bytes"], event["files"] = _usage(path)
        matched = record(event, [t0, t1, t1, peak, float(ok)])

        if _rank() == 0 and not (keep and rep == repeats):
            shutil.rmtree(path, ignore_errors=True)
            if not os.listdir(ckpt.folder):
                os.rmdir(ckpt.folder)
        dist.barrier()
        if matched is not True:
            raise RuntimeError(
                f"checkpoint load {rep}/{repeats} of step {step} ({content}): "
                f"checksum_match={matched}, the weights did not come back (rank {_rank()}: "
                f"{restored!r} loaded, {reference!r} saved)"
            )
