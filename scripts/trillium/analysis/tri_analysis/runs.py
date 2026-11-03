"""Trillium additions to the run data (../../FORMAT.md section 1): layout, precision recipe,
experiment ID, loss events and checkpoint events.

The common fields (throughput, MFU, memory, startup, ranks, EP groups) come from the Rorqual
parser (``tt_analysis.runs.parse_run``); this module only reads what it does not.
"""

from __future__ import annotations

import sys
from pathlib import Path

from tt_analysis.runs import read_events

# Checkpoint events: the TorchTitan pack's names -> FORMAT.md's (one shape for both workloads).
CKPT_RENAME = {"kind": "content", "async_mode": "mode", "fs_type": "filesystem", "rep": "repetition"}
CKPT_VALUES = {"content": {"model": "weights"},
               "mode": {"disabled": "sync", "async_with_pinned_mem": "async_pinned"}}


def normalize_checkpoint(c: dict) -> dict:
    """One checkpoint event in the FORMAT.md shape; {min, max} over ranks -> max (and *_min)."""
    out = {}
    for k, v in c.items():
        k = CKPT_RENAME.get(k, k)
        if isinstance(v, dict) and "max" in v:
            out[f"{k}_min"] = v.get("min")
            v = v["max"]
        out[k] = CKPT_VALUES.get(k, {}).get(v, v)
    if out.get("op") == "load":
        out.setdefault("mode", "sync")
    return out


def parse_extras(path: Path, run_id: str):
    """One data file -> (run columns, loss rows, checkpoint rows, batch_id of each rate event)."""
    summary, run_start, losses, ckpts, batch_ids = None, None, [], [], []
    no_step = 0
    for e in read_events(path):
        if e.get("event") != "data":
            continue
        d = e.get("data") or {}
        if "rate" in d and d.get("task") == "train":
            batch_ids.append(d.get("batch_id", d.get("step")))
        if d.get("loss") is not None and d.get("task", "train") == "train" and "rate" not in d:
            # Own event (FORMAT.md); TorchTitan numbers the step batch_id, Megatron step.
            step = d.get("batch_id", d.get("step"))
            no_step += step is None
            losses.append({"run_id": run_id, "step": step, "loss": d["loss"], "time": d.get("time")})
        elif "checkpoint" in d:
            ckpts.append({"run_id": run_id, **normalize_checkpoint(d["checkpoint"])})
        elif "summary" in d:
            summary = d["summary"]
        elif "run_start" in d:
            run_start = d["run_start"]
    src = summary or run_start or {}
    r, w, p = src.get("run", {}), src.get("workload", {}), src.get("parallelism", {})
    m, st = (summary or {}).get("measurement", {}), (summary or {}).get("startup", {})
    row = {
        "experiment": r.get("experiment"),
        "node_set": r.get("node_set"),
        "condition": r.get("condition"),
        "layout": p.get("layout"),
        "pp_schedule": p.get("pp_schedule"),
        "vpp": p.get("vpp"),
        "cp": p.get("cp", 1),
        "copies": p.get("copies"),
        "microbatches": p.get("microbatches_per_pipeline"),
        "expert_dp": p.get("expert_dp"),
        "grad_reduce": w.get("grad_reduce_dtype"),
        "params_total": w.get("params_total"),
        "params_active": w.get("params_active"),
        "seed": w.get("seed"),
        "measure_start": m.get("measurement_start_epoch"),
        "measure_end": m.get("measurement_end_epoch"),
        "first_step_total_s": (st.get("first_step_done") or {}).get("max_s"),
    }
    if not summary and run_start:  # failed run: settings from run_start
        row.update({"model": w.get("model"), "precision": w.get("precision"),
                    "tokens_per_gpu": w.get("tokens_per_gpu_per_step"),
                    "ac": w.get("activation_checkpointing"), "gpus": p.get("gpus")})
    for c in ckpts:  # throughput is derived, never logged
        c["gb_s"] = c["bytes"] / c["total_s"] / 1e9 if c.get("total_s") else None
    if no_step:  # kept (step empty) but unusable for C-loss: the event lacks batch_id / step
        print(f"[build] {path.parent.name}: {no_step} of {len(losses)} loss events without batch_id/step",
              file=sys.stderr)
    row["loss_events"], row["loss_events_no_step"] = len(losses), no_step
    if any(b is None for b in batch_ids):
        batch_ids = []
    return row, losses, ckpts, batch_ids
