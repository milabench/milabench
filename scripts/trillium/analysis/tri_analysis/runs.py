"""Trillium additions to the run data (../../FORMAT.md section 1): layout, precision recipe,
experiment ID, loss events and checkpoint events.

The common fields (throughput, MFU, memory, startup, ranks, EP groups) come from the Rorqual
parser (``tt_analysis.runs.parse_run``); this module only reads what it does not.
"""

from __future__ import annotations

import sys
from pathlib import Path

import tt_analysis.runs as tt_runs
from tt_analysis.runs import percentile, read_events

from .style import GLM, QWEN

# Checkpoint events: the TorchTitan pack's older names -> FORMAT.md's, for the test runs that have them.
CKPT_RENAME = {"kind": "content", "async_mode": "mode", "fs_type": "filesystem", "rep": "repetition",
               "restored": "checksum_match"}
CKPT_VALUES = {"content": {"model": "weights"},
               "mode": {"disabled": "sync", "async_with_pinned_mem": "async_pinned"}}


def canonical_model(name):
    """One name per workload: the benchmarks write the Hugging Face name (GLM: ``GLM-5.3``)."""
    low = str(name).lower()
    return GLM if low.startswith("glm") else QWEN if low.startswith("qwen") else name


def _clean_events(path: Path):
    """``read_events`` for the Rorqual parser: the TorchTitan pack writes ``per_ep_group: null`` (no MoE),
    where the parser expects a list."""
    for e in read_events(path):
        d = e.get("data")
        s = d.get("summary") if isinstance(d, dict) else None
        if isinstance(s, dict):
            for key in ("per_ep_group", "per_rank"):
                if s.get(key) is None:
                    s[key] = []
        yield e


tt_runs.read_events = _clean_events  # this process reads Trillium's run data only


def normalize_checkpoint(c: dict) -> dict:
    """One checkpoint event in the FORMAT.md shape; {min, max} over ranks -> max (and *_min); nested
    ``saved_by`` -> ``saved_by_mode``, ``saved_by_run``."""
    out = {}
    for k, v in c.items():
        k = CKPT_RENAME.get(k, k)
        if k == "saved_by":
            out["saved_by_mode"], out["saved_by_run"] = (v or {}).get("mode"), (v or {}).get("run")
            continue
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
            losses.append({"run_id": run_id, "step": step, "loss": d["loss"], "grad_norm": d.get("grad_norm"),
                           "time": d.get("time")})
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
        "run_cluster": r.get("cluster"),
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
        "warmup_steps": m.get("warmup_steps"),
        "summary_measured_steps": m.get("measured_steps"),
        "measure_start": m.get("measurement_start_epoch"),
        "measure_end": m.get("measurement_end_epoch"),
        "first_step_total_s": (st.get("first_step_done") or {}).get("max_s"),
    }
    if not summary and run_start:  # failed run: settings from run_start
        row.update({"model": canonical_model(w.get("model")), "precision": w.get("precision"),
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


def drop_warmup(row: dict, steps: list[dict]) -> list[dict]:
    """Megatron's rate events include the warm-up steps (TorchTitan's start after them): the official
    value is the median of the measured steps only, as the summary's ``measurement`` says."""
    warmup, measured = row.get("warmup_steps"), row.get("summary_measured_steps")
    if not warmup or not measured or len(steps) != warmup + measured:
        return steps
    steps = steps[warmup:]
    rates = [s["tok_s_gpu"] for s in steps]
    tokens, gpus = row["global_batch_tokens"], row["gpus"]
    row.update({
        "measured_steps": len(rates),
        "tok_s_gpu": percentile(rates, 0.5),
        "tok_s_gpu_p10": percentile(rates, 0.10),
        "tok_s_gpu_p90": percentile(rates, 0.90),
        "step_s": tokens / (percentile(rates, 0.5) * gpus),
        "step_s_p10": percentile([s["step_s"] for s in steps], 0.10),
        "step_s_p90": percentile([s["step_s"] for s in steps], 0.90),
    })
    return steps
