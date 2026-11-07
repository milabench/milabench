"""Trillium additions to the run data (../../FORMAT.md section 1): layout, precision recipe,
experiment ID, loss events and checkpoint events.

The common fields (throughput, MFU, memory, startup, ranks, EP groups) come from the Rorqual
parser (``tt_analysis.runs.parse_run``); this module only reads what it does not.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import tt_analysis.runs as tt_runs
from tt_analysis.runs import percentile
from tt_analysis.runs import read_events as _read_events

from .machine import GPUS_PER_NODE
from .style import GLM, QWEN

# Checkpoint events: the TorchTitan pack's older names -> FORMAT.md's, for the test runs that have them.
CKPT_RENAME = {"kind": "content", "async_mode": "mode", "rep": "repetition"}
CKPT_VALUES = {"content": {"model": "weights"},
               "mode": {"disabled": "sync", "async_with_pinned_mem": "async_pinned"}}


def canonical_model(name):
    """One name per workload: the benchmarks write the Hugging Face name (GLM: ``GLM-5.3``)."""
    low = str(name).lower()
    return GLM if low.startswith("glm") else QWEN if low.startswith("qwen") else name


def garbled_summary(path: Path) -> dict | None:
    """The summary event whose smuggled line came out garbled: the monitor thread wrote another escape sequence inside it (inferred;
    1240808, 1246920). It stays in a very long ``line`` event, hex-encoded (as glm/ag/recover.py on Trillium)."""
    for e in _read_events(path):
        text = e.get("data") if e.get("event") == "line" else None
        if not isinstance(text, str) or len(text) < 20000 or "\x1b[" not in text:
            continue
        body = re.sub(r"\[[0-?]*z", "", text[text.index("\x1b[") + 2:])
        try:
            return json.loads(bytes.fromhex(body[:body.index("z")].translate(str.maketrans(":;<=>?", "abcdef"))).decode())["summary"]
        except (ValueError, KeyError):
            continue
    return None


def _clean_events(path: Path):
    """``read_events`` for the Rorqual parser: the TorchTitan pack writes ``per_ep_group: null`` (no MoE),
    where the parser expects a list. A run whose summary came out garbled gets it back from its line."""
    found = False
    for e in _read_events(path):
        d = e.get("data")
        s = d.get("summary") if isinstance(d, dict) else None
        if isinstance(s, dict):
            found = True
            for key in ("per_ep_group", "per_rank"):
                if s.get(key) is None:
                    s[key] = []
        yield e
    if not found and (s := garbled_summary(path)) is not None:
        yield {"event": "data", "data": {"summary": s, "task": "train"}}


tt_runs.read_events = read_events = _clean_events  # this process reads Trillium's run data only


def expected_operations(entry: str):
    """(content, [(op, mode), ...]) an F1 entry does, in order, from its name; None if it does not say.
    Megatron: one operation per run (...-ckpt-full-sync-save-n13, ...-ckpt-weights-load-n13). TorchTitan: a save
    and its load (tri-ckpt-full-pinned)."""
    m = re.search(r"(?:^|-)(?:ckpt|ck)(?:-ep\d+)?-(full|weights|model)(?:-ep\d+)?-(sync-save|async-save|load|sync|async|pinned)"
                  r"(?:-|$)", entry or "")
    if m is None:
        return None
    content = "weights" if m[1] == "model" else m[1]
    mode = {"pinned": "async_pinned"}.get(m[2], m[2].removesuffix("-save"))
    if m[2] == "load":
        return content, [("load", "sync")]
    return content, [("save", mode)] if m[2].endswith("-save") else [("save", mode), ("load", "sync")]


def failed_operation(row: dict, events: list[dict]):
    """The checkpoint operation a failed F1 run did not finish: the first of its entry's without an event."""
    expected = expected_operations(row.get("variant")) or expected_operations(row.get("bench"))
    if expected is None:
        return None
    content, ops = expected
    done = {(e.get("op"), e.get("mode")) for e in events}
    for op, mode in ops:
        if (op, mode) not in done:
            return {"run_id": row["run_id"], "content": content, "op": op, "mode": mode}
    return None


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
    elapsed, items = [], []
    no_step = 0
    rates_t0 = None  # absolute time of t_s = 0 in the steps table
    for e in read_events(path):
        if e.get("event") != "data":
            continue
        d = e.get("data") or {}
        if "rate" in d and d.get("task") == "train":
            batch_ids.append(d.get("batch_id", d.get("step")))
            if rates_t0 is None:
                rates_t0 = d["time"] - d.get("elapsed", 0)
            elapsed.append(d.get("elapsed"))
            items.append(float(d["rate"]) * d["elapsed"] if d.get("elapsed") else None)
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
        # The ranks' process starts, max - min: milabench's launch before them (about 12.5 s per node, Nov 7) is not in first_step_total_s
        # as long as this stays small.
        "start_spread_s": ((st.get("process_start_epoch") or {}).get("max") or 0) - ((st.get("process_start_epoch") or {}).get("min") or 0)
        if (st.get("process_start_epoch") or {}).get("max") is not None else None,
        "rates_t0": rates_t0,
    }
    if not summary and run_start:  # failed run: settings from run_start
        hosts = (run_start.get("placement") or {}).get("hosts") or []
        gpus = p.get("gpus") or GPUS_PER_NODE * len(hosts) or None
        tokens = w.get("tokens_per_gpu_per_step")
        row.update({"model": canonical_model(w.get("model")), "precision": w.get("precision"),
                    "tokens_per_gpu": tokens, "ac": w.get("activation_checkpointing"), "gpus": gpus,
                    "seq_len": w.get("seq_len"), "mb_seqs": w.get("microbatch_sequences"),
                    "global_batch_tokens": w.get("global_batch_tokens") or (tokens * gpus if tokens and gpus else None),
                    "full_model": w.get("n_layers") == w.get("n_layers_full")})
    for c in ckpts:  # throughput is derived, never logged
        c["gb_s"] = c["bytes"] / c["total_s"] / 1e9 if c.get("total_s") else None
    if no_step:  # kept (step empty) but unusable for C-loss: the event lacks batch_id / step
        print(f"[build] {path.parent.name}: {no_step} of {len(losses)} loss events without batch_id/step",
              file=sys.stderr)
    row["loss_events"], row["loss_events_no_step"] = len(losses), no_step
    if any(b is None for b in batch_ids):
        batch_ids = []
    return row, losses, ckpts, batch_ids, elapsed, items


def real_step_times(row: dict, steps: list[dict], elapsed: list, items: list) -> None:
    """Step times are the time measured, not declared tokens / rate: the rate counts valid tokens, and a Qwen micro-batch of 49,152 slots hits
    the packer's 64 documents and pads about 15 % of them. ``valid_token_share``: valid tokens / the tokens declared."""
    if len(elapsed) != len(steps) or not steps or any(e is None for e in elapsed):
        return
    for s, e in zip(steps, elapsed):
        s["step_s"] = e
    if row.get("global_batch_tokens"):
        row["valid_token_share"] = sum(items) / len(items) / row["global_batch_tokens"]
    row["step_s"], row["step_s_p10"], row["step_s_p90"] = percentile(elapsed, 0.5), percentile(elapsed, 0.1), percentile(elapsed, 0.9)


def drop_warmup(row: dict, steps: list[dict]) -> list[dict]:
    """Megatron's rate events include the warm-up steps (TorchTitan's start after them): the official
    value is the median of the measured steps only, as the summary's ``measurement`` says."""
    warmup, measured = row.get("warmup_steps"), row.get("summary_measured_steps")
    if not warmup or not measured or len(steps) != warmup + measured:
        return steps
    steps = steps[warmup:]
    rates = [s["tok_s_gpu"] for s in steps]
    row.update({
        "measured_steps": len(rates),
        "tok_s_gpu": percentile(rates, 0.5),
        "tok_s_gpu_p10": percentile(rates, 0.10),
        "tok_s_gpu_p90": percentile(rates, 0.90),
        "step_s": percentile([s["step_s"] for s in steps], 0.5),
        "step_s_p10": percentile([s["step_s"] for s in steps], 0.10),
        "step_s_p90": percentile([s["step_s"] for s in steps], 0.90),
    })
    return steps
