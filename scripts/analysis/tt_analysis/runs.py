"""Milabench run data (``<run>/<bench>.<n>.data``, voir JSONL) -> run, step and rank rows.

Accepts the TorchTitan ``summary`` schema v2 (current pack) and the target
``milabench.pretrain.v1`` schema (FORMAT.md). Every run directory gives one row
in ``runs``, including runs that failed (``end`` event with a non-zero return
code, or no summary).
"""

from __future__ import annotations

import json
import math
import re
import statistics
from datetime import datetime
from pathlib import Path

from .topology import placement

# Runs before the v2 measurement path (2026-10-21) are excluded: different timing
# definition and configs (user decision 2026-10-25).
V2_SINCE = "2026-10-21"


def percentile(values, q):
    ordered = sorted(values)
    if not ordered:
        return math.nan
    k = (len(ordered) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def read_events(path: Path):
    for line in path.read_text(errors="replace").splitlines():
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def data_files(root: Path):
    return sorted(p for p in root.glob("runs/*/*.data") if ".nolog." not in p.name)


def _family_variant(tag: str | None, bench: str, nodes: int) -> tuple[str, str]:
    """(family, variant) from the run tag ("topo-H6-1" -> H, H6)."""
    if tag and (m := re.fullmatch(r"topo-([HSP])(\d+)-\d+", tag)):
        return m.group(1), f"{m.group(1)}{m.group(2)}"
    if tag and tag.startswith("topo-"):
        return tag[5:], tag[5:]
    return ("1node" if nodes == 1 else "slurm"), bench


def _v2_fields(s: dict) -> dict:
    """Common columns from a TorchTitan v2 summary."""
    c = s["config"]
    p = c["parallelism"]
    return {
        "model": f"{c.get('model', '?')}-{c.get('flavor', '')}".strip("-"),
        "framework": "torchtitan",
        "framework_commit": c.get("torchtitan_commit"),
        "torch": c.get("torch_version"),
        "n_layers": c.get("n_layers"),
        "n_layers_full": 64,
        "gpus": c["gpus"],
        "nodes": c["nodes"],
        "hosts": c["hosts"],
        "seq_len": c["seq_len"],
        "mb_seqs": c.get("microbatch_sequences_per_gpu"),
        "grad_accum": c.get("gradient_accumulation_steps", 1),
        "tokens_per_gpu": c["tokens_per_microbatch_per_dp_rank"] * c.get("gradient_accumulation_steps", 1),
        "global_batch_tokens": c["global_batch_tokens"],
        "dp_replicate": p["dp_replicate"],
        "dp_shard": p["dp_shard"],
        "tp": p["tp"],
        "pp": p["pp"],
        "ep": 1,
        "compile": c.get("compile"),
        "ac": c.get("activation_checkpointing"),
        "adam_moments": c.get("adam_moments"),
        "precision": "bf16",
        "optimizer": "AdamW",
        "job_id": c.get("slurm_job_id"),
        "run_tag": c.get("run_tag"),
        "campaign": "prelim",
        "repetition": None,
        "excluded_nodes": None,
        "deviations": "AdamW (not Muon); MTP off; random tokens",
    }


def _v3_fields(s: dict) -> dict:
    """Common columns from a milabench.pretrain.v1 summary (FORMAT.md)."""
    r, w, par, pl = s["run"], s["workload"], s["parallelism"], s["placement"]
    return {
        "model": w["model"],
        "framework": w["framework"],
        "framework_commit": w.get("framework_commit"),
        "torch": w.get("torch_version"),
        "n_layers": w["n_layers"],
        "n_layers_full": w["n_layers_full"],
        "params_trainable_fraction": w.get("params_trainable_fraction", 1.0),
        "gpus": par["gpus"],
        "nodes": par["nodes"],
        "hosts": pl["hosts"],
        "seq_len": w["seq_len"],
        "mb_seqs": w["microbatch_sequences"],
        "grad_accum": w["grad_accum"],
        "tokens_per_gpu": w["tokens_per_gpu_per_step"],
        "global_batch_tokens": w["global_batch_tokens"],
        "dp_replicate": par.get("dp_replicate", 1),
        "dp_shard": par.get("dp_shard", par["gpus"]),
        "tp": par.get("tp", 1),
        "pp": par.get("pp", 1),
        "ep": par.get("ep", 1),
        "compile": w.get("compile"),
        "ac": w.get("activation_checkpointing"),
        "adam_moments": w.get("optimizer_state_dtype"),
        "precision": w["precision"],
        "optimizer": w["optimizer"],
        "job_id": r.get("slurm_job_id"),
        "run_tag": r.get("variant"),
        "campaign": r.get("campaign"),
        "repetition": r.get("repetition"),
        "family": pl.get("family"),
        "variant": r.get("variant"),
        "excluded_nodes": ";".join(f"{h}:{why}" for h, why in pl.get("excluded_nodes", {}).items()) or None,
        "deviations": "; ".join(d["text"] for d in s.get("deviations", [])),
    }


def parse_run(path: Path, cluster: str, leaf_of: dict[str, str], synthetic: bool = False):
    """One data file -> (run row, step rows, rank rows), or None if out of scope."""
    run_name = path.parent.name
    bench = path.name.split(".")[0]
    summary, run_start, rates, rc, start_time = None, None, [], None, None
    for e in read_events(path):
        ev, d = e.get("event"), e.get("data") or {}
        if ev == "start":
            start_time = d.get("time")
        elif ev == "end":
            rc = d.get("return_code")
        elif ev == "data":
            if "rate" in d and d.get("task") == "train":
                rates.append(d)
            elif "summary" in d:
                summary = d["summary"]
            elif "run_start" in d:
                run_start = d["run_start"]
    started = datetime.fromtimestamp(start_time) if start_time else None
    if summary is None and (started is None or started.strftime("%Y-%m-%d") < V2_SINCE):
        return None
    if summary is None and rc == 0:
        return None  # legacy (v1) record path: no summary event

    row = {
        "run_id": f"{cluster}:{run_name}:{bench}",
        "cluster": cluster,
        "synthetic": synthetic,
        "run_name": run_name,
        "bench": bench,
        "started": started.isoformat(timespec="seconds") if started else None,
        "return_code": rc,
        "status": "completed" if summary is not None and rc == 0 else
                  ("partial" if summary is not None else "failed"),
        "schema": summary.get("schema") if summary else None,
    }
    if summary is None:
        if run_start:  # failed before the summary: identity from run_start
            r, pl = run_start["run"], run_start["placement"]
            row.update({
                "job_id": r.get("slurm_job_id"), "campaign": r.get("campaign"),
                "variant": r.get("variant"), "run_tag": r.get("variant"),
                "repetition": r.get("repetition"), "family": pl.get("family"),
                "nodes": len(pl["hosts"]), "hosts": ";".join(pl["hosts"]),
            })
        return row, [], [], []

    fields = _v3_fields(summary) if "run" in summary else _v2_fields(summary)
    hosts = fields.pop("hosts")
    row.update(fields)
    row["hosts"] = ";".join(hosts)
    row.update(placement(hosts, leaf_of))
    if not row.get("family"):
        row["family"], row["variant"] = _family_variant(row["run_tag"], bench, row["nodes"])
    row["full_model"] = row["n_layers"] == row["n_layers_full"]

    values = [float(r["rate"]) for r in rates]
    gpus, tokens = row["gpus"], row["global_batch_tokens"]
    if values:
        med = statistics.median(values)
        steps_s = [tokens / v for v in values]
        row.update({
            "measured_steps": len(values),
            "tok_s_gpu": med / gpus,
            "tok_s_gpu_p10": percentile(values, 0.10) / gpus,
            "tok_s_gpu_p90": percentile(values, 0.90) / gpus,
            "step_s": tokens / med,
            "step_s_p10": percentile(steps_s, 0.10),
            "step_s_p90": percentile(steps_s, 0.90),
        })
    mfu = summary["mfu_inputs"]
    row["flops_per_token"] = mfu["flops_per_token"]
    row["peak_flops_per_gpu"] = mfu["peak_flops_per_gpu"]
    if values and mfu.get("peak_flops_per_gpu"):
        row["mfu_pct"] = 100 * mfu["flops_per_token"] * row["tok_s_gpu"] / mfu["peak_flops_per_gpu"]
    mem = summary["memory_gib"]
    row["mem_alloc_gib"] = mem["peak_allocated"]["max"]
    row["mem_reserved_gib"] = mem["peak_reserved"]["max"]
    st = summary["startup"]
    row["init_s"] = st["model_initialized"]["max_s"]
    row["first_step_s"] = st["first_step_done"]["max_s"] - st["model_initialized"]["max_s"]
    row["to_measurement_s"] = st["measurement_start"]["max_s"]
    row["compile_cache_hits"] = (st.get("compile_rank0") or {}).get("fxgraph_cache_hit")

    t0 = rates[0]["time"] - rates[0].get("elapsed", 0) if rates else 0
    steps = [
        {
            "run_id": row["run_id"],
            "step": r.get("step", i),
            "t_s": r["time"] - t0,
            "tok_s_gpu": float(r["rate"]) / gpus,
            "step_s": tokens / float(r["rate"]),
        }
        for i, r in enumerate(rates)
    ]
    groups = summary.get("placement", {}).get("ep_groups") or []
    if groups:
        crossing = sum(len({leaf_of.get(h, "?") for h in g}) > 1 for g in groups)
        row["ep_groups"] = len(groups)
        row["ep_groups_crossing"] = crossing
    ep_rows = [
        {"run_id": row["run_id"], "group": i, "hosts": ";".join(g["hosts"]),
         "n_switches": len({leaf_of.get(h, "?") for h in g["hosts"]}), "a2a_s": g.get("a2a_s_per_step")}
        for i, g in enumerate(summary.get("per_ep_group", []))
    ]
    ranks = [
        {
            "run_id": row["run_id"],
            "rank": pr["rank"],
            "host": pr["host"],
            "leaf": leaf_of.get(pr["host"], "?"),
            "local_rank": pr.get("local_rank"),
            "gpu_uuid": pr.get("gpu_uuid"),
            "step_median": pr["step_median"],
            "step_p90": pr["step_p90"],
            "step_max": pr["step_max"],
            "compute_s": pr.get("compute_s_per_step"),
            "nccl_s": pr.get("nccl_s_per_step"),
        }
        for pr in summary.get("per_rank", [])
    ]
    return row, steps, ranks, ep_rows
