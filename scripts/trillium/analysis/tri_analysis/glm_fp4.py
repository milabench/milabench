"""GLM-5 with NVFP4 (GLM-FP4-01, job 1271210): C-speed at 13 nodes, pp13-ep8, BF16 / MXFP8 / NVFP4 in one job, the campaign's
Megatron entries in the new workloads' own venv, the last 5 layers (the last stage) in BF16 under NVFP4. One row per entry, from its run directory under ``data/raw/trillium/glm-fp4/runs/``
and the GPU telemetry under ``glm-fp4/nodemon/<run>/`` (``analyze.py collect``, from ``$SCRATCH/trillium/new/glm-fp4/``). They stay
out of ``runs/``, so the campaign's tables, counts and ledger never see them (their campaign tag, ``trillium-new-1108``, is not a
campaign one either).

Read with the campaign's own parser (``tt_analysis.runs.parse_run`` and ``runs.parse_extras``): the same summary fields and the
``.0.data`` rate events, milabench's timer, which in this job equals rank 0's ``metrics`` elapsed to a few ms. Step times are over
steps 3-20, as the review's: step 1 is the warm-up, step 2 the slow first measured step (the rate events' ``batch_id`` counts from
0, so step = batch_id + 1). Memory: the summary's peak allocated, max over ranks. Loss: the loss event of step 20, after 20 steps at
the command line's constant learning rate (``--lr``, ``--lr-decay-style`` in the ``start`` event; 1e-6, constant, and Megatron's
argument dump has ``lr_warmup_iters`` 0), one run each: not a loss comparison. Software power cap: share of the GPU samples whose ``clocks_event_reasons`` has NVML's SwPowerCap bit (0x4),
over steps 3-20, per node.

Qwen3 32B dense, the same three recipes (Megatron + Transformer Engine, random weights, NVFP4 with its last 4 of 64 layers in BF16), in
tables of their own (``qw_fp4_*``), so the GLM-5 ones never see them: C-speed at 13 nodes, TP2 x data parallel 52 with CUDA graphs, three
entries of one job (GLM-FP4-03, 1272011), steps 3-20 as above; and C-loss on 1 node, TP2 x PP2 x data parallel 2, 500 steps from seed 1234,
one job per run (GLM-FP4-02, 1272019-22). Keyed on the job: all four loss runs record repetition 1, and 1272020 is BF16's second run by its
entry name only (``-rep2``). Loss: the ``.0.data`` loss events (the last stage's, rank 4). Step times of the loss runs: steps 3-500.
GPU telemetry over the same steps (wall clock, from rank 0's per-rank metrics), all nodes: software power cap share per node, mean power and SM clock over every sample, Xid rows.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pandas as pd
from tt_analysis.runs import data_files, parse_run

from .machine import CLUSTER, SUBNET_OF
from .runs import parse_extras, read_events

JOBS = {"1271210": "GLM-FP4-01"}  # REQUESTS.md; bring-up (1 and 4 nodes) is not read
# Qwen3 32B dense: C-speed at 13 nodes (one job, three entries) and C-loss on 1 node (one job per run); REQUESTS.md
QW_JOBS = {"1272011": "GLM-FP4-03", **{j: "GLM-FP4-02" for j in ("1272019", "1272020", "1272021", "1272022")}}
FIRST_STEP, LAST_STEP = 3, 20
SW_POWER_CAP = 0x4  # nvmlClocksEventReasonSwPowerCap


def _cap_shares(nodemon: Path, t0: float, t1: float) -> dict[str, float]:
    """Per host: % of GPU samples in [t0, t1] with the software power cap among the clock event reasons."""
    out = {}
    for path in sorted(nodemon.glob("*.gpu.tsv")) if nodemon.is_dir() else []:
        g = pd.read_csv(path, sep="\t", usecols=["time_ms", "clocks_event_reasons"])
        g = g[(g.time_ms / 1000 >= t0) & (g.time_ms / 1000 <= t1)]
        if len(g):
            reasons = g.clocks_event_reasons.map(lambda s: int(str(s), 16) if pd.notna(s) else 0)
            out[path.name.split(".")[0]] = 100 * ((reasons & SW_POWER_CAP) != 0).mean()
    return out


def _gpu_window(nodemon: Path, t0: float, t1: float) -> tuple[pd.DataFrame, dict]:
    """GPU samples in [t0, t1], every node: per host (% at the software power cap, samples), and over all samples the mean power, mean and
    lowest SM clock, and rows with an Xid."""
    frames = []
    for path in sorted(nodemon.glob("*.gpu.tsv")) if nodemon.is_dir() else []:
        g = pd.read_csv(path, sep="\t", usecols=lambda c: c in {"time_ms", "clocks_event_reasons", "power_w", "sm_clock_mhz", "xid"})
        g = g[(g.time_ms / 1000 >= t0) & (g.time_ms / 1000 <= t1)]
        if len(g):
            reasons = g.clocks_event_reasons.map(lambda s: int(str(s), 16) if pd.notna(s) else 0)
            frames.append(g.assign(host=path.name.split(".")[0], cap=(reasons & SW_POWER_CAP) != 0))
    if not frames:
        return pd.DataFrame(), {}
    g = pd.concat(frames, ignore_index=True)
    xid = pd.to_numeric(g.xid, errors="coerce").fillna(0) if "xid" in g else pd.Series(0, index=g.index)
    per = g.groupby("host").agg(cap_pct=("cap", "mean"), samples=("cap", "size")).assign(cap_pct=lambda d: 100 * d.cap_pct)
    return per, {"power_w": g.power_w.mean(), "sm_mhz": g.sm_clock_mhz.mean(), "sm_min": g.sm_clock_mhz.min(), "xid_rows": int((xid != 0).sum())}


def _wall_window(run_dir: Path, first: int, last: int) -> tuple[float, float] | None:
    """Wall-clock [end of step first - 1, end of step last] from rank 0's per-rank metrics (its step rows' ``time``; batch_id is the step).
    The rate events' times are their start plus the summed device-timer step times, which fall 1-2 s behind the wall clock by step 20."""
    rank0 = sorted(run_dir.glob("metrics.*.0.jsonl"))
    if len(rank0) != 1:
        return None
    ends = {}
    for line in rank0[0].read_text().splitlines():
        d = json.loads(line)
        if d.get("kind") == "step" and d.get("task") == "train":
            ends[d["batch_id"]] = d["time"]
    return (ends[first - 1], ends[last]) if first - 1 in ends and last in ends else None


def _summary(path: Path) -> tuple[dict, dict]:
    """The summary, and the command line's --flag value pairs (the ``start`` event)."""
    summary, argv = {}, {}
    for e in read_events(path):
        d = e.get("data")
        if e.get("event") == "start" and isinstance(d, dict) and isinstance(d.get("command"), list):
            c = d["command"]
            argv = {k: v for k, v in zip(c, c[1:]) if k.startswith("--") and not v.startswith("--")}
        elif isinstance(d, dict) and isinstance(d.get("summary"), dict):
            summary = d["summary"]
    return summary, argv


def build(root: Path, sacct: Path, unfinished: tuple) -> dict[str, pd.DataFrame]:
    """glm_fp4_runs (one row per entry) and glm_fp4_steps (milabench's step times, every step). Only jobs that have ended."""
    states = {}
    if sacct.exists():
        s = pd.read_csv(sacct, sep="|", dtype=str)
        states = dict(zip(s.JobID, s.State))
    runs, steps = [], []
    qw = {"qw_fp4_runs": [], "qw_fp4_steps": [], "qw_fp4_loss": []}
    for f in data_files(root) if root.is_dir() else []:
        job = f.parent.name.split(".")[1]
        state = states.get(job, "RUNNING")
        if job in QW_JOBS and not state.startswith(unfinished):
            for name, rows in _qwen(f, job, state, root).items():
                qw[name] += rows
            continue
        if job not in JOBS or state.startswith(unfinished):
            continue  # a job still running is not a failed run: left out until it ends
        parsed = parse_run(f, CLUSTER, SUBNET_OF)
        if parsed is None:
            continue
        row, _, _, _ = parsed
        extra, loss, _, batch_ids, elapsed, _ = parse_extras(f, row["run_id"])
        st = pd.DataFrame({"step": [b + 1 for b in batch_ids], "step_s": elapsed}) if len(batch_ids) == len(elapsed) else pd.DataFrame()
        if st.empty:
            continue
        window = st[st.step.between(FIRST_STEP, LAST_STEP)].step_s
        losses = {x["step"]: x["loss"] for x in loss if x.get("step") is not None}
        hosts = row["hosts"].split(";")
        summ, argv = _summary(f)
        detail, split = summ.get("workload", {}).get("precision_detail") or {}, summ.get("parallelism", {}).get("pp_split") or [None]
        caps = _cap_shares(root / "nodemon" / f.parent.name, extra["rates_t0"] + st.step_s[st.step < FIRST_STEP].sum(),
                           extra["rates_t0"] + st.step_s.sum())
        others = [v for h, v in caps.items() if h != hosts[-1]]
        runs.append({"request": JOBS[job], "job": job, "job_state": state.split()[0], "entry": row["bench"], "run_id": row["run_id"],
                     "precision": row["precision"], "started": row["started"], "nodes": row["nodes"], "gpus": row["gpus"], "hosts": row["hosts"],
                     "layout": extra["layout"], "seq_len": row["seq_len"], "seqs": row["global_batch_tokens"] // row["seq_len"],
                     "global_batch_tokens": row["global_batch_tokens"], "n_layers": row["n_layers"], "status": row["status"],
                     "bf16_layers_last": detail.get("bf16_layers_last", 0), "last_stage_layers": split[-1],
                     "lr": float(argv["--lr"]) if "--lr" in argv else None, "lr_schedule": argv.get("--lr-decay-style"),
                     "steps_logged": len(st), "measured": len(window), "step_mean": window.mean(), "step_median": window.median(),
                     "step_sd": window.std(), "step_first": window.iloc[0], "step_last": window.iloc[-1],
                     "step_end3": st[st.step > LAST_STEP - 3].step_s.mean(), "step_start3": st[st.step.between(FIRST_STEP, FIRST_STEP + 2)].step_s.mean(),
                     "tok_s_gpu": row["global_batch_tokens"] / window.mean() / row["gpus"],
                     # The campaign runs table's measure (runs.drop_warmup): median rate over the summary's measured steps, 2-20
                     "tok_s_gpu_measured": row["global_batch_tokens"] / st[st.step > (extra["warmup_steps"] or 1)].step_s.median() / row["gpus"],
                     "mem_alloc_gib": row["mem_alloc_gib"], "loss_last": losses.get(LAST_STEP), "last_host": hosts[-1],
                     "cap_last_pct": caps.get(hosts[-1]), "cap_others_lo": min(others) if others else None,
                     "cap_others_hi": max(others) if others else None})
        steps += [{"run_id": row["run_id"], "precision": row["precision"], "step": s, "step_s": v} for s, v in zip(st.step, st.step_s)]
    return {"glm_fp4_runs": pd.DataFrame(runs), "glm_fp4_steps": pd.DataFrame(steps), **{k: pd.DataFrame(v) for k, v in qw.items()}}


def _qwen(f: Path, job: str, state: str, root: Path) -> dict[str, list]:
    """One Qwen3 32B entry: its run row, its step times (milabench's timer) and, for C-loss, its loss per step. Empty if it has no steps."""
    parsed = parse_run(f, CLUSTER, SUBNET_OF)
    if parsed is None:
        return {}
    row, _, _, _ = parsed
    extra, loss, _, batch_ids, elapsed, _ = parse_extras(f, row["run_id"])
    st = pd.DataFrame({"step": [b + 1 for b in batch_ids], "step_s": elapsed}) if len(batch_ids) == len(elapsed) else pd.DataFrame()
    if st.empty:
        return {}
    last = int(st.step.max())
    window = st[st.step.between(FIRST_STEP, last)].step_s
    summ, argv = _summary(f)
    run, par = summ.get("run", {}), summ.get("parallelism", {})
    detail = summ.get("workload", {}).get("precision_detail") or {}
    label = re.search(r"(bf16|mxfp8|nvfp4)(-rep\d+)?", row["bench"]).group(0)  # the entry's own name: the repeat is told apart by it only
    losses = pd.DataFrame([x for x in loss if x.get("step") is not None])
    at = dict(zip(losses.step, losses.loss)) if len(losses) else {}
    window_t = _wall_window(f.parent, FIRST_STEP, last) or (extra["rates_t0"] + st.step_s[st.step < FIRST_STEP].sum(), extra["rates_t0"] + st.step_s.sum())
    per, pooled = _gpu_window(root / "nodemon" / f.parent.name, *window_t)
    after50 = losses[losses.step > 50].loss if len(losses) else pd.Series(dtype=float)
    out = {"request": QW_JOBS[job], "job": job, "job_state": state.split()[0], "entry": row["bench"], "run": label, "run_id": row["run_id"],
           "experiment": run.get("experiment"), "precision": row["precision"], "repetition": row.get("repetition"), "seed": extra.get("seed"),
           "started": row["started"], "nodes": row["nodes"], "gpus": row["gpus"], "hosts": row["hosts"], "layout": extra["layout"],
           "tp": par.get("tp"), "pp": par.get("pp"), "dp": par.get("dp_replicate"), "seq_len": row["seq_len"], "seqs": row["global_batch_tokens"] // row["seq_len"],
           "global_batch_tokens": row["global_batch_tokens"], "n_layers": row["n_layers"], "status": row["status"],
           "bf16_layers_last": detail.get("bf16_layers_last", 0), "lr": float(argv["--lr"]) if "--lr" in argv else None,
           "lr_schedule": argv.get("--lr-decay-style"), "lr_warmup": int(argv["--lr-warmup-iters"]) if "--lr-warmup-iters" in argv else 0,
           "min_lr": float(argv["--min-lr"]) if "--min-lr" in argv else None,
           "steps_logged": len(st), "first": FIRST_STEP, "last": last, "measured": len(window), "step_mean": window.mean(),
           "step_median": window.median(), "step_sd": window.std(), "tok_s_gpu": row["global_batch_tokens"] / window.mean() / row["gpus"],
           "mem_alloc_gib": row["mem_alloc_gib"], "loss_events": len(losses), "loss_last": at.get(last),
           "loss_nonfinite": int((~pd.Series(list(at.values()), dtype=float).map(math.isfinite)).sum()),
           "loss_max_after50": after50.max() if len(after50) else None,
           "cap_med": per.cap_pct.median() if len(per) else None, "cap_lo": per.cap_pct.min() if len(per) else None,
           "cap_hi": per.cap_pct.max() if len(per) else None, "cap_nodes": len(per), **pooled}
    return {"qw_fp4_runs": [out],
            "qw_fp4_steps": [{"job": job, "run": label, "step": s, "step_s": v} for s, v in zip(st.step, st.step_s)],
            "qw_fp4_loss": ([{"job": job, "run": label, "step": int(x.step), "loss": x.loss, "grad_norm": x.grad_norm} for x in losses.itertuples()]
                            if out["experiment"] == "C-loss" else [])}
