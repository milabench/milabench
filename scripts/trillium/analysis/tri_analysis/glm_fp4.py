"""GLM-5 with NVFP4 (GLM-FP4-01, job 1271210): C-speed at 13 nodes, pp13-ep8, BF16 / MXFP8 / NVFP4 in one job, the campaign's
Megatron entries in the new workloads' own venv, the last 5 layers (the last stage) in BF16 under NVFP4. One row per entry, from its run directory under ``data/raw/trillium/glm-fp4/runs/``
and the GPU telemetry under ``glm-fp4/nodemon/<run>/`` (``analyze.py collect``, from ``$SCRATCH/trillium/new/glm-fp4/``). They stay
out of ``runs/``, so the campaign's tables, counts and ledger never see them (their campaign tag, ``trillium-new-1108``, is not a
campaign one either).

Read with the campaign's own parser (``tt_analysis.runs.parse_run`` and ``runs.parse_extras``): the same summary fields and the
``.0.data`` rate events, milabench's timer, which in this job equals rank 0's ``metrics`` elapsed to a few ms. Step times are over
steps 3-20, as the review's: step 1 is the warm-up, step 2 the slow first measured step (the rate events' ``batch_id`` counts from
0, so step = batch_id + 1). Memory: the summary's peak allocated, max over ranks. Loss: the loss event of step 20 (one run each,
early in training). Software power cap: share of the GPU samples whose ``clocks_event_reasons`` has NVML's SwPowerCap bit (0x4),
over steps 3-20, per node.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from tt_analysis.runs import data_files, parse_run

from .machine import CLUSTER, SUBNET_OF
from .runs import parse_extras, read_events

JOBS = {"1271210": "GLM-FP4-01"}  # REQUESTS.md; bring-up (1 and 4 nodes) and the Qwen3 32B entries are not read
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


def _summary(path: Path) -> dict:
    for e in read_events(path):
        d = e.get("data")
        if isinstance(d, dict) and isinstance(d.get("summary"), dict):
            return d["summary"]
    return {}


def build(root: Path, sacct: Path, unfinished: tuple) -> dict[str, pd.DataFrame]:
    """glm_fp4_runs (one row per entry) and glm_fp4_steps (milabench's step times, every step). Only jobs that have ended."""
    states = {}
    if sacct.exists():
        s = pd.read_csv(sacct, sep="|", dtype=str)
        states = dict(zip(s.JobID, s.State))
    runs, steps = [], []
    for f in data_files(root) if root.is_dir() else []:
        job = f.parent.name.split(".")[1]
        state = states.get(job, "RUNNING")
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
        summ = _summary(f)
        detail, split = summ.get("workload", {}).get("precision_detail") or {}, summ.get("parallelism", {}).get("pp_split") or [None]
        caps = _cap_shares(root / "nodemon" / f.parent.name, extra["rates_t0"] + st.step_s[st.step < FIRST_STEP].sum(),
                           extra["rates_t0"] + st.step_s.sum())
        others = [v for h, v in caps.items() if h != hosts[-1]]
        runs.append({"request": JOBS[job], "job": job, "job_state": state.split()[0], "entry": row["bench"], "run_id": row["run_id"],
                     "precision": row["precision"], "started": row["started"], "nodes": row["nodes"], "gpus": row["gpus"], "hosts": row["hosts"],
                     "layout": extra["layout"], "seq_len": row["seq_len"], "seqs": row["global_batch_tokens"] // row["seq_len"],
                     "global_batch_tokens": row["global_batch_tokens"], "n_layers": row["n_layers"], "status": row["status"],
                     "bf16_layers_last": detail.get("bf16_layers_last", 0), "last_stage_layers": split[-1],
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
    return {"glm_fp4_runs": pd.DataFrame(runs), "glm_fp4_steps": pd.DataFrame(steps)}
