"""Kimi K3 pretraining (TorchTitan) at 48 nodes: one row per job, per step and per pipeline stage, from the run directories
under ``data/raw/trillium/k3-trn/runs/`` and the GPU telemetry of the completed runs (``k3-trn/nodemon/<run>/<host>.gpu.tsv``),
both copied by ``analyze.py collect`` from ``$SCRATCH/trillium/new/k3-trn/``.

Step times: milabench's timer (the ``.0.data`` rate events, the measured steps, as the campaign's tables), and rank 0's
``metrics`` file (benchmate rankstore, wall clock) for every step, warm-up included. Memory: the summary's ``per_rank`` peaks,
max over each stage's ranks (one node, 8 ranks, per stage). Telemetry: GPU 0 of each stage's node. Loss: the run's loss events
(the last stage's; the other stages log a placeholder).
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# The 48-node rows (REQUESTS.md K3-TRN-01..09). The 1-, 2- and 4-node entries are bring-up and stand-ins: not read.
SCALE_ENTRIES = ("k3-pp48-full-rbf16", "k3-pp48-l49")
FULL_LAYERS = 93  # moonshotai/Kimi-K3 config.json, text_config.num_hidden_layers
# Request row, and why a run stopped or what sets it apart: RUNS.md (experiment coordinator) and the journal, 2026-11-07 22:04 to
# 2026-11-08 03:00. Text, not measured here; the state comes from sacct.
NOTES = {
    "1262377": ("K3-TRN-01", "no step 1: the pipeline's P2P groups kept a 600 s timeout through the serial first-step warm-up"),
    "1262896": ("K3-TRN-04", "no step 1: last stage out of memory (1F1B kept every micro-batch's loss)"),
    "1263259": ("K3-TRN-06", "cancelled after step 1 by a rule that could not work with a log line every 10 steps; probably training"),
    "1263476": ("K3-TRN-07", "49 of 93 layers, a compromise"),
    "1263757": ("K3-TRN-08", "no step 2: every group's timeout lowered to 180 s after step 1, shorter than step 2's first backward"),
    "1263956": ("K3-TRN-09", "full size"),
}
LOW_POWER_W = 350  # GPU 0 under this: waiting (pipeline bubble, data, communication); computing stages draw 450-900 W
SWEEP_SPLIT = 13  # step 2's sweep is slower from the top stage down to stage 13 than below it (1263956, telemetry)

_STEP_LINE = re.compile(r"\[titan\] (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+ .*step:\s+(\d+)\s")
_CAPACITY = re.compile(r"CUDA capacity: .* with ([\d.]+)GiB memory")
_ACTIVE = re.compile(r"Total parameter count: ([\d,]+), active parameters: ([\d,]+)")


def _events(path: Path):
    with open(path) as f:
        for line in f:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _read_data(path: Path) -> dict:
    """Config, summary, rate and loss events, and the facts the log lines of node 0 give (step lines, capacity, parameters)."""
    out = {"argv": {}, "summary": None, "rates": [], "losses": {}, "step_lines": {}}
    for e in _events(path):
        ev, d = e.get("event"), e.get("data")
        if ev == "config" and isinstance(d, dict):
            out["argv"] = d.get("argv") or {}
        elif ev == "line" and isinstance(d, str):
            if (m := _STEP_LINE.search(d)) and int(m[2]) not in out["step_lines"]:
                out["step_lines"][int(m[2])] = datetime.fromisoformat(m[1])
            elif (m := _CAPACITY.search(d)):
                out["capacity_gib"] = float(m[1])
            elif (m := _ACTIVE.search(d)):
                out["params_total"], out["params_active"] = (int(m[i].replace(",", "")) for i in (1, 2))
        elif ev == "data" and isinstance(d, dict):
            if "rate" in d and d.get("task") == "train":
                out["rates"].append(d)
            elif "loss" in d and "batch_id" in d:
                out["losses"][int(d["batch_id"])] = d["loss"]
            elif "summary" in d:
                out["summary"] = d["summary"]
    return out


def _rank0_steps(run_dir: Path, entry: str) -> dict[int, dict]:
    """Rank 0's step rows (wall clock, every step), by step."""
    files = sorted(run_dir.glob(f"metrics.{entry}.*.0.jsonl"))
    return {int(r["batch_id"]): r for r in _events(files[0]) if r.get("kind") == "step"} if files else {}


def _stages(summary: dict, run_dir: Path, nodemon: Path, step1_end: float | None, step2_end: float | None) -> list[dict]:
    """Per stage: memory peaks (max over its ranks), and from GPU 0's telemetry the first dip after step 1 (step 2's sweep) and
    the share of the measured window under LOW_POWER_W."""
    pr = pd.DataFrame(summary.get("per_rank") or [])
    if pr.empty:
        return []
    per_stage = len(pr) // int(summary["parallelism"]["pp"])
    pr["stage"] = pr["rank"] // per_stage
    m = summary.get("measurement") or {}
    rows = []
    for stage, g in pr.groupby("stage"):
        host = g.host.iloc[0]
        row = {"stage": int(stage), "host": host, "reserved_gib": g.peak_reserved_gib.max(), "allocated_gib": g.peak_alloc_gib.max()}
        tsv = nodemon / f"{host}.gpu.tsv"
        if tsv.exists():
            t = pd.read_csv(tsv, sep="\t", usecols=["time_ms", "gpu", "power_w", "util_mean"])
            t = t[t.gpu == 0].assign(time=lambda x: x.time_ms / 1000)
            w = t[t.time.between(m.get("measurement_start_epoch", np.inf), m.get("measurement_end_epoch", -np.inf))]
            if len(w):
                row["low_power_pct"] = 100 * (w.power_w < LOW_POWER_W).mean()
            if step1_end and step2_end:
                # the stage's one-time pause when step 2's first backward reaches it: GPU 0 nearly idle for some seconds (the
                # 1-2 s transients at step 2's start stay above 0.3, and come within its first 5 s on the stages that pause early)
                dip = t[t.time.between(step1_end + 5, step2_end) & (t.util_mean < 0.25)]
                if len(dip):
                    row["sweep_s"] = dip.time.iloc[0] - step1_end
        rows.append(row)
    return rows


def read_run(run_dir: Path, nodemon: Path, job_start: datetime | None) -> tuple[dict, list[dict], list[dict]]:
    """(run, steps, stages) of one run directory <entry>.<job>."""
    entry, _, job = run_dir.name.rpartition(".")
    d = _read_data(run_dir / f"{entry}.0.data")
    s, argv = d["summary"] or {}, d["argv"]
    w, par = s.get("workload") or {}, s.get("parallelism") or {}
    layer = re.search(r"-l(\d+)", entry)
    request, note = NOTES.get(job, ("", ""))
    run = {"entry": entry, "job": job, "request": request, "note": note,
           "layers": int(w.get("n_layers") or argv.get("--n-layers") or (layer[1] if layer else FULL_LAYERS)),
           "grad_reduce": w.get("grad_reduce_dtype") or argv.get("--grad-reduce") or ("bf16" if "rbf16" in entry else "fp32"),
           "detach_losses": str(argv.get("--pp-detach-losses", "0")) in ("1", "True", "true"),
           "steps_logged": max(d["step_lines"], default=0), "capacity_gib": d.get("capacity_gib"),
           "params_total": d.get("params_total"), "params_active": d.get("params_active")}
    if job_start is not None and 1 in d["step_lines"]:
        run["step1_min"] = (d["step_lines"][1] - job_start).total_seconds() / 60  # node 0's log, cluster time, as sacct
    rank0 = _rank0_steps(run_dir, entry)
    if 2 in rank0:
        run["step2_s"] = rank0[2]["elapsed"]
    steps, stages = [], []
    if s:
        gpus, tokens = int(par["gpus"]), int(w["global_batch_tokens"])
        run.update(nodes=int(par["nodes"]), gpus=gpus, pp=int(par["pp"]), ep=int(par["ep"]), schedule=par.get("pp_schedule"),
                   microbatches=int(par["microbatches_per_pipeline"]), global_tokens=tokens, seq_len=int(w["seq_len"]),
                   flops_per_token=s["mfu_inputs"]["flops_per_token"], peak_flops=s["mfu_inputs"]["peak_flops_per_gpu"],
                   data_wait_s=(s.get("data_loading") or {}).get("median_s_per_step"))
        meas = {int(r["batch_id"]): r["elapsed"] for r in d["rates"]}
        if meas:
            e = np.array(list(meas.values()))
            rate = np.median(tokens / e) / gpus  # milabench: median of the per-step rates
            run.update(steps_measured=len(e), measured_from=min(meas), step_median=np.median(e), step_p90=np.percentile(e, 90),
                       step_max=e.max(), tok_s_gpu=rate, tok_s_gpu_lo=(tokens / e).min() / gpus, tok_s_gpu_hi=(tokens / e).max() / gpus,
                       mfu_pct=100 * run["flops_per_token"] * rate / run["peak_flops"])
            if run["params_active"]:
                run["mfu_6n_pct"] = 100 * 6 * run["params_active"] * rate / run["peak_flops"]
        for step in sorted(set(rank0) | set(meas) | set(d["losses"])):
            st = meas.get(step)
            steps.append({"job": job, "step": step, "measured": st is not None, "step_s": st,
                          "step_s_rank0": rank0.get(step, {}).get("elapsed"), "loss": d["losses"].get(step),
                          "tok_s_gpu": tokens / st / gpus if st else None})
        if d["losses"]:
            run.update(loss_first=d["losses"][min(d["losses"])], loss_last=d["losses"][max(d["losses"])])
        ends = {k: r["time"] for k, r in rank0.items()}
        stages = [{"job": job, **r} for r in _stages(s, run_dir, nodemon / run_dir.name, ends.get(1), ends.get(2))]
    return run, steps, stages


def build(runs_dir: Path, nodemon: Path, sacct: Path, unfinished: tuple) -> dict[str, pd.DataFrame]:
    """kimi_train_runs, kimi_train_steps, kimi_train_stages. Only jobs that have ended (the collector's sacct snapshot)."""
    states = {}
    if sacct.exists():
        s = pd.read_csv(sacct, sep="|", dtype=str)
        states = dict(zip(s.JobID, zip(s.State, s.Start, s.End, s.NNodes)))
    runs, steps, stages = [], [], []
    for d in sorted(runs_dir.glob("*.*")) if runs_dir.is_dir() else []:
        entry, _, job = d.name.rpartition(".")
        if entry not in SCALE_ENTRIES or not job.isdigit():
            continue
        state, start, end, nodes = states.get(job, ("RUNNING", None, None, None))
        if state.startswith(unfinished):
            continue  # a job still running is not a failed run: left out until it ends
        r, st, sg = read_run(d, nodemon, datetime.fromisoformat(start) if start else None)
        r.update(job_state=state.split()[0], start=start, end=end)
        r.setdefault("nodes", int(nodes) if nodes else None)
        runs.append(r)
        steps += st
        stages += sg
    runs = pd.DataFrame(runs)
    if len(runs):
        runs = runs.sort_values("start").reset_index(drop=True)
    return {"kimi_train_runs": runs, "kimi_train_steps": pd.DataFrame(steps), "kimi_train_stages": pd.DataFrame(stages)}
