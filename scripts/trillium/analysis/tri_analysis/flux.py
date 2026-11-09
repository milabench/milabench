"""FLUX.1 training, MLPerf Training v5.1, with the University of Florida's submission code (FLUX-01: job 1276503, 8 nodes, one run to the
quality target), against Florida's 10 published runs at 8 nodes, in tables of their own (``flux_*``). Also read: the 4-node real-data bring-up
(job 1276381, 300 steps, for the step from 4 to 8 nodes) and Florida's 10 runs at 4 nodes. Ours from ``flux/runs/<run>/`` (the MLPerf log
``*_1.log`` and the nvidia-smi telemetry ``gpu-<host>.csv``; collect, from ``$SCRATCH/trillium/new/flux/runs/``); Florida's from
``flux/florida/n<nodes>/result_<i>.txt`` (public, mlcommons/training_results_v5.1, fetched by hand: see its README). Outside ``runs/``: the
campaign's tables never see them.

From each log's ``:::MLLOG`` lines (rank 0 only): time to train = ``run_stop`` - ``run_start`` (MLPerf's score is the olympic mean of 10: drop
the best and the worst); samples and status from ``run_stop``; one block per ``eval_accuracy`` (an evaluation every 262,144 samples);
``tracked_stats.train_step_time``, one value per block; the steady step = its median over blocks 4 to the last; init = ``run_start`` -
``init_start`` (untimed: CUDA graphs, warm-up steps). The ten-step step: the median interval between ``train_loss`` events (every 10 steps) from
step 30, intervals holding an evaluation left out. GPU telemetry (nvidia-smi every 5 s, Trillium's local time) over [run_start, run_stop], per GPU.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from statistics import median
from zoneinfo import ZoneInfo

import pandas as pd

RUNS = {"1276503": "FLUX-01", "1276381": "FLUX step 4"}  # keyed on the job; the synthetic-data bring-ups are not read
FROM_BLOCK = 4  # steady step: blocks 4 to the last (the first three are slower: data workers starting, a slow third block)
TEN_FROM = 30  # ten-step intervals from step 30 on (the 4-node bring-up's convention: data workers starting before)
HOT_W = 950  # share of 5 s samples at or above this draw
LOG_TZ = ZoneInfo("America/Toronto")


def _events(path: Path) -> list[dict]:
    out = []
    for line in path.read_text(errors="replace").splitlines():
        i = line.find(":::MLLOG ")
        if i >= 0:
            out.append(json.loads(line[i + 9:]))
    return out


def _read(path: Path) -> tuple[dict, list[dict]]:
    """One run: its summary row and its blocks (evaluation and the block's step time)."""
    ev = _events(path)
    one = {e["key"]: e for e in reversed(ev)}  # the first event of each key
    stats = lambda k: [e for e in ev if e["key"] == "tracked_stats" and k in e["value"]]
    t0, t1 = one["run_start"]["time_ms"], one["run_stop"]["time_ms"] if "run_stop" in one else None
    acc = [e for e in ev if e["key"] == "eval_accuracy"]
    step = [e["value"]["train_step_time"] for e in stats("train_step_time")]
    val = [e["value"]["validation_time"] for e in stats("validation_time")]
    evals = [e["time_ms"] for e in ev if e["key"] == "eval_start"]
    loss = [(e["metadata"]["step"], e["time_ms"]) for e in stats("train_loss")]
    ten = [(b - a) / 1000 / (s1 - s0) for (s0, a), (s1, b) in zip(loss, loss[1:])
           if s0 >= TEN_FROM and not any(a <= x <= b for x in evals)]
    stop = one["run_stop"]["metadata"] if "run_stop" in one else {}
    row = {"status": stop.get("status"), "samples": stop.get("samples_count"), "blocks": len(acc),
           "ttt_min": (t1 - t0) / 60000 if t1 else None, "init_s": (t0 - one["init_start"]["time_ms"]) / 1000,
           "gbs": one["global_batch_size"]["value"], "lr": one["opt_base_learning_rate"]["value"],
           "warmup_steps": one["opt_learning_rate_warmup_steps"]["value"], "seed": one["seed"]["value"],
           "target": one["target_accuracy"]["value"], "block_samples": one["evaluation_frequency"]["value"],
           "first_acc": acc[0]["value"] if acc else None, "last_acc": acc[-1]["value"] if acc else None,
           "step_first": step[0] if step else None, "steady_s": median(step[FROM_BLOCK - 1:]) if len(step) >= FROM_BLOCK else None,
           "steady_lo": min(step[FROM_BLOCK - 1:]) if len(step) >= FROM_BLOCK else None,
           "steady_hi": max(step[FROM_BLOCK - 1:]) if len(step) >= FROM_BLOCK else None,
           "ten_s": median(ten) if ten else None, "ten_n": len(ten), "val_s": median(val) if val else None,
           "run_start_ms": t0, "run_stop_ms": t1}
    blocks = [{"block": i + 1, "samples": e["metadata"]["samples_count"], "accuracy": e["value"],
               "step_s": step[i] if i < len(step) else None, "val_s": val[i] if i < len(val) else None} for i, e in enumerate(acc)]
    return row, blocks


def _gpu(run_dir: Path, t0_ms: float, t1_ms: float) -> dict:
    """Per GPU over [t0, t1]: mean power, share of samples at or above HOT_W, median SM clock; ranges over the GPUs."""
    frames = []
    for path in sorted(run_dir.glob("gpu-*.csv")):
        g = pd.read_csv(path, skipinitialspace=True)
        g.columns = ["time", "gpu", "power_w", "sm_mhz", "util", "temp", "mem"]
        t = pd.to_datetime(g.time, format="%Y/%m/%d %H:%M:%S.%f").dt.tz_localize(LOG_TZ)
        ms = (t - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(milliseconds=1)
        g = g[(ms >= t0_ms) & (ms <= t1_ms)].assign(host=path.stem.removeprefix("gpu-"))
        g["power_w"] = pd.to_numeric(g.power_w.astype(str).str.removesuffix(" W"), errors="coerce")
        g["sm_mhz"] = pd.to_numeric(g.sm_mhz.astype(str).str.removesuffix(" MHz"), errors="coerce")
        frames.append(g)
    g = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if g.empty:
        return {}
    per = g.groupby(["host", "gpu"]).agg(p=("power_w", "mean"), hot=("power_w", lambda x: 100 * (x >= HOT_W).mean()), sm=("sm_mhz", "median"))
    return {"gpu_nodes": g.host.nunique(), "gpus_seen": len(per), "power_lo": per.p.min(), "power_hi": per.p.max(), "power_w": g.power_w.mean(),
            "hot_lo": per.hot.min(), "hot_hi": per.hot.max(), "sm_lo": per.sm.min(), "sm_hi": per.sm.max(), "sm_min": g.sm_mhz.min()}


def build(root: Path, sacct: Path, unfinished: tuple) -> dict[str, pd.DataFrame]:
    """flux_runs (one row per run, ours and Florida's) and flux_blocks (every evaluation block of every run). Our jobs only once ended."""
    s = pd.read_csv(sacct, sep="|", dtype=str) if sacct.exists() else pd.DataFrame(columns=["JobID", "State", "NNodes"])
    states, nodes = dict(zip(s.JobID, s.State)), dict(zip(s.JobID, s.NNodes))
    runs, blocks = [], []
    for run_dir in sorted((root / "runs").glob("nw-flux-*.*")) if (root / "runs").is_dir() else []:
        job = run_dir.name.rsplit(".", 1)[1]
        state = states.get(job, "RUNNING")
        logs = sorted(run_dir.glob("*_1.log"))
        if job not in RUNS or state.startswith(unfinished) or not logs:
            continue  # a job still running is not a failed run: left out until it ends
        row, b = _read(logs[0])
        n = int(nodes[job])
        runs.append({"source": "ours", "request": RUNS[job], "job": job, "job_state": state.split()[0], "run": run_dir.name, "nodes": n, "gpus": 8 * n,
                     **row, **_gpu(run_dir, row["run_start_ms"], row["run_stop_ms"] or float("inf"))})
        blocks += [{"source": "ours", "run": run_dir.name, "nodes": n, **x} for x in b]
    for path in sorted((root / "florida").glob("n*/result_*.txt")) if (root / "florida").is_dir() else []:
        n = int(path.parent.name.removeprefix("n"))
        row, b = _read(path)
        runs.append({"source": "florida", "request": "Florida", "job": None, "job_state": "COMPLETED", "run": f"n{n}/{path.stem}", "nodes": n,
                     "gpus": 8 * n, **row})
        blocks += [{"source": "florida", "run": f"n{n}/{path.stem}", "nodes": n, **x} for x in b]
    return {"flux_runs": pd.DataFrame(runs), "flux_blocks": pd.DataFrame(blocks)}
