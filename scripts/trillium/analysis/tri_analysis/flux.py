"""FLUX.1 training, MLPerf Training v5.1, with the University of Florida's submission code (FLUX-01: job 1276503, 8 nodes; FLUX-02: job 1278821,
16 nodes; FLUX-03: job 1279253, 4 nodes; one run each to the quality target), against Florida's 10 published runs at the same node count, in
tables of their own (``flux_*``). Also read: the 4-node real-data bring-up (job 1276381, 300 steps, for the step from 4 to 8 nodes) and
Florida's 10 runs at 4 nodes. Ours from ``flux/runs/<run>/`` (the MLPerf log ``*_1.log`` and the nvidia-smi telemetry ``gpu-<host>.csv``;
collect, from ``$SCRATCH/trillium/new/flux/runs/``); Florida's from ``flux/florida/n<nodes>/result_<i>.txt`` (public,
mlcommons/training_results_v5.1, fetched by hand: see its README). Outside ``runs/``: the campaign's tables never see them.

From each log's ``:::MLLOG`` lines (rank 0 only): time to train = ``run_stop`` - ``run_start`` (MLPerf's score is the olympic mean of 10: drop
the best and the worst); samples and status from ``run_stop``; one block per ``eval_accuracy`` (an evaluation every 262,144 samples);
``tracked_stats.train_step_time``, one value per block; the steady step = its median over blocks 4 to the last; init = ``run_start`` -
``init_start`` (untimed: CUDA graphs, warm-up steps). The ten-step step: the median interval between ``train_loss`` events (every 10 steps) from
step 30, intervals holding an evaluation left out; over steps 30-200 (``ten_gap_s``), also the bootstrap 95 % range of its median. GPU
telemetry (nvidia-smi every 5 s, Trillium's local time) over [run_start, run_stop], per GPU and per node (pooled over its GPUs).

The step-time gap against Florida at 2 nodes (``flux_gap``, ``flux_gap_nodes``): job 1278802's five 200-step runs, one after the other on the
same two nodes (``flux/dev/gap.1278802/<run>/``), and the earlier job 1277931 (``flux/dev/gap1/``, 150 steps, Nsight Systems on rank 0), against
Florida's 10 runs at 2 nodes on the same window: the ten-step step from step 30 to step 200 (``ten_gap_s``), with a bootstrap 95 % range of
its median; nvidia-smi every 1 s with the clock event reasons, per node, between the step-30 and the last ``train_loss`` event.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from statistics import median
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

RUNS = {"1276503": "FLUX-01", "1278821": "FLUX-02", "1279253": "FLUX-03",
        "1276381": "FLUX step 4"}  # keyed on the job; the synthetic-data bring-ups are not read
FROM_BLOCK = 4  # steady step: blocks 4 to the last (the first three are slower: data workers starting, a slow third block)
TEN_FROM = 30  # ten-step intervals from step 30 on (the 4-node bring-up's convention: data workers starting before)
HOT_W = 950  # share of 5 s samples at or above this draw
LOG_TZ = ZoneInfo("America/Toronto")
GAP_TO = 200  # the gap runs' 200 steps: ten-step intervals from step 30 to step 200, Florida's 2-node runs on the same window
GAP_RUNS = (("1278802", "gap.1278802/base", "base"), ("1278802", "gap.1278802/proto-simple", "NCCL_PROTO=Simple"),
            ("1278802", "gap.1278802/pxn-off", "NCCL_PXN_DISABLE=1"), ("1278802", "gap.1278802/cpu-default", "CPU_SHARE=0"),
            ("1278802", "gap.1278802/base2", "base again"), ("1277931", "gap1", "earlier job"))  # (job, directory under flux/dev, run), in order


def _events(path: Path) -> list[dict]:
    out = []
    for line in path.read_text(errors="replace").splitlines():
        i = line.find(":::MLLOG ")
        if i >= 0:
            out.append(json.loads(line[i + 9:]))
    return out


def _ten(ev: list[dict], to: int | None = None) -> list[float]:
    """Seconds per step between consecutive ``train_loss`` events (every 10 steps), from step TEN_FROM (to step ``to``), evaluations left out."""
    evals = [e["time_ms"] for e in ev if e["key"] == "eval_start"]
    loss = [(e["metadata"]["step"], e["time_ms"]) for e in ev if e["key"] == "tracked_stats" and "train_loss" in e["value"]]
    return [(b - a) / 1000 / (s1 - s0) for (s0, a), (s1, b) in zip(loss, loss[1:])
            if s0 >= TEN_FROM and (to is None or s1 <= to) and not any(a <= x <= b for x in evals)]


def _read(path: Path) -> tuple[dict, list[dict]]:
    """One run: its summary row and its blocks (evaluation and the block's step time)."""
    ev = _events(path)
    one = {e["key"]: e for e in reversed(ev)}  # the first event of each key
    stats = lambda k: [e for e in ev if e["key"] == "tracked_stats" and k in e["value"]]
    t0, t1 = one["run_start"]["time_ms"], one["run_stop"]["time_ms"] if "run_stop" in one else None
    acc = [e for e in ev if e["key"] == "eval_accuracy"]
    step = [e["value"]["train_step_time"] for e in stats("train_step_time")]
    val = [e["value"]["validation_time"] for e in stats("validation_time")]
    ten, ten_gap = _ten(ev), _ten(ev, GAP_TO)
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
           "ten_gap_s": median(ten_gap) if ten_gap else None, "ten_gap_n": len(ten_gap),
           **dict(zip(("ten_gap_lo", "ten_gap_hi"), _boot_median(ten_gap) if ten_gap else (None, None))),  # bootstrap 95 % range of that median
           "run_start_ms": t0, "run_stop_ms": t1}
    blocks = [{"block": i + 1, "samples": e["metadata"]["samples_count"], "accuracy": e["value"],
               "step_s": step[i] if i < len(step) else None, "val_s": val[i] if i < len(val) else None} for i, e in enumerate(acc)]
    return row, blocks


def _gpu(run_dir: Path, t0_ms: float, t1_ms: float) -> dict:
    """Per GPU over [t0, t1]: mean power, share of samples at or above HOT_W, median SM clock; ranges over the GPUs, the GPU with the lowest
    median clock; the same per node (``node_*``, pooled over the node's GPUs)."""
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
    agg = dict(p=("power_w", "mean"), hot=("power_w", lambda x: 100 * (x >= HOT_W).mean()), sm=("sm_mhz", "median"))
    per, node = g.groupby(["host", "gpu"]).agg(**agg), g.groupby("host").agg(**agg)
    return {"gpu_nodes": g.host.nunique(), "gpus_seen": len(per), "power_lo": per.p.min(), "power_hi": per.p.max(), "power_w": g.power_w.mean(),
            "hot_lo": per.hot.min(), "hot_hi": per.hot.max(), "sm_lo": per.sm.min(), "sm_hi": per.sm.max(), "sm_min": g.sm_mhz.min(),
            "sm_lo_host": per.sm.idxmin()[0], "sm_lo_gpu": int(per.sm.idxmin()[1]), "node_power_lo": node.p.min(), "node_power_hi": node.p.max(),
            "node_hot_lo": node.hot.min(), "node_hot_hi": node.hot.max(), "node_sm_lo": node.sm.min(), "node_sm_hi": node.sm.max()}


def _boot_median(x: list[float], n: int = 10000) -> tuple[float, float]:
    """Percentile bootstrap 95 % range of the median (fixed seed)."""
    a = np.asarray(x)
    m = np.median(np.random.default_rng(0).choice(a, size=(n, len(a))), axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def _gpu_reasons(run_dir: Path, t0_ms: float, t1_ms: float) -> list[dict]:
    """nvidia-smi at 1 s with the clock event reasons, per node over [t0, t1]: mean power per GPU, share of samples with the software power cap
    active, median SM clock (pooled over the node's GPUs, and the range over its GPUs), samples with any other slowdown reason active."""
    out = []
    for path in sorted(run_dir.glob("gpu-*.csv")):
        g = pd.read_csv(path, skipinitialspace=True)
        g.columns = [c.strip() for c in g.columns]
        t = pd.to_datetime(g.timestamp, format="%Y/%m/%d %H:%M:%S.%f").dt.tz_localize(LOG_TZ)
        ms = (t - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(milliseconds=1)
        g = g[(ms >= t0_ms) & (ms <= t1_ms)]
        power = pd.to_numeric(g["power.draw [W]"].astype(str).str.removesuffix(" W"), errors="coerce")
        sm = pd.to_numeric(g["clocks.current.sm [MHz]"].astype(str).str.removesuffix(" MHz"), errors="coerce")
        cap = g["clocks_event_reasons.sw_power_cap"].str.strip() == "Active"
        other = g[[c for c in g.columns if c.startswith("clocks_event_reasons.") and c.endswith(("slowdown", "power_brake_slowdown"))]]
        per = pd.DataFrame({"gpu": g["index"], "p": power, "sm": sm, "cap": cap}).groupby("gpu").agg(p=("p", "mean"), sm=("sm", "median"), cap=("cap", "mean"))
        out.append({"host": path.stem.removeprefix("gpu-"), "gpus": len(per), "samples": len(g), "power_w": power.mean(), "power_lo": per.p.min(),
                    "power_hi": per.p.max(), "cap_pct": 100 * cap.mean(), "cap_lo": 100 * per.cap.min(), "cap_hi": 100 * per.cap.max(),
                    "sm_mhz": sm.median(), "sm_lo": per.sm.min(), "sm_hi": per.sm.max(), "other_slowdown": int((other.apply(lambda c: c.str.strip()) == "Active").any(axis=1).sum())})
    return out


def _gap(dev: Path, states: dict, unfinished: tuple) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The step-time gap runs at 2 nodes: one row per run (ten-step step from step 30 to the last loss event, its bootstrap range) and one per node."""
    runs, nodes = [], []
    for job, sub, label in GAP_RUNS:
        state = states.get(job, "RUNNING")
        logs = sorted((dev / sub).glob("*_1.log"))
        if state.startswith(unfinished) or not logs:
            continue
        ev = _events(logs[0])
        loss = [(e["metadata"]["step"], e["time_ms"]) for e in ev if e["key"] == "tracked_stats" and "train_loss" in e["value"]]
        ten = _ten(ev, GAP_TO)
        lo, hi = _boot_median(ten) if ten else (None, None)
        t0 = next(t for s, t in loss if s == TEN_FROM)
        runs.append({"job": job, "job_state": state.split()[0], "run": sub, "label": label, "seed": next(e["value"] for e in ev if e["key"] == "seed"),
                     "from_step": TEN_FROM, "to_step": max(s for s, _ in loss), "ten_s": median(ten), "ten_n": len(ten), "boot_lo": lo, "boot_hi": hi})
        nodes += [{"job": job, "run": sub, "label": label, **x} for x in _gpu_reasons(dev / sub, t0, loss[-1][1])]
    return pd.DataFrame(runs), pd.DataFrame(nodes)


def gap_limits(dev: Path) -> pd.DataFrame:
    """Job 1278802's power and clock limits, one row per GPU (nvidia-smi at the job's start; the two nodes' lines interleaved, no host)."""
    path = dev / "gap.1278802" / "gpu-limits.csv"
    if not path.exists():
        return pd.DataFrame()
    g = pd.read_csv(path, skipinitialspace=True)
    g.columns = [c.strip() for c in g.columns]
    g = g[g["index"].astype(str).str.strip() != "index"]  # the second node's header line
    num = lambda c, u: pd.to_numeric(g[c].astype(str).str.removesuffix(u), errors="coerce")
    return pd.DataFrame({"enforced_w": num("enforced.power.limit [W]", " W"), "default_w": num("power.default_limit [W]", " W"),
                         "max_w": num("power.max_limit [W]", " W"), "max_sm_mhz": num("clocks.max.sm [MHz]", " MHz")})


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
    gap, gap_nodes = _gap(root / "dev", states, unfinished)
    return {"flux_runs": pd.DataFrame(runs), "flux_blocks": pd.DataFrame(blocks), "flux_gap": gap, "flux_gap_nodes": gap_nodes,
            "flux_gap_limits": gap_limits(root / "dev")}
