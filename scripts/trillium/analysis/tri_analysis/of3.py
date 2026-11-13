"""OpenFold3 training (OF3): the first training stage's throughput at 1, 2, 4 and 13 nodes, on the uniform data subset, in tables of their own
(``of3_*``). Step 2 (1273887 on 1 node, 1273885 on 2, 1273884 on 4) and OF3-01 (1274059, 13 nodes), keyed on the job and the variant: the
published configuration (``uniform``), DeepSpeed's Evoformer attention (``uniform,ds``, 1 and 13 nodes) and cuEquivariance's triangle kernels
with TF32 matmuls (``uniform,cueq+tf32``). Read from ``of3/runs/<run>/`` (each rank's ``steps-r<rank>.jsonl`` from the implementers' train.py,
and ``runner.yml``) and ``of3/nodemon/<run>/*.{gpu,ib,cpu}.tsv`` (collect, from ``$SCRATCH/trillium/new/of3/runs/<run>/``). The profile runs,
the step-1 subset's runs (1273594-1273810) and the 1-node bring-ups are not read. Outside ``runs/``: the campaign's tables never see them.

Throughput by wall clock, the implementers' convention (their summary.py): the period from rank 0's end of step 6 to its end of step 40, over
the 34 steps between, which is the mean step period over steps 7-40, the time between steps (waiting for the batch, the EMA update, Lightning's
hooks) included. Batch 1 per GPU, so samples/s per GPU = 1 / period. The median period over steps 7-40 beside it. The share of each rank's time
spent waiting before a step (``gap_s`` over ``gap_s + step_s``, steps 7-40): the mean over ranks and the worst rank. GPU telemetry over
[end of step 6, end of step 40] on every node: mean power, median SM clock, mean utilization, software power cap share, Xid rows; CPU: the busy
cores' clock. InfiniBand bytes sent per node and step on the four compute ports: from the node monitor sample nearest the middle of step 7 to
the one nearest the middle of step 40, over the 33 step-end gradient all-reduces between (the counters are read once a second and the
all-reduce lasts about 20 ms, so a window ending at a step end would cut a burst); against one FP32 gradient buffer (the start event's
parameter count x 4 bytes).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

RUNS = {("1273887", "uniform"): "OF3 step 2", ("1273887", "uniform,cueq+tf32"): "OF3 step 2", ("1273887", "uniform,ds"): "OF3 step 2",
        ("1273885", "uniform"): "OF3 step 2", ("1273885", "uniform,cueq+tf32"): "OF3 step 2",
        ("1273884", "uniform"): "OF3 step 2", ("1273884", "uniform,cueq+tf32"): "OF3 step 2",
        ("1274059", "uniform"): "OF3-01", ("1274059", "uniform,cueq+tf32"): "OF3-01", ("1274059", "uniform,ds"): "OF3-01"}
LABEL = {"uniform": "published", "uniform,ds": "ds", "uniform,cueq+tf32": "cueq+tf32"}
FIRST, LAST = 6, 40  # window: rank 0's end of step 6 to its end of step 40 (summary.py: skip 5)
SW_POWER_CAP = 0x4  # nvmlClocksEventReasonSwPowerCap
IB_PORTS = ("mlx5_0", "mlx5_1", "mlx5_4", "mlx5_9")  # the compute ports (mlx5_2, 3: storage)


def _ranks(run_dir: Path) -> tuple[dict, dict]:
    """Per rank: its start event and its step rows by step."""
    starts, steps = {}, {}
    for path in run_dir.glob("steps-r*.jsonl"):
        rank = int(path.stem.rsplit("-r", 1)[1])
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        starts[rank] = next((r for r in rows if r.get("event") == "start"), {})
        steps[rank] = {r["step"]: r for r in rows if "step" in r}
    return starts, steps


def _yml(run_dir: Path, key: str):
    """The first ``key: value`` in runner.yml (no YAML parser in this venv)."""
    path = run_dir / "runner.yml"
    m = re.search(rf"^\s*{key}:\s*(\S+)\s*$", path.read_text(), re.M) if path.exists() else None
    return m[1] if m else None


def _gpu(nodemon: Path, t0: float, t1: float) -> dict:
    frames = []
    for path in sorted(nodemon.glob("*.gpu.tsv")) if nodemon.is_dir() else []:
        g = pd.read_csv(path, sep="\t", usecols=lambda c: c in {"time_ms", "power_w", "sm_clock_mhz", "util_mean", "clocks_event_reasons", "xid"})
        g = g[(g.time_ms >= 1000 * t0) & (g.time_ms <= 1000 * t1)]
        if len(g):
            frames.append(g.assign(host=path.name.split(".")[0]))
    if not frames:
        return {}
    g = pd.concat(frames, ignore_index=True)
    cap = g.clocks_event_reasons.map(lambda s: (int(str(s), 16) & SW_POWER_CAP) != 0 if pd.notna(s) else False)
    xid = pd.to_numeric(g.xid, errors="coerce").fillna(0) if "xid" in g else pd.Series(0, index=g.index)
    per = g.groupby("host").power_w.mean()
    return {"gpu_nodes": g.host.nunique(), "gpu_samples": len(g), "power_w": g.power_w.mean(), "power_max": g.power_w.max(),
            "power_node_lo": per.min(), "power_node_hi": per.max(), "sm_med": g.sm_clock_mhz.median(), "sm_min": g.sm_clock_mhz.min(),
            "util_pct": 100 * g.util_mean.mean(), "cap_pct": 100 * cap.mean(), "xid_rows": int((xid != 0).sum())}


def _cpu(nodemon: Path, t0: float, t1: float) -> dict:
    frames = []
    for path in sorted(nodemon.glob("*.cpu.tsv")) if nodemon.is_dir() else []:
        c = pd.read_csv(path, sep="\t", usecols=["time_ms", "cpu_mhz_busy_mean"])
        frames.append(c[(c.time_ms >= 1000 * t0) & (c.time_ms <= 1000 * t1)].dropna())
    c = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return {"cpu_busy_med": c.cpu_mhz_busy_mean.median(), "cpu_busy_min": c.cpu_mhz_busy_mean.min()} if len(c) else {}


def _ib(nodemon: Path, ends: dict[int, float]) -> pd.DataFrame:
    """Per host: bytes sent per step on the compute ports, from the sample nearest the middle of step FIRST + 1 to the one nearest the middle of
    step LAST, over the step-end all-reduces of steps FIRST + 1 .. LAST - 1 between them."""
    rows = []
    for path in sorted(nodemon.glob("*.ib.tsv")) if nodemon.is_dir() else []:
        ib = pd.read_csv(path, sep="\t", usecols=["time_ms", "hca", "tx_bytes"]).dropna()
        tx = ib[ib.hca.isin(IB_PORTS)].groupby("time_ms").tx_bytes.agg(["sum", "size"])
        tx = tx[tx["size"] == len(IB_PORTS)]["sum"]
        if tx.empty:
            continue
        mids = [1000 * (ends[s - 1] + ends[s]) / 2 for s in (FIRST + 1, LAST)]
        a, b = (tx.index[abs(tx.index - m).argmin()] for m in mids)
        rows.append({"host": path.name.split(".")[0], "ib_bytes": float(tx[b] - tx[a]) / (LAST - FIRST - 1)})
    return pd.DataFrame(rows)


def build(root: Path, sacct: Path, unfinished: tuple) -> dict[str, pd.DataFrame]:
    """of3_runs (one row per run) and of3_steps (rank 0's step periods, every step). Only jobs that have ended."""
    states = {}
    if sacct.exists():
        s = pd.read_csv(sacct, sep="|", dtype=str)
        states = dict(zip(s.JobID, s.State))
    runs, steps = [], []
    for run_dir in sorted((root / "runs").glob("of3.*")) if (root / "runs").is_dir() else []:
        _, job, variant = run_dir.name.split(".", 2)
        state = states.get(job, "RUNNING")
        if (job, variant) not in RUNS or state.startswith(unfinished):
            continue  # a job still running is not a failed run: left out until it ends
        starts, by_rank = _ranks(run_dir)
        if 0 not in by_rank or not {FIRST, LAST} <= set(by_rank[0]):
            continue
        r0 = by_rank[0]
        ends = {s: r["t"] for s, r in r0.items()}
        period = {s: ends[s] - ends[s - 1] for s in ends if s - 1 in ends}
        use = range(FIRST + 1, LAST + 1)
        wall = (ends[LAST] - ends[FIRST]) / (LAST - FIRST)
        gap = {k: sum(v[s]["gap_s"] for s in use) / sum(v[s]["gap_s"] + v[s]["step_s"] for s in use)
               for k, v in by_rank.items() if all(s in v for s in use)}
        world = starts[0].get("world_size", len(by_rank))
        hosts = sorted({x.get("host") for x in starts.values() if x.get("host")})
        nodemon = root / "nodemon" / run_dir.name
        ib = _ib(nodemon, ends) if len(hosts) > 1 else pd.DataFrame()
        losses = [r.get("loss") for v in by_rank.values() for r in v.values()]
        batch = int(_yml(run_dir, "batch_size") or 1)
        buffer = 4 * starts[0].get("params", 0)
        runs.append({"request": RUNS[job, variant], "job": job, "job_state": state.split()[0], "variant": variant, "run": LABEL[variant],
                     "nodes": len(hosts), "gpus": world, "ranks_logged": len(by_rank), "hosts": ";".join(hosts), "batch_per_gpu": batch,
                     "workers": int(_yml(run_dir, "num_workers") or 0), "precision": _yml(run_dir, "precision"),
                     "crop_tokens": int(_yml(run_dir, "token_budget") or 0), "params": starts[0].get("params"), "buffer_bytes": buffer,
                     "steps_logged": len(r0), "first": FIRST + 1, "last": LAST, "measured": LAST - FIRST,
                     "step_wall": wall, "step_median": pd.Series([period[s] for s in use]).median(), "step1_s": r0[1]["step_s"] if 1 in r0 else None,
                     "sps_gpu": batch / wall, "sps_total": batch * world / wall,
                     "gap_pct": 100 * pd.Series(gap).mean() if gap else None, "gap_worst_pct": 100 * max(gap.values()) if gap else None,
                     "mem_alloc_gib": max(r["alloc_peak_gib"] for v in by_rank.values() for r in v.values()),
                     "mem_reserved_gib": max(r["reserved_peak_gib"] for v in by_rank.values() for r in v.values()),
                     "loss_nonfinite": sum(1 for x in losses if x is None or x != x or abs(x) == float("inf")),
                     **_gpu(nodemon, ends[FIRST], ends[LAST]), **_cpu(nodemon, ends[FIRST], ends[LAST]),
                     "ib_nodes": len(ib), "ib_lo_gb": ib.ib_bytes.min() / 1e9 if len(ib) else None, "ib_hi_gb": ib.ib_bytes.max() / 1e9 if len(ib) else None,
                     "ib_total_gb": ib.ib_bytes.sum() / 1e9 if len(ib) else None,
                     "ib_buffers": ib.ib_bytes.sum() / buffer if len(ib) and buffer else None})
        steps += [{"job": job, "run": LABEL[variant], "nodes": len(hosts), "step": s, "period_s": period.get(s), "step_s": r["step_s"],
                   "gap_s": r.get("gap_s"), "recycles": r.get("recycles")} for s, r in sorted(r0.items())]
    return {"of3_runs": pd.DataFrame(runs), "of3_steps": pd.DataFrame(steps)}
