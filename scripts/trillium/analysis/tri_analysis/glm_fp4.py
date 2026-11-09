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
one job per run, 3 runs per recipe (GLM-FP4-02, 1272019-22; GLM-FP4-04, 1272416-31). Keyed on the job: the repetition numbers do not tell the
runs apart (GLM-FP4-02's four all record 1, and 1272020 is BF16's second run by its entry name only, ``-rep2``; GLM-FP4-04's record 2 or 3,
under entry names without it), so ``recipe`` is the precision and ``run`` the entry's own label. Loss: the ``.0.data`` loss events (the
last stage's, rank 4). Step times of the loss runs: steps 3-500.
GPU telemetry over the same steps (wall clock, from rank 0's per-rank metrics), all nodes: software power cap share per node, mean power and SM clock over every sample, Xid rows.

Qwen3 32B at 32k and 128k tokens with context parallelism (QW-CP-01, job 1273505, and QW-CP-02, 1273588; ``build_cp``), in tables of their own
(``qw_cp_*``): QW-CP-01's three entries on GLM-FP4-03's 13 nodes (the CP8 one on 12 of them) and QW-CP-02's one (128k CP4 with selective recompute,
the same nodes), read from ``qw-cp/`` (collect: ``qw-cp/runs/<run>/`` and
``qw-cp/nodemon/<run>/*.{gpu,ib}.tsv``), keyed on the job and the entry, with GLM-FP4-03's BF16 entry (1272011) read again from ``glm-fp4/`` as
their 8k reference, so the four rows share one statistic: means over steps 3-20 on milabench's timer, as above. Tokens/s per GPU at the mean step;
MFU from the summary's FLOPs per token (causal attention counted per packed slot) against 2.25 PFLOP/s. Bytes across nodes per node and step: the
four compute ports' tx counters (``ib.tsv``), each interpolated at rank 0's wall-clock end of step 2 and of the last step that ended before that
node's monitor stopped (19 or 20), over the steps between. Stragglers (``qw_cp_strag``, one row per entry, step and node): how far the node's
slowest rank's forward plus backward (per-rank metrics, ``fwd_ms + bwd_ms``) is behind the median rank's, and the largest drop of any of its GPUs'
memory in use within the step (NVML, the local rank 0's 0.5 s ``gpudata`` samples in the per-rank metrics, and milabench's ``.0.data`` for rank 0's
node: the most any sample falls below the highest one before it), over rank 0's wall-clock step. MXFP8 against BF16 at 128k on 4 nodes (job 1278205,
``CP_MX_JOBS``): QW-CP-01's 128k CP4 entry with full recompute as CP4 x DP4, BF16 then MXFP8, one run each, read the same way and labelled
``n4-128k-cp4-full-<precision>`` so the 13-node entries' labels stay unique; the report keeps them apart.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tt_analysis.runs import data_files, parse_run

from .machine import CLUSTER, PEAK_BF16, SUBNET_OF
from .runs import parse_extras, read_events

JOBS = {"1271210": "GLM-FP4-01"}  # REQUESTS.md; bring-up (1 and 4 nodes) is not read
# Qwen3 32B dense: C-speed at 13 nodes (one job, three entries) and C-loss on 1 node (one job per run); REQUESTS.md
QW_JOBS = {"1272011": "GLM-FP4-03", **{j: "GLM-FP4-02" for j in ("1272019", "1272020", "1272021", "1272022")},
           **{j: "GLM-FP4-04" for j in ("1272416", "1272419", "1272423", "1272429", "1272431")}}
FIRST_STEP, LAST_STEP = 3, 20
SW_POWER_CAP = 0x4  # nvmlClocksEventReasonSwPowerCap
# Qwen3 32B with context parallelism (QW-CP-01; REQUESTS.md), and its 8k reference: GLM-FP4-03's BF16 entry, on the same nodes
CP_JOBS = {"1273505": "QW-CP-01", "1273588": "QW-CP-02", "1278205": "QW-CP-n4-mxfp8"}
CP_MX_JOBS = ("1278205",)  # 128k CP4 x DP4 on 4 nodes, BF16 then MXFP8 (the job's experiment tag QW-CP-n4-mxfp8): not 13-node entries
SLOW_S = 1.0  # a node this far behind the median rank's forward + backward in a step is slow in that step
CP_REF = ("1272011", "megatron-qwen3-32b-n13-bf16")
IB_PORTS = ("mlx5_0", "mlx5_1", "mlx5_4", "mlx5_9")  # the compute ports (mlx5_2, 3: storage)


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
    per = (g.groupby("host").agg(cap_pct=("cap", "mean"), samples=("cap", "size"), power_w=("power_w", "mean"), sm_med=("sm_clock_mhz", "median"))
           .assign(cap_pct=lambda d: 100 * d.cap_pct))
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
    out = {"request": QW_JOBS[job], "job": job, "job_state": state.split()[0], "entry": row["bench"], "run": label, "recipe": label.split("-")[0],
           "run_id": row["run_id"],
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
            "qw_fp4_steps": [{"job": job, "run": label, "recipe": out["recipe"], "step": s, "step_s": v} for s, v in zip(st.step, st.step_s)],
            "qw_fp4_loss": ([{"job": job, "run": label, "recipe": out["recipe"], "step": int(x.step), "loss": x.loss, "grad_norm": x.grad_norm}
                             for x in losses.itertuples()]
                            if out["experiment"] == "C-loss" else [])}


def _step_ends(run_dir: Path) -> dict[int, float]:
    """Rank 0's wall-clock step ends from its per-rank metrics (batch_id is the step)."""
    rank0 = sorted(run_dir.glob("metrics.*.0.jsonl"))
    ends = {}
    for line in rank0[0].read_text().splitlines() if len(rank0) == 1 else []:
        d = json.loads(line)
        if d.get("kind") == "step" and d.get("task") == "train":
            ends[d["batch_id"]] = d["time"]
    return ends


def _ib_window(nodemon: Path, ends: dict[int, float], first: int, last: int) -> pd.DataFrame:
    """Per host: GiB sent per step on each compute port and in all, from step first to the last step <= last that ended before the node's monitor
    stopped (each port's tx counter interpolated at the window's ends), the last step used, and the busiest port's mean rate in GB/s."""
    rows = []
    for path in sorted(nodemon.glob("*.ib.tsv")) if nodemon.is_dir() else []:
        ib = pd.read_csv(path, sep="\t", usecols=["time_ms", "hca", "tx_bytes"]).dropna()
        ib = ib[ib.hca.isin(IB_PORTS)].assign(t=lambda d: d.time_ms / 1000)
        if ib.empty or first - 1 not in ends:
            continue
        stop = ib.groupby("hca").t.max().min()
        done = [b for b, t in ends.items() if first <= b <= last and t <= stop]
        if not done:
            continue
        end = max(done)
        t0, t1, n = ends[first - 1], ends[end], end - first + 1
        sent = {h: float(np.diff(np.interp([t0, t1], x.t, x.tx_bytes.astype(float)))[0]) for h, x in ib.sort_values("t").groupby("hca")}
        rows.append({"host": path.name.split(".")[0], "last": end, **{f"ib_{h}": v / n / 2**30 for h, v in sent.items()},
                     "ib_gib": sum(sent.values()) / n / 2**30, "port_max_gbs": max(sent.values()) / (t1 - t0) / 1e9})
    return pd.DataFrame(rows)


def _stragglers(run_dir: Path, first: int, last: int) -> list[dict]:
    """Per step and node: seconds the node's slowest rank's forward + backward is behind the median rank's, and the largest drop of a GPU's memory in
    use within rank 0's wall-clock step (GiB), from the per-rank metrics' step rows and gpudata samples, and milabench's .data for rank 0's node."""
    steps, samples, host_of = {}, {}, {}
    for path in run_dir.glob("metrics.*.jsonl"):
        host, rank = path.name.rsplit(".", 3)[1], int(path.name.rsplit(".", 2)[1])
        host_of[rank] = host
        for line in path.read_text().splitlines():
            d = json.loads(line)
            if d.get("kind") == "step" and d.get("task") == "train":
                steps.setdefault(d["batch_id"], {})[rank] = (d.get("fwd_ms", 0) + d.get("bwd_ms", 0)) / 1000
            elif "gpudata" in d:
                samples.setdefault(host, []).append((d["time"], {g: v["memory"][0] / 1024 for g, v in d["gpudata"].items()}))
    if 0 not in host_of:
        return []
    for e in (x for f in run_dir.glob("*.data") if ".nolog." not in f.name for x in read_events(f)):
        d = e.get("data")
        if isinstance(d, dict) and "gpudata" in d:
            samples.setdefault(host_of[0], []).append((d["time"], {g: v["memory"][0] / 1024 for g, v in d["gpudata"].items()}))
    ends, rows = _step_ends(run_dir), []
    for step in range(first, last + 1):
        tot = steps.get(step, {})
        if len(tot) != len(host_of) or step - 1 not in ends or step not in ends:
            continue
        med = float(np.median(list(tot.values())))
        for host in sorted(set(host_of.values())):
            w = [m for t, m in sorted(samples.get(host, []), key=lambda x: x[0]) if ends[step - 1] <= t <= ends[step]]
            drop = max((float((pd.Series([m[g] for m in w]).cummax() - pd.Series([m[g] for m in w])).max()) for g in w[0]), default=None) if w else None
            rows.append({"step": step, "host": host, "behind_s": max(v for r, v in tot.items() if host_of[r] == host) - med, "drop_gib": drop})
    return rows


def build_cp(root: Path, ref_root: Path, sacct: Path, unfinished: tuple) -> dict[str, pd.DataFrame]:
    """qw_cp_runs (one row per entry: QW-CP-01's three and the 8k reference) and qw_cp_steps (milabench's step times, every step). Only jobs
    that have ended."""
    states = {}
    if sacct.exists():
        s = pd.read_csv(sacct, sep="|", dtype=str)
        states = dict(zip(s.JobID, s.State))
    files = [f for f in (data_files(root) if root.is_dir() else []) if f.parent.name.split(".")[1] in CP_JOBS]
    files += [f for f in (data_files(ref_root) if ref_root.is_dir() else []) if f.parent.name == f"glm-fp4.{CP_REF[0]}.{CP_REF[1]}"]
    runs, steps, strag = [], [], []
    for f in files:
        job = f.parent.name.split(".")[1]
        state = states.get(job, "RUNNING")
        if state.startswith(unfinished):
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
        summ, argv = _summary(f)
        par, work = summ.get("parallelism", {}), summ.get("workload", {})
        command = next((e["data"]["command"] for e in read_events(f) if e.get("event") == "start" and isinstance(e.get("data"), dict)
                        and isinstance(e["data"].get("command"), list)), [])
        ends = _step_ends(f.parent)
        window_t = (ends[FIRST_STEP - 1], ends[LAST_STEP]) if {FIRST_STEP - 1, LAST_STEP} <= set(ends) else (
            extra["rates_t0"] + st.step_s[st.step < FIRST_STEP].sum(), extra["rates_t0"] + st.step_s.sum())
        nodemon = (root if job in CP_JOBS else ref_root) / "nodemon" / f.parent.name
        per, pooled = _gpu_window(nodemon, *window_t)
        ib = _ib_window(nodemon, ends, FIRST_STEP, LAST_STEP)
        losses = {x["step"]: x["loss"] for x in loss if x.get("step") is not None}
        tok = row["global_batch_tokens"] / window.mean() / row["gpus"]
        label = "8k" if job not in CP_JOBS else f"{row['seq_len'] // 1024}k-cp{par.get('cp')}-{work.get('activation_checkpointing')}"
        if job in CP_MX_JOBS:
            label = f"n{row['nodes']}-{label}-{row['precision']}"
        sg = pd.DataFrame(_stragglers(f.parent, FIRST_STEP, LAST_STEP) if job in CP_JOBS else [])
        slow = sg[sg.behind_s > SLOW_S] if len(sg) else sg
        ever = set(slow.host) if len(slow) else set()
        clean = st[st.step.isin(set(sg.step) - set(slow.step))].step_s if len(sg) else pd.Series(dtype=float)
        strag += [{"job": job, "run": label, **x} for x in sg.to_dict("records")]
        runs.append({"request": CP_JOBS.get(job, "GLM-FP4-03"), "job": job, "job_state": state.split()[0], "entry": row["bench"], "run": label,
                     "run_id": row["run_id"], "experiment": summ.get("run", {}).get("experiment"), "precision": row["precision"],
                     "started": row["started"], "nodes": row["nodes"], "gpus": row["gpus"], "hosts": row["hosts"],
                     "tp": par.get("tp"), "cp": par.get("cp"), "dp": par.get("dp_replicate"), "seq_len": row["seq_len"],
                     "seqs": row["global_batch_tokens"] // row["seq_len"], "global_batch_tokens": row["global_batch_tokens"],
                     "tokens_per_gpu": row["global_batch_tokens"] // row["gpus"], "n_layers": row["n_layers"], "status": row["status"],
                     "recompute": work.get("activation_checkpointing"), "recompute_modules": argv.get("--recompute-modules"),
                     "cuda_graphs": "--cuda-graph-impl" in argv,
                     "doc_masking": "--dataloader-inter-document-masking" in command, "cp_comm": argv.get("--cp-comm-type"),
                     "lr": float(argv["--lr"]) if "--lr" in argv else None, "lr_schedule": argv.get("--lr-decay-style"),
                     "steps_logged": len(st), "first": FIRST_STEP, "measured": len(window), "step_mean": window.mean(),
                     "step_median": window.median(), "step_sd": window.std(), "step_min": window.min(), "step_max": window.max(),
                     "step1_s": st.step_s[st.step == 1].sum(), "tok_s_gpu": tok, "flops_per_token": row["flops_per_token"],
                     "mfu_pct": 100 * row["flops_per_token"] * tok / PEAK_BF16, "mem_alloc_gib": row["mem_alloc_gib"],
                     "mem_res_gib": summ.get("memory_gib", {}).get("peak_reserved", {}).get("max"),
                     "loss_last": losses.get(LAST_STEP),
                     "cap_med": per.cap_pct.median() if len(per) else None, "cap_lo": per.cap_pct.min() if len(per) else None,
                     "cap_hi": per.cap_pct.max() if len(per) else None, "cap_nodes": len(per),
                     "power_lo": per.power_w.min() if len(per) else None, "power_hi": per.power_w.max() if len(per) else None,
                     "sm_med_lo": per.sm_med.min() if len(per) else None, "sm_med_hi": per.sm_med.max() if len(per) else None, **pooled,
                     "ib_nodes": len(ib), "ib_lo": ib.ib_gib.min() if len(ib) else None, "ib_hi": ib.ib_gib.max() if len(ib) else None,
                     "ib_last_lo": ib["last"].min() if len(ib) else None, "ib_last_hi": ib["last"].max() if len(ib) else None,
                     "ib_port_max_gbs": ib.port_max_gbs.max() if len(ib) else None,
                     **{f"ib_{h}_{e}": getattr(ib[f"ib_{h}"], e)() if f"ib_{h}" in ib else None for h in IB_PORTS for e in ("min", "max")},
                     "behind_max": sg.behind_s.max() if len(sg) else None, "slow_steps": slow.step.nunique() if len(sg) else None,
                     "slow_steps_multi": int((slow.groupby("step").size() > 1).sum()) if len(sg) else None,
                     "slow_nodes": ";".join(sorted(ever)) if len(sg) else None, "slow_behind_lo": slow.behind_s.min() if len(slow) else None,
                     "slow_behind_hi": slow.behind_s.max() if len(slow) else None,
                     "drop_slow_lo": sg[sg.host.isin(ever)].groupby("host").drop_gib.max().min() if ever else None,
                     "drop_slow_hi": sg[sg.host.isin(ever)].drop_gib.max() if ever else None,
                     "drop_other_max": sg[~sg.host.isin(ever)].drop_gib.max() if len(sg) else None,
                     "clean_steps": len(clean), "clean_lo": clean.min() if len(clean) else None, "clean_hi": clean.max() if len(clean) else None})
        steps += [{"job": job, "run": label, "step": s, "step_s": v} for s, v in zip(st.step, st.step_s)]
    return {"qw_cp_runs": pd.DataFrame(runs), "qw_cp_steps": pd.DataFrame(steps), "qw_cp_strag": pd.DataFrame(strag)}
