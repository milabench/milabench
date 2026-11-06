"""Raw sources -> tidy tables (CSV) in ``data/tables/``.

Tables: runs, steps, loss, checkpoints, ep_groups, node_health, pairs, collectives,
run_power (GPU power per run, for energy), gpu_monitor (node telemetry of the long F2 runs
only: the other runs keep their per-run power aggregate). Every row carries ``synthetic``, except
steps and ep_groups (through their run).

Synthetic data only fills gaps: a synthetic run is dropped as soon as a measured run has the same
configuration (CONFIG_KEY), with everything attached to it; a synthetic network check is dropped
as soon as the same check (D1 metric, D2, D3 or D4) has measured rows. F1 is replaced per checkpoint
operation (model, nodes, content, op, mode): a save and its load can be separate runs. MFU is
recomputed for every run against the B200 BF16 spec peak (the benchmarks write their own peak, or none).

A failed measured run replaces too: its configuration has no synthetic point left, and the failure is what the
report shows. ``runs.failed_unresolved`` marks failed runs whose configuration has no successful run; table
``failed_ops`` lists F1 operations likewise (run data has no event for them: from the entry's name).

Runs another cluster wrote (``run.cluster``) stop the build unless ``other_clusters=True``: the report
only holds Trillium runs. The directories milabench keeps for install and prepare are not runs.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from tt_analysis.monitor import parse_nodemon
from tt_analysis.runs import data_files, parse_run

from .fabric import parse_collectives, parse_health, parse_pairs
from .machine import CLUSTER, PEAK_BF16, PSTATE_BOOTS, SUBNET_OF
from .runs import canonical_model, drop_warmup, failed_operation, parse_extras
from .select import UNFINISHED_JOB_STATES, mark_beside, mark_selection

TABLES = ["runs", "steps", "loss", "checkpoints", "failed_ops", "ep_groups", "node_health", "pairs", "collectives",
          "run_power", "node_telemetry", "gpu_monitor"]
GPU_COLUMNS = {"time_ms", "gpu", "sm_clock_mhz", "power_w", "util_mean"}
CPU_COLUMNS = {"time_ms", "cpu_mhz_busy_mean"}
FULL_TELEMETRY = {"F2"}  # experiments whose node telemetry is kept row by row
# ... and the metrics kept from it (stability, stragglers).
MONITOR_METRICS = {"sm_clock_mhz", "clocks_event_reasons", "power_w", "util_mean", "tensor_util",
                   "cpu_mhz_busy_mean"}


# Settings that make two runs the same configuration (for replacing synthetic runs by measured ones). The seed
# counts in C-loss only (loss_seed): its repeats differ by seed. F1 goes per checkpoint operation (CKPT_KEY).
CONFIG_KEY = ["experiment", "model", "layout", "nodes", "tokens_per_gpu", "global_batch_tokens", "grad_reduce",
              "ac", "precision", "mb_seqs", "node_set", "condition", "loss_seed"]
CKPT_KEY = ["model", "nodes", "content", "op", "mode"]
RUN_TABLES = ["steps", "loss", "checkpoints", "ep_groups", "run_power", "node_telemetry", "gpu_monitor"]


def _sources(raw: Path, synthetic: Path | None):
    if (raw / CLUSTER).is_dir():
        yield raw / CLUSTER, False
    if synthetic and (synthetic / CLUSTER).is_dir():
        yield synthetic / CLUSTER, True


def _window(frame: pd.DataFrame, run: dict) -> pd.DataFrame:
    if run.get("measure_start") and run.get("measure_end"):
        t = frame.time_ms / 1000
        return frame[(t >= run["measure_start"]) & (t <= run["measure_end"])]
    return frame


def _node_telemetry(d: Path, run: dict, is_syn: bool) -> list[dict]:
    """Per host, over the run's measurement window: GPU power, clock and activity, CPU clock. Reads only these columns:
    the whole telemetry is 5 GB. Only the F2 runs are kept row by row (gpu_monitor)."""
    rows = []
    for path in sorted((d / "nodes").glob("*.gpu.tsv")):
        host = path.name.split(".")[0]
        g = _window(pd.read_csv(path, sep="\t", usecols=lambda c: c in GPU_COLUMNS), run)
        if g.empty:
            continue
        per_sample = g.groupby("time_ms").power_w.sum()
        row = {"run_id": run["run_id"], "host": host, "gpus": g.gpu.nunique(), "gpu_w_mean": g.groupby("gpu").power_w.mean().mean(),
               "node_gpu_kw_mean": per_sample.mean() / 1000, "node_gpu_kw_max": per_sample.max() / 1000,
               "sm_clock_mhz": g.sm_clock_mhz.median(), "synthetic": is_syn}
        if "util_mean" in g:
            row["gpu_util"] = g.util_mean.mean()
        cpu_path = path.with_name(f"{host}.cpu.tsv")
        if cpu_path.exists():
            c = _window(pd.read_csv(cpu_path, sep="\t", usecols=lambda x: x in CPU_COLUMNS), run)
            c = c[c.cpu_mhz_busy_mean.notna()]
            if len(c):
                row.update(cpu_mhz=c.cpu_mhz_busy_mean.mean(), cpu_low_share=(c.cpu_mhz_busy_mean < 500).mean())
        rows.append(row)
    return rows


def _power(nodes: list[dict], run: dict) -> dict:
    """Per run: GPUs sampled and their mean power over the measurement window (W)."""
    n = pd.DataFrame(nodes)
    return {"run_id": run["run_id"], "gpus_sampled": int(n.gpus.sum()), "power_w_gpu": (n.gpu_w_mean * n.gpus).sum() / n.gpus.sum(),
            "power_w_total": (n.gpu_w_mean * n.gpus).sum(), "synthetic": bool(n.synthetic.any())}


def pstate(started, hosts, boots: dict[str, datetime]) -> str | None:
    """Was a run before or after the admins' kernel change, on its nodes? A node not rebooted yet counts as before."""
    if pd.isna(started) or not hosts or pd.isna(hosts):
        return None
    after = [h in boots and datetime.fromisoformat(started) >= boots[h] for h in hosts.split(";")]
    return "after" if all(after) else "mixed" if any(after) else "before"


def mark_jobs(r: pd.DataFrame, sacct: Path) -> pd.DataFrame:
    """job and job_state, from the run name and the collector's sacct snapshot. A run whose job has not ended is not a
    failed run, even if its entry finished: status "running", and left out. So is a job newer than the snapshot."""
    r = r.assign(job=r.run_name.str.extract(r"\.(\d{7})(?:\.|$)")[0])
    if not sacct.exists():
        return r.assign(job_state=None)
    jobs = pd.read_csv(sacct, sep="|", dtype=str)
    state = r.job.map(jobs.set_index("JobID").State)
    unfinished = state.fillna("RUNNING").str.startswith(UNFINISHED_JOB_STATES) & r.job.notna()
    return r.assign(job_state=state, status=r.status.where(~unfinished, "running"))


def build(raw: Path, synthetic: Path | None, out: Path, other_clusters=False) -> dict[str, pd.DataFrame]:
    rows = {name: [] for name in TABLES}
    for cdir, is_syn in _sources(raw, synthetic):
        for hdir in sorted(d for d in (cdir / "health").glob("*") if (d / "report.json").exists()):
            rows["node_health"] += parse_health(hdir, synthetic=is_syn)
        for f in sorted((cdir / "nccl").glob("*/pairs.jsonl")):
            rows["pairs"] += parse_pairs(f, synthetic=is_syn)
        for f in sorted((cdir / "nccl").glob("*/collectives.jsonl")):
            rows["collectives"] += parse_collectives(f, synthetic=is_syn)
        by_name = {}
        for f in data_files(cdir):
            if f.parent.name.startswith(("install.", "prepare.")):
                continue
            parsed = parse_run(f, CLUSTER, SUBNET_OF, synthetic=is_syn)
            if parsed is None:
                continue
            row, steps, _, ep = parsed
            extra, loss, ckpt, batch_ids = parse_extras(f, row["run_id"])
            if not str(extra["run_cluster"] or CLUSTER).startswith(CLUSTER) and not other_clusters:
                raise SystemExit(f"[build] {f}: written on {extra['run_cluster']}, not {CLUSTER}: not report data")
            if len(batch_ids) == len(steps):  # the step index of the rate events, not their position
                for s, b in zip(steps, batch_ids):
                    s["step"] = b
            row.update({k: v for k, v in extra.items() if v is not None or k not in row})
            row["model"] = canonical_model(row.get("model"))
            steps = drop_warmup(row, steps)
            rows["runs"].append(row)
            rows["steps"] += steps
            rows["ep_groups"] += ep
            rows["loss"] += [{**r, "synthetic": is_syn} for r in loss]
            rows["checkpoints"] += [{**r, "synthetic": is_syn} for r in ckpt]
            if row["experiment"] == "F1" and row["status"] != "completed" and not is_syn:
                if (failed := failed_operation(row, ckpt)) is not None:
                    rows["failed_ops"].append(failed)
            by_name.setdefault(row["run_name"], []).append(row)  # a job can run several entries in one directory
        for d in sorted((cdir / "nodemon").glob("*")):
            runs = by_name.get(d.name)
            if runs is None:
                continue
            for run in runs:
                if nodes := _node_telemetry(d, run, is_syn):
                    rows["node_telemetry"] += nodes
                    rows["run_power"].append(_power(nodes, run))
                if run.get("experiment") in FULL_TELEMETRY:
                    mon = parse_nodemon(d, CLUSTER, synthetic=is_syn)
                    rows["gpu_monitor"].append(mon[mon.metric.isin(MONITOR_METRICS)].drop(columns="text"))

    tables = {name: pd.DataFrame(r) for name, r in rows.items() if name != "gpu_monitor"}
    mon = [m for m in rows["gpu_monitor"] if not m.empty]
    tables["gpu_monitor"] = pd.concat(mon, ignore_index=True) if mon else pd.DataFrame()
    r = tables["runs"]
    if not r.empty:
        r["peak_flops_per_gpu"] = PEAK_BF16
        r["mfu_pct"] = 100 * r.flops_per_token * r.tok_s_gpu / PEAK_BF16
        # started is naive local time (tt_analysis.runs), so the boots are too
        boots = {b.node: datetime.fromisoformat(b.boot).astimezone().replace(tzinfo=None)
                 for b in pd.read_csv(PSTATE_BOOTS, sep="\t").itertuples()}
        r["pstate"] = [None if s else pstate(t, h, boots) for s, t, h in zip(r.synthetic, r.started, r.hosts)]
        r = mark_beside(mark_selection(mark_jobs(r, raw / CLUSTER / "sacct.txt")))
        tables["runs"] = r.sort_values("started").reset_index(drop=True)
    _prefer_measured(tables)
    out.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(out / f"{name}.csv", index=False)
    return tables


def _drop_synthetic(df: pd.DataFrame, key) -> pd.DataFrame:
    """Synthetic rows whose key has measured rows -> dropped."""
    if df.empty or not df.synthetic.any() or df.synthetic.all():
        return df
    cols = df[key if isinstance(key, list) else [key]].astype(object)
    k = cols.where(cols.notna(), "").map(str).agg("|".join, axis=1)
    measured = set(k[~df.synthetic])
    return df[~(df.synthetic & k.isin(measured))].reset_index(drop=True)


def _replace_checkpoints(t: dict[str, pd.DataFrame], runs: pd.DataFrame) -> set:
    """F1: a save and its load can be separate runs, or one run can hold several. A synthetic operation
    goes as soon as the same operation is measured, or failed; a synthetic F1 run goes when it has none left.
    ``failed_ops`` keeps the failures of operations that never succeeded."""
    ck, failed = t["checkpoints"], t["failed_ops"]
    if ck.empty:
        return set()
    key = runs[["run_id", "model", "nodes"]]
    merged = ck.merge(key, on="run_id")
    if not failed.empty:
        failed = failed.merge(key, on="run_id").assign(synthetic=False)
        done = {tuple(x) for x in merged[~merged.synthetic.astype(bool)][CKPT_KEY].astype(str).values}
        failed = failed[[tuple(x) not in done for x in failed[CKPT_KEY].astype(str).values]].reset_index(drop=True)
    both = merged.assign(failure=False)
    if not failed.empty:
        both = pd.concat([both, failed.assign(failure=True)], ignore_index=True)
    both = both.assign(synthetic=both.synthetic.astype(bool), failure=both.failure.astype(bool))
    kept = _drop_synthetic(both, CKPT_KEY)
    kept = kept[~kept.failure]
    t["checkpoints"] = kept[ck.columns].reset_index(drop=True)
    t["failed_ops"] = failed
    f1 = runs[(runs.experiment == "F1") & runs.synthetic & runs.run_id.isin(merged.run_id)]
    return set(f1.run_id) - set(kept.run_id)


def _failed_unresolved(runs: pd.DataFrame) -> pd.Series:
    """Measured runs that failed, tagged with an experiment, whose configuration has no measured success."""
    cols = runs[[c for c in CONFIG_KEY if c in runs.columns]].astype(object)
    k = cols.where(cols.notna(), "").map(str).agg("|".join, axis=1)
    measured = ~runs.synthetic.astype(bool)
    done = set(k[measured & (runs.status == "completed")])
    return measured & ~runs.status.isin(["completed", "running"]) & runs.experiment.notna() & ~k.isin(done)


def _prefer_measured(t: dict[str, pd.DataFrame]) -> None:
    runs = t["runs"]
    if not runs.empty:
        keyed = runs.assign(loss_seed=runs.seed.where(runs.experiment == "C-loss"))
        others = keyed[keyed.experiment != "F1"]
        kept = _drop_synthetic(others, [c for c in CONFIG_KEY if c in runs.columns or c == "loss_seed"])
        gone = (set(others.run_id) - set(kept.run_id)) | _replace_checkpoints(t, runs)
        kept = runs[~runs.run_id.isin(gone)].reset_index(drop=True)
        kept["failed_unresolved"] = _failed_unresolved(kept.assign(loss_seed=kept.seed.where(kept.experiment == "C-loss")))
        if gone:
            print(f"[build] {len(gone)} synthetic runs replaced by measured ones")
            for name in RUN_TABLES:
                if not t[name].empty and "run_id" in t[name]:
                    t[name] = t[name][~t[name].run_id.isin(gone)].reset_index(drop=True)
            if not t["gpu_monitor"].empty:
                names = set(runs[runs.run_id.isin(gone)].run_name)
                t["gpu_monitor"] = t["gpu_monitor"][~t["gpu_monitor"].source.isin(names)]
        t["runs"] = kept
    t["node_health"] = _drop_synthetic(t["node_health"], "metric")  # D1, per check
    if not t["pairs"].empty:  # D2: one sweep
        t["pairs"] = t["pairs"].assign(check="D2")
        t["pairs"] = _drop_synthetic(t["pairs"], "check").drop(columns="check")
    t["collectives"] = _drop_synthetic(t["collectives"], "experiment")  # D3, D4


def load(out: Path) -> dict[str, pd.DataFrame]:
    tables = {}
    for name in TABLES:
        try:
            tables[name] = pd.read_csv(out / f"{name}.csv", low_memory=False)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            tables[name] = pd.DataFrame()
    return tables
