"""Raw sources -> tidy tables (CSV) in ``data/tables/``.

Tables: runs, steps, loss, checkpoints, ep_groups, node_health, pairs, collectives,
run_power (GPU power per run, for energy), gpu_monitor (node telemetry of the long F2 runs
only: the other runs keep their per-run power aggregate). Every row carries ``synthetic``.

Synthetic data only fills gaps: a synthetic run is dropped as soon as a measured run has the same
configuration (CONFIG_KEY), with everything attached to it; a synthetic network check is dropped
as soon as the same check (D1 metric, D2, D3 or D4) has measured rows. F1 is replaced per checkpoint
operation (model, nodes, content, op, mode): a save and its load can be separate runs. MFU is
recomputed for every run against the B200 BF16 spec peak (the benchmarks write their own peak, or none).

Runs another cluster wrote (``run.cluster``) stop the build unless ``other_clusters=True``: the report
only holds Trillium runs. The directories milabench keeps for install and prepare are not runs.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tt_analysis.monitor import parse_nodemon
from tt_analysis.runs import data_files, parse_run

from .fabric import parse_collectives, parse_health, parse_pairs
from .machine import CLUSTER, PEAK_BF16, SUBNET_OF
from .runs import canonical_model, drop_warmup, parse_extras

TABLES = ["runs", "steps", "loss", "checkpoints", "ep_groups", "node_health", "pairs", "collectives",
          "run_power", "gpu_monitor"]
FULL_TELEMETRY = {"F2"}  # experiments whose node telemetry is kept row by row
# ... and the metrics kept from it (stability, stragglers).
MONITOR_METRICS = {"sm_clock_mhz", "clocks_event_reasons", "power_w", "util_mean", "tensor_util",
                   "cpu_mhz_busy_mean"}


# Settings that make two runs the same configuration (for replacing synthetic runs by measured ones).
CONFIG_KEY = ["experiment", "model", "layout", "nodes", "tokens_per_gpu", "global_batch_tokens", "grad_reduce",
              "ac", "precision", "mb_seqs", "node_set", "condition"]
RUN_TABLES = ["steps", "loss", "checkpoints", "ep_groups", "run_power", "gpu_monitor"]


def _sources(raw: Path, synthetic: Path | None):
    if (raw / CLUSTER).is_dir():
        yield raw / CLUSTER, False
    if synthetic and (synthetic / CLUSTER).is_dir():
        yield synthetic / CLUSTER, True


def _power(mon: pd.DataFrame, run: dict) -> dict | None:
    """Per run: GPUs sampled and their mean power over the measurement window (W)."""
    p = mon[mon.metric == "power_w"]
    if run.get("measure_start") and run.get("measure_end"):
        p = p[(p.time >= run["measure_start"]) & (p.time <= run["measure_end"])]
    if p.empty:
        return None
    per_gpu = p.groupby(["host", "gpu"]).value.mean()
    return {"run_id": run["run_id"], "gpus_sampled": len(per_gpu), "power_w_gpu": per_gpu.mean(),
            "power_w_total": per_gpu.sum(), "synthetic": bool(p.synthetic.any())}


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
            by_name.setdefault(row["run_name"], []).append(row)  # a job can run several entries in one directory
        for d in sorted((cdir / "nodemon").glob("*")):
            runs = by_name.get(d.name)
            if runs is None:
                continue
            mon = parse_nodemon(d, CLUSTER, synthetic=is_syn)
            if mon.empty:
                continue
            for run in runs:
                if (p := _power(mon, run)) is not None:
                    rows["run_power"].append(p)
                if run.get("experiment") in FULL_TELEMETRY:
                    rows["gpu_monitor"].append(mon[mon.metric.isin(MONITOR_METRICS)].drop(columns="text"))

    tables = {name: pd.DataFrame(r) for name, r in rows.items() if name != "gpu_monitor"}
    mon = [m for m in rows["gpu_monitor"] if not m.empty]
    tables["gpu_monitor"] = pd.concat(mon, ignore_index=True) if mon else pd.DataFrame()
    r = tables["runs"]
    if not r.empty:
        r["peak_flops_per_gpu"] = PEAK_BF16
        r["mfu_pct"] = 100 * r.flops_per_token * r.tok_s_gpu / PEAK_BF16
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


def _prefer_measured(t: dict[str, pd.DataFrame]) -> None:
    runs = t["runs"]
    if not runs.empty:
        kept = _drop_synthetic(runs, [c for c in CONFIG_KEY if c in runs.columns])
        gone = set(runs.run_id) - set(kept.run_id)
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
