"""Raw sources -> tidy tables (CSV) in ``data/tables/``.

Tables: runs, steps, loss, checkpoints, ep_groups, node_health, pairs, collectives,
run_power (GPU power per run, for energy), gpu_monitor (node telemetry of the long F2 runs
only: the other runs keep their per-run power aggregate). Every row carries ``synthetic``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tt_analysis.monitor import parse_nodemon
from tt_analysis.runs import data_files, parse_run

from .fabric import parse_collectives, parse_health, parse_pairs
from .machine import CLUSTER, SUBNET_OF
from .runs import parse_extras

TABLES = ["runs", "steps", "loss", "checkpoints", "ep_groups", "node_health", "pairs", "collectives",
          "run_power", "gpu_monitor"]
FULL_TELEMETRY = {"F2"}  # experiments whose node telemetry is kept row by row
# ... and the metrics kept from it (stability, stragglers).
MONITOR_METRICS = {"sm_clock_mhz", "clocks_event_reasons", "power_w", "util_mean", "tensor_util",
                   "cpu_mhz_busy_mean"}


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


def build(raw: Path, synthetic: Path | None, out: Path) -> dict[str, pd.DataFrame]:
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
            parsed = parse_run(f, CLUSTER, SUBNET_OF, synthetic=is_syn)
            if parsed is None:
                continue
            row, steps, _, ep = parsed
            extra, loss, ckpt = parse_extras(f, row["run_id"])
            row.update({k: v for k, v in extra.items() if v is not None or k not in row})
            rows["runs"].append(row)
            rows["steps"] += steps
            rows["ep_groups"] += ep
            rows["loss"] += [{**r, "synthetic": is_syn} for r in loss]
            rows["checkpoints"] += [{**r, "synthetic": is_syn} for r in ckpt]
            by_name[row["run_name"]] = row
        for d in sorted((cdir / "nodemon").glob("*")):
            run = by_name.get(d.name)
            if run is None:
                continue
            mon = parse_nodemon(d, CLUSTER, synthetic=is_syn)
            if mon.empty:
                continue
            if (p := _power(mon, run)) is not None:
                rows["run_power"].append(p)
            if run.get("experiment") in FULL_TELEMETRY:
                rows["gpu_monitor"].append(mon[mon.metric.isin(MONITOR_METRICS)].drop(columns="text"))

    tables = {name: pd.DataFrame(r) for name, r in rows.items() if name != "gpu_monitor"}
    mon = [m for m in rows["gpu_monitor"] if not m.empty]
    tables["gpu_monitor"] = pd.concat(mon, ignore_index=True) if mon else pd.DataFrame()
    if not tables["runs"].empty:
        tables["runs"] = tables["runs"].sort_values("started").reset_index(drop=True)
    out.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(out / f"{name}.csv", index=False)
    return tables


def load(out: Path) -> dict[str, pd.DataFrame]:
    tables = {}
    for name in TABLES:
        try:
            tables[name] = pd.read_csv(out / f"{name}.csv", low_memory=False)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            tables[name] = pd.DataFrame()
    return tables
