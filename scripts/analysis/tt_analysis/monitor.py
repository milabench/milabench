"""Node telemetry -> ``gpu_monitor`` rows (long format).

Two sources (FORMAT.md section 3):
- benchmate.nodemon run directories (``nodemon/<run_name>/nodes/<host>.<collector>.tsv``),
  one wide TSV per collector, melted here; the directory name is the run name, which is
  what joins the telemetry with the run data;
- DCGM exports (``dcgm/<name>.csv``), already long, with the DCGM field names mapped onto
  the nodemon ones so both sources share one vocabulary.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

SCHEMA = "benchmate.nodemon.v1"

# Per collector, the columns that identify a row's entity; everything else is a metric.
# `gpu_uuid` is no longer one of them -- it is in the status file -- but it is still listed so
# the directories written before that change parse the same way.
ENTITY_COLUMNS = {
    "gpu": ["gpu", "gpu_uuid"],
    "nvlink": ["gpu"],
    "ib": ["hca", "port"],
    "cpu": ["package"],
    "mem": [],
    "pcie": ["device", "driver"],
    "ecc": ["gpu"],
    "edac": ["mc"],
    "dcgm": ["gpu"],
}

# DCGM field -> the nodemon metric it stands for; unlisted fields keep their DCGM name.
DCGM_METRIC = {
    "DCGM_FI_DEV_SM_CLOCK": "sm_clock_mhz",
    "DCGM_FI_DEV_CLOCKS_EVENT_REASONS": "clocks_event_reasons",
    "DCGM_FI_DEV_POWER_USAGE": "power_w",
    "DCGM_FI_DEV_GPU_TEMP": "temp_c",
    "DCGM_FI_DEV_GPU_UTIL": "util",
}
# DCGM reports utilization in percent, nodemon as a fraction.
DCGM_SCALE = {"DCGM_FI_DEV_GPU_UTIL": 0.01}

COLUMNS = ["cluster", "source", "schema", "host", "entity", "gpu", "gpu_uuid", "metric",
           "value", "text", "time", "bench", "truncated", "synthetic"]


def _numeric(series: pd.Series) -> pd.Series:
    """Cell text -> float, reading the hex bitmasks nodemon writes for throttle reasons."""
    text = series.astype(str)
    mask = text.str.startswith("0x")
    out = pd.to_numeric(series.where(~mask), errors="coerce")
    if mask.any():
        out[mask] = text[mask].map(lambda v: float(int(v, 16)))
    return out


def _entity(frame: pd.DataFrame, collector: str) -> pd.Series:
    """The entity a row describes: GPU index, ``hca:port``, CPU package, PCIe device."""
    keys = [c for c in ENTITY_COLUMNS[collector] if c != "gpu_uuid" and c in frame.columns]
    if not keys:
        return pd.Series("", index=frame.index)
    return frame[keys].astype(str).agg(":".join, axis=1)


def _status(nodes: Path, host: str) -> dict:
    """One host's sampler status file, or an empty dict when it is missing."""
    path = nodes / f"{host}.status.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _gpu_uuids(status: dict) -> dict[int, str]:
    """GPU index -> UUID, which the NVML collectors publish once in the status file."""
    found = {}
    for described in status.get("collectors", {}).values():
        for index, uuid in (described.get("gpus") or {}).items():
            found[int(index)] = uuid
    return found


def parse_nodemon(run: Path, cluster: str, synthetic: bool = False) -> pd.DataFrame:
    """One nodemon run directory -> one long row per (time, entity, metric)."""
    nodes = run / "nodes"
    frames = []
    for path in sorted(nodes.glob("*.tsv")):
        host, collector = path.name.split(".")[:2]
        if collector not in ENTITY_COLUMNS:
            continue
        wide = pd.read_csv(path, sep="\t")
        if wide.empty:
            continue
        ids = ["time_ms", *[c for c in ENTITY_COLUMNS[collector] if c in wide.columns]]
        wide["entity"] = _entity(wide, collector)
        long = wide.melt(id_vars=[*ids, "entity"], var_name="metric", value_name="text")
        long["value"] = _numeric(long.text)
        long["time"] = long.time_ms / 1000
        long["host"] = host
        if "gpu" not in long.columns:
            long["gpu"] = pd.NA
        status = _status(nodes, host)
        if "gpu_uuid" not in long.columns:
            long["gpu_uuid"] = long.gpu.map(_gpu_uuids(status))
        long["schema"] = status.get("schema") or SCHEMA
        long["bench"] = status.get("bench")
        long["truncated"] = status.get("state") == "running"
        frames.append(long)
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    out = pd.concat(frames, ignore_index=True)
    # Nullable integer: the node-wide collectors have no GPU, and a float index would print as "0.0".
    out["gpu"] = pd.to_numeric(out.gpu, errors="coerce").astype("Int64")
    out["cluster"] = cluster
    out["source"] = run.name
    out["synthetic"] = synthetic
    return out[COLUMNS]


def parse_dcgm(export: Path, cluster: str, synthetic: bool = False) -> pd.DataFrame:
    """One DCGM export (Prometheus range query, flattened) -> the same long rows."""
    df = pd.read_csv(export)
    if df.empty:
        return pd.DataFrame(columns=COLUMNS)
    df["value"] = df.value * df.metric.map(DCGM_SCALE).fillna(1.0)
    df["metric"] = df.metric.map(lambda m: DCGM_METRIC.get(m, m))
    df["entity"] = df.gpu.astype(str)
    df["gpu"] = df.gpu.astype("Int64")
    df["gpu_uuid"] = df.uuid if "uuid" in df.columns else None
    df["text"] = df.value.astype(str)
    df["bench"] = None
    df["truncated"] = False
    df["cluster"] = cluster
    df["source"] = export.stem
    df["schema"] = "dcgm"
    df["synthetic"] = synthetic
    return df[COLUMNS]
