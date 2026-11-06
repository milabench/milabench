"""Node telemetry -> ``gpu_monitor`` rows (long format).

benchmate.nodemon run directories (``nodemon/<run_name>/nodes/<host>.<collector>.tsv``, FORMAT.md
section 3): one wide TSV per collector, melted here; the directory name is the run name, which is
what joins the telemetry with the run data.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

SCHEMA = "benchmate.nodemon.v3"

# NVML clock-event bits. Test with a mask: 0x84 is both, and it is not equal to either.
# Same values as benchmate.nodemon.collectors.gpu.ClockReason.
_SW_POWER_CAP = 0x4
_HW_POWER_BRAKE = 0x80
_REASON_BITS = (("sw_power_cap", _SW_POWER_CAP), ("hw_power_brake", _HW_POWER_BRAKE))

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
    "gpm": ["gpu"],
}

COLUMNS = ["cluster", "source", "schema", "host", "entity", "gpu", "gpu_uuid", "metric",
           "value", "text", "time", "bench", "truncated", "synthetic"]


def reason_bit(text: object, bit: int) -> float | None:
    """1.0 when `bit` is set in a hex mask, 0.0 when it is not, None when the cell is empty.

    ``0x84`` has both the software power cap (0x4) and the external power brake (0x80).
    Comparing the mask for equality with 4 drops the brake.
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return None
    raw = str(text).strip()
    if not raw or raw == "nan":
        return None
    try:
        return float(bool(int(raw, 16) & bit))
    except ValueError:
        return None


def _expand_clock_reasons(frame: pd.DataFrame) -> pd.DataFrame:
    """One 0/1 row per power-cap bit, beside the raw ``clocks_event_reasons`` mask."""
    reasons = frame[frame.metric == "clocks_event_reasons"]
    if reasons.empty:
        return frame
    extra = []
    for name, bit in _REASON_BITS:
        rows = reasons.copy()
        rows["metric"] = name
        rows["value"] = rows["text"].map(lambda text, bit=bit: reason_bit(text, bit))
        rows["text"] = rows["value"].map(lambda value: "" if value is None or pd.isna(value) else str(int(value)))
        extra.append(rows)
    return pd.concat([frame, *extra], ignore_index=True)


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
    return frame[keys].fillna("").astype(str).agg(lambda row: ":".join(v for v in row if v), axis=1)


def _status(run: Path, host: str) -> dict:
    """One host's sampler status file, which sits beside ``nodes/``; empty when it is missing."""
    path = run / f"{host}.status.json"
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
        status = _status(run, host)
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
    return _expand_clock_reasons(out[COLUMNS])
