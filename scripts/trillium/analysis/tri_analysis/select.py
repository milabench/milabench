"""Run selections and derived quantities shared by the figures and the macros (one definition each)."""

from __future__ import annotations

import numpy as np
import pandas as pd

QWEN_BASE = {"grad_reduce": "fp32", "ac": "full", "precision": "bf16"}  # A-Q cross: every other lever fixed


def completed(runs: pd.DataFrame, *experiments) -> pd.DataFrame:
    """Completed runs, of the given experiments (all when none given)."""
    if runs.empty:
        return runs
    r = runs[runs.status == "completed"]
    return r[r.experiment.isin(experiments)] if experiments else r


def qwen_base(runs: pd.DataFrame) -> pd.DataFrame:
    """A-Q runs of the layout x tokens cross: FP32 reduction, full recompute, BF16."""
    r = completed(runs, "A-Q")
    for k, v in QWEN_BASE.items():
        r = r[r[k] == v] if not r.empty else r
    return r


def drift(r: pd.DataFrame) -> tuple[pd.Series, float]:
    """Reference runs (R): hours since the first one, and the linear trend over the window (%)."""
    hours = (pd.to_datetime(r.started) - pd.to_datetime(r.started).min()).dt.total_seconds() / 3600
    if len(r) < 3:
        return hours, float("nan")
    slope, _ = np.polyfit(hours, r.tok_s_gpu, 1)
    return hours, 100 * slope * (hours.max() - hours.min()) / r.tok_s_gpu.median()


def with_energy(runs: pd.DataFrame, power: pd.DataFrame) -> pd.DataFrame:
    """Runs with GPU power -> + mtok_kwh: million tokens per kWh of GPU energy (GPU power only).
    The mean power of the GPUs sampled stands for every GPU of the run."""
    if runs.empty or power.empty:
        return pd.DataFrame()
    e = runs.merge(power.drop(columns="synthetic"), on="run_id")
    return e.assign(mtok_kwh=e.tok_s_gpu / e.power_w_gpu * 3.6e6 / 1e6)
