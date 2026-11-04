"""Run selections and derived quantities shared by the figures and the macros (one definition each)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .machine import GLM_COPY_NODES, GLM_SEQS_PER_COPY, GLM_STRONG_SEQS, GLM_TOP_COPIES, GLM_TOP_NODES
from .style import GLM

QWEN_BASE = {"grad_reduce": "fp32", "ac": "full", "precision": "bf16"}  # A-Q cross: every other lever fixed

# GLM-5 runs that serve two experiments. run.experiment holds one value, the launcher's tag, and these
# runs are not repeated: (layout, nodes, global batch in sequences) -> (tag, experiment also served).
# Full recompute and BF16 only: the -sel and FP8 entries of the same size are other configurations.
SERVES_TWICE = [
    ("pp13-ep8", 13, 256, "A-G", "B-G-weak"),
    ("pp13-ep8", 52, 1024, "A-G", "B-G-weak"),
    ("pp13-ep8", 39, 768, "B-G-weak", "B-G-strong"),
    ("pp13-ep8", 26, 512, "B-G-weak", "A-G"),
]
if GLM_TOP_NODES == 39:  # 47 nodes: the 39-node run is also the largest pp13-ep8 of the layout comparison
    SERVES_TWICE.append(("pp13-ep8", 39, 768, "B-G-weak", "A-G"))


def with_second_experiments(runs: pd.DataFrame) -> pd.DataFrame:
    """Runs under every experiment they serve: a run of SERVES_TWICE also appears as a second row, with
    ``experiment`` = the one it also serves (``tagged`` keeps the launcher's tag). The one place that knows."""
    if runs.empty or "layout" not in runs:
        return runs
    r = runs.assign(tagged=runs.experiment)
    seqs = r.global_batch_tokens / r.seq_len
    base = (r.model == GLM) & (r.ac == "full") & (r.precision == "bf16") & (r.mb_seqs == 1)
    extra = [r[base & (r.layout == layout) & (r.nodes == nodes) & (seqs == batch) & (r.experiment == tag)]
             .assign(experiment=also) for layout, nodes, batch, tag, also in SERVES_TWICE]
    return pd.concat([r, *extra], ignore_index=True)


def completed(runs: pd.DataFrame, *experiments) -> pd.DataFrame:
    """Completed runs of the full model, under the given experiments, counting the runs that serve two
    (with_second_experiments). All completed runs, once each, when none is given."""
    if runs.empty:
        return runs
    r = runs[runs.status == "completed"]
    if "full_model" in r:
        r = r[r.full_model.astype(object) != False]  # noqa: E712  (truncated test models, NaN = not known)
    if not experiments:
        return r
    r = with_second_experiments(r)
    return r[r.experiment.isin(experiments)].reset_index(drop=True)


def glm_series() -> dict[str, dict[int, int]]:
    """Series -> {nodes: global batch in sequences}: what each GLM-5 scaling series must hold."""
    copies = range(1, GLM_TOP_COPIES + 1)
    nodes = {c * GLM_COPY_NODES for c in copies}
    return {"B-G-weak": {c * GLM_COPY_NODES: c * GLM_SEQS_PER_COPY for c in copies},
            "B-G-strong": {n: GLM_STRONG_SEQS for n in nodes},
            "A-G": {n: n // GLM_COPY_NODES * GLM_SEQS_PER_COPY for n in (GLM_COPY_NODES, 26, GLM_TOP_NODES)}}


def series_gaps(runs: pd.DataFrame) -> dict[str, list[int]]:
    """Node counts missing from each GLM-5 series of glm_series (pp13-ep8, full recompute, BF16), once
    the runs that serve two experiments are counted. Empty when every series is complete."""
    r = completed(runs, *glm_series())
    gaps = {}
    for name, want in glm_series().items():
        d = r[r.experiment == name] if len(r) else r
        if len(d):
            d = d[(d.layout == "pp13-ep8") & (d.ac == "full") & (d.precision == "bf16")]
            have = {(int(n), round(s)) for n, s in zip(d.nodes, d.global_batch_tokens / d.seq_len)}
        else:
            have = set()
        missing = sorted(n for n, batch in want.items() if (n, batch) not in have)
        if missing:
            gaps[name] = missing
    return gaps


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
