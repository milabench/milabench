"""Run selections and derived quantities shared by the figures and the macros (one definition each)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .machine import GLM_COPY_NODES, GLM_SEQS_PER_COPY, GLM_STRONG_SEQS, GLM_TOP_COPIES, GLM_TOP_NODES
from .style import GLM, QWEN

# A-Q cross: every other lever fixed, at the defaults of config/torchtitan_native_trillium.yaml
QWEN_BASE = {"grad_reduce": "fp32", "ac": "selective", "precision": "bf16"}

# Which runs enter the report. Every run stays in runs.csv and in the ledger; `completed`, `failed_runs` and
# `checkpoints` see only ``in_report``. The hardware reruns are read directly, in the machine block.
REPORT_CAMPAIGNS = {"phase1", "trillium-1105"}
MACHINE_CAMPAIGNS = {"post-pstate", "hw-reboot", "hw-probe"}
REPORT_EXPERIMENTS = {"A-Q", "A-G", "B-Q-weak", "B-Q-strong", "B-G-weak", "B-G-strong", "C-speed", "C-loss", "E1", "R",
                      "F1", "F2"}
# Tagged for the campaign, but kept out whatever their tags: job -> why. All runs of the job.
SET_ASIDE = {
    "1239932": "capped trig0123 in the job, 26.3 s per step, cancelled: evidence for the cap only",
    "1241330": "cancelled",
    "1241064": "whole measurement beside a PP2 job in its bandwidth group, 4.955 s; rerun on the same nodes as 1241200",
}
# Left out of anything that measures noise between repetitions: rank 2 on trig0101 stalled twice.
NOISE_ASIDE = {"1240555"}
SLOW_NODES = {"trig0134", "trig0135", "trig0148"}  # low CPU clocks, about 8.5 % on 13 nodes (DESIGN.md, E)
CAPPED_NODES = {"trig0122", "trig0123", "trig0133", "trig0138", "trig0140", "trig0143", "trig0145", "trig0152"}  # 200 W
UNFINISHED_JOB_STATES = ("RUNNING", "PENDING", "COMPLETING", "SUSPENDED", "REQUEUED", "CONFIGURING")


def bandwidth_group(host: str) -> int:
    """Two bandwidth groups, measured (DESIGN.md, D2): trig0101-0128 and trig0129-0152."""
    return 1 if host <= "trig0128" else 2


def mark_selection(r: pd.DataFrame) -> pd.DataFrame:
    """Columns that say what a run is for: role (report / machine / test), in_report, set_aside (why not),
    slow_nodes, capped_nodes, bw_groups."""
    measured = ~r.synthetic.astype(bool)
    tagged = r.campaign.isin(REPORT_CAMPAIGNS) & r.experiment.isin(REPORT_EXPERIMENTS)
    tagged |= (r.campaign == "hw-reboot") & (r.experiment == "R")  # the reference run after the reboot belongs to the drift
    role = np.select([tagged, r.campaign.isin(MACHINE_CAMPAIGNS)], ["report", "machine"], "test")
    aside = r.job.map(SET_ASIDE) if "job" in r else pd.Series(None, index=r.index, dtype=object)
    hosts = r.hosts.fillna("").str.split(";")
    return r.assign(role=np.where(measured, role, "report"), set_aside=aside,
                    in_report=(tagged & aside.isna()) | ~measured,
                    slow_nodes=[";".join(sorted(SLOW_NODES & set(h))) for h in hosts],
                    capped_nodes=[";".join(sorted(CAPPED_NODES & set(h))) for h in hosts],
                    bw_groups=["+".join(str(g) for g in sorted({bandwidth_group(x) for x in h if x})) for h in hosts])


def mark_beside(r: pd.DataFrame) -> pd.DataFrame:
    """beside_pp: the Qwen PP runs whose measurement overlapped this one's, in the same bandwidth group, and how much of
    this measurement that covers. A pipeline layout next to another multi-node job in its group ran slower once
    (tp4-fsdp 12k: 4.955 s beside PP2, 3.465 s beside FSDP)."""
    has = r.measure_start.notna() & r.measure_end.notna() & (r.nodes > 1)
    pp = r[has & (r.pp > 1) & (r.model == QWEN)]
    names, shares = {}, {}
    for row in r[has].itertuples():
        seen, covered = [], 0.0
        for o in pp.itertuples():
            overlap = min(row.measure_end, o.measure_end) - max(row.measure_start, o.measure_start)
            same_group = set(row.bw_groups.split("+")) & set(o.bw_groups.split("+"))
            if o.run_id != row.run_id and overlap > 0 and same_group and not set(row.hosts.split(";")) & set(o.hosts.split(";")):
                seen.append(f"{o.layout}:{o.job}")
                covered += overlap
        if seen:
            names[row.run_id] = ";".join(seen)
            shares[row.run_id] = min(1.0, covered / (row.measure_end - row.measure_start))
    return r.assign(beside_pp=r.run_id.map(names), beside_pp_share=r.run_id.map(shares))


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
    # Qwen: the weak-scaling runs other than at 13 nodes are the layout comparison's 2-node and top-scale points.
    extra.append(r[(r.model == QWEN) & (r.experiment == "B-Q-weak") & (r.nodes != 13)].assign(experiment="A-Q"))
    return pd.concat([r, *extra], ignore_index=True)


def completed(runs: pd.DataFrame, *experiments) -> pd.DataFrame:
    """Completed runs of the full model, under the given experiments, counting the runs that serve two
    (with_second_experiments). All completed runs, once each, when none is given."""
    if runs.empty:
        return runs
    r = runs[runs.status == "completed"]
    if "in_report" in r:
        r = r[r.in_report.astype(bool)]
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


def checkpoints(ck: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    """Checkpoint operations of the completed F1 runs, with the run's model, nodes and peak memory."""
    f1 = runs[runs.experiment == "F1"] if len(runs) else runs  # a failed run's finished operations count too
    if "in_report" in f1:
        f1 = f1[f1.in_report.astype(bool)]
    if "full_model" in f1:
        f1 = f1[f1.full_model.astype(object) != False]  # noqa: E712
    if ck.empty or f1.empty:
        return pd.DataFrame()
    out = ck.merge(f1[["run_id", "model", "nodes", "mem_alloc_gib"]], on="run_id")
    if "peak_allocated_gib" not in out:
        out = out.assign(peak_allocated_gib=np.nan)
    # A save without its own peak (Megatron): one run per save, so the run's peak holds it.
    saves = out.op == "save"
    out.loc[saves, "peak_allocated_gib"] = out.peak_allocated_gib[saves].fillna(out.mem_alloc_gib[saves])
    return out


def failed_runs(runs: pd.DataFrame, *experiments) -> pd.DataFrame:
    """Measured runs that failed and whose configuration has no success, under the experiments they serve."""
    if runs.empty or "failed_unresolved" not in runs:
        return runs.iloc[0:0]
    r = runs[runs.failed_unresolved.astype(bool)]
    if "in_report" in r:
        r = r[r.in_report.astype(bool)]
    if not experiments:
        return r  # each run once
    r = with_second_experiments(r)
    return r[r.experiment.isin(experiments)]


def pinned_blocked(saves: pd.DataFrame) -> pd.DataFrame:
    """Pinned async saves, for ``blocked_s``: in a run with a repetition 2, only that one on. Its first
    save also allocates the pinned buffers (``pinned_buffers_allocated``), an upper bound; it stays only
    when the run has nothing else (FORMAT.md)."""
    if saves.empty or "repetition" not in saves:
        return saves
    later = saves.groupby("run_id").repetition.transform("max") >= 2
    return saves[~later | (saves.repetition >= 2)]


def qwen_base(runs: pd.DataFrame) -> pd.DataFrame:
    """A-Q runs of the layout x tokens cross: FP32 reduction, selective recompute, BF16."""
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
