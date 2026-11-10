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
REPORT_CAMPAIGNS = r"^(phase1|trillium-\d{4})$"  # GLM-5's launcher tags trillium-MMDD, by the day it was submitted; trillium-p0 and -p3-ckfix are tests
MACHINE_CAMPAIGNS = {"post-pstate", "hw-reboot", "hw-probe"}
REPORT_EXPERIMENTS = {"A-Q", "A-G", "B-Q-weak", "B-Q-strong", "B-G-weak", "B-G-strong", "C-speed", "C-loss", "E1", "R",
                      "F1", "F2"}
# Tagged for the campaign, but kept out whatever their tags: job -> why. All runs of the job.
SET_ASIDE = {
    "1239932": "capped trig0123 in the job, 26.3 s per step, cancelled: evidence for the cap only",
    "1241330": "cancelled",
    "1277484": "Qwen long run at 51 nodes, cancelled by the site at 16:26 on Nov 9 after 41 min (other users waiting): not a long run",
    "1277485": "GLM-5 long run at 50 nodes, cancelled by the site at 16:26 on Nov 9 before it started",
    "1278797": "GLM-5 C-loss mxfp8 repeat, trig0118 lost GPU 2 (Xid 79) at 21:56 on Nov 9 after step 170 of 500, cancelled: not a result (HW-33)",
    "1251904": "trig0114 stopped responding at 17:03 on Nov 6, after step 30 of 500, NODE_FAIL: says nothing about the recipe",
    "1241064": "whole measurement beside a PP2 job in its bandwidth group, 4.955 s; rerun on the same nodes as 1241200",
    "1268278": "R on the fixed nodes after trig0150 GPU 2 lost all NVLinks (HW-30, between Nov 8 12:03 and 14:56), 6.13 s per "
               "step against 2.89-2.91 s: not a drift point; R resumes with 1277886, after the node's reboot (Nov 9)",
}
# 1265619 (1 node, repetition 3) was cancelled after the site drained trig0102 (08:25:48), 43 s after full sync's last
# operation: its two completed entries, model sync and full sync, are in; full async wrote no event (no run data), and
# 1265822 ran it and full pinned.
# Stopped before they wrote any event, so their run data is empty; they still answer something, or belong in the ledger. Job -> what the launcher ran.
NO_DATA_RUNS = {
    "1247022": {"campaign": "phase1", "experiment": "B-Q-strong", "layout": "hsdp8", "model": QWEN, "bench": "tri-bqs-hsdp8-n26", "tokens_per_gpu": 49152,
                "ac": "selective", "grad_reduce": "fp32", "precision": "bf16"},
    "1239932": {"campaign": "phase1", "experiment": "E1", "layout": "fsdp", "model": QWEN, "bench": "tri-aq-fsdp-12k", "tokens_per_gpu": 12288,
                "ac": "selective", "grad_reduce": "fp32", "precision": "bf16"},
}
# Left out of anything that measures noise between repetitions: rank 2 on trig0101 stalled twice.
NOISE_ASIDE = {"1240555"}
SLOW_NODES = {"trig0134", "trig0135", "trig0148"}  # low CPU clocks, about 8.5 % on 13 nodes (DESIGN.md, E)
SLOW_FIXED = "2026-11-06T13:49"  # the nodes' reboot with the administrators' power supply change: later runs on them are clean
CAPPED_FROM, CAPPED_UNTIL = "2026-11-04T23:55", "2026-11-06T13:00"  # last normal reading before the glitch; all back by then
CAPPED_NODES = {"trig0122", "trig0123", "trig0133", "trig0138", "trig0140", "trig0143", "trig0145", "trig0152"}  # 200 W
UNFINISHED_JOB_STATES = ("RUNNING", "PENDING", "COMPLETING", "SUSPENDED", "REQUEUED", "CONFIGURING")


def bandwidth_group(host: str) -> int:
    """Two bandwidth groups, measured (DESIGN.md, D2): trig0101-0128 and trig0129-0152."""
    return 1 if host <= "trig0128" else 2


def mark_selection(r: pd.DataFrame) -> pd.DataFrame:
    """Columns that say what a run is for: role (report / machine / test), in_report, set_aside (why not),
    slow_nodes, capped_nodes, bw_groups."""
    measured = ~r.synthetic.astype(bool)
    tagged = r.campaign.fillna("").str.match(REPORT_CAMPAIGNS) & r.experiment.isin(REPORT_EXPERIMENTS)
    tagged |= (r.campaign == "hw-reboot") & (r.experiment == "R")  # the reference run after the reboot belongs to the drift
    role = np.select([tagged, r.campaign.isin(MACHINE_CAMPAIGNS)], ["report", "machine"], "test")
    aside = r.job.map(SET_ASIDE) if "job" in r else pd.Series(None, index=r.index, dtype=object)
    hosts = r.hosts.fillna("").str.split(";")
    started = r.started.fillna("").astype(str)
    return r.assign(role=np.where(measured, role, "report"), set_aside=aside,
                    in_report=(tagged & aside.isna()) | ~measured,
                    slow_nodes=[";".join(sorted(SLOW_NODES & set(h))) if s < SLOW_FIXED else "" for h, s in zip(hosts, started)],
                    capped_nodes=[";".join(sorted(CAPPED_NODES & set(h))) if CAPPED_FROM <= s < CAPPED_UNTIL else "" for h, s in zip(hosts, started)],
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
    ("pp13-ep8", 39, 768, "B-G-weak", "A-G"),  # the largest pp13-ep8 of the layout comparison while fewer than 52 nodes are usable
]


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
    # Not a run with slow nodes on purpose (tri-aq-fsdp-24k at 43 nodes, labelled n43-slow3).
    clean = r.slow_nodes.fillna("") == "" if "slow_nodes" in r else True
    extra.append(r[(r.model == QWEN) & (r.experiment == "B-Q-weak") & (r.nodes != 13) & clean].assign(experiment="A-Q"))
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


def loss_gaps(runs: pd.DataFrame, loss: pd.DataFrame) -> pd.DataFrame:
    """C-loss: every comparison of two runs, on their own step index. ``recipe``: FP8 recipe minus BF16 at the same seed, on the same nodes when
    a BF16 run has them. ``same seed``: two runs of a recipe. ``other seed``: first runs of a recipe at two seeds. Gap = a - b, over steps 401-500, at step 500, and over steps 51-100 (early)."""
    r = completed(runs, "C-loss")
    if r.empty or loss.empty:
        return pd.DataFrame()
    r = r.assign(rep=r.repetition.fillna(1)).sort_values(["rep", "started"])
    curves = {k: g.dropna(subset=["step"]).groupby("step").loss.mean() for k, g in loss[loss.run_id.isin(r.run_id)].groupby("run_id")}
    rows = []

    def add(kind, a, b):
        if a.run_id in curves and b.run_id in curves:
            d = (curves[a.run_id] - curves[b.run_id]).dropna()
            rows.append({"kind": kind, "model": a.model, "recipe": a.precision, "seed_a": int(a.seed), "seed_b": int(b.seed), "run_a": a.run_id, "run_b": b.run_id,
                         "same_nodes": a.hosts == b.hosts, "last100": d[(d.index > 400) & (d.index <= 500)].mean(), "step500": d.get(500, float("nan")), "early": d[(d.index > 50) & (d.index <= 100)].mean(),
                         "synthetic": bool(a.synthetic or b.synthetic)})

    for model, g in r.groupby("model"):
        bf = g[g.precision == "bf16"]
        for a in g[g.precision != "bf16"].itertuples():
            same_seed = bf[bf.seed == a.seed]
            same_nodes = same_seed[same_seed.hosts == a.hosts]
            for b in (same_nodes if len(same_nodes) else same_seed.iloc[:1]).itertuples():
                add("recipe", a, b)
        for (_, _), s in g.groupby(["precision", "seed"]):
            rows_s = list(s.itertuples())
            for i, a in enumerate(rows_s):
                for b in rows_s[:i]:
                    add("same seed", a, b)
        for _, s in g.groupby("precision"):
            first = list(s.groupby("seed").head(1).itertuples())
            for i, a in enumerate(first):
                for b in first[:i]:
                    add("other seed", a, b)
    return pd.DataFrame(rows)


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


REF_NODES = "trig0147;trig0149;trig0150;trig0151"  # the reference run's fixed nodes, from its third run on


def drift(r: pd.DataFrame) -> tuple[pd.Series, float]:
    """Reference runs (R): hours since the first one, and the linear trend over the window (%), on the fixed nodes only."""
    hours = (pd.to_datetime(r.started) - pd.to_datetime(r.started).min()).dt.total_seconds() / 3600
    fixed = r.hosts == REF_NODES if not r.synthetic.all() else pd.Series(True, index=r.index)
    if fixed.sum() < 3:
        return hours, float("nan")
    slope, _ = np.polyfit(hours[fixed], r.tok_s_gpu[fixed], 1)
    return hours, 100 * slope * (hours[fixed].max() - hours[fixed].min()) / r.tok_s_gpu[fixed].median()


def e1_changes(runs: pd.DataFrame, steps: pd.DataFrame) -> pd.DataFrame:
    """E1, each set together against alone: same batch ids (same seed and data), only steps inside the common window of its draw's
    together runs, after batch 10. Mean step time, as the run-and-monitor agent's qwen/e1/compare.py; the median beside it."""
    r = completed(runs, "E1")
    if r.empty or steps.empty or r.rates_t0.isna().all():
        return pd.DataFrame()
    s = steps[steps.run_id.isin(r.run_id)].merge(r[["run_id", "rates_t0"]], on="run_id")
    s = s.assign(time=s.rates_t0 + s.t_s)
    rows = []
    for draw, g in r.groupby(r.node_set.str[:2]):
        together = g[g.condition == "together"]
        here = s[s.run_id.isin(together.run_id)]
        t0 = here[here.step > 10].groupby("run_id").time.min().max()
        t1 = here.groupby("run_id").time.max().min()
        for node_set, d in g.groupby("node_set"):
            if set(d.condition) != {"alone", "together"}:
                continue
            alone_run, together_run = d[d.condition == "alone"].iloc[0], d[d.condition == "together"].iloc[0]
            w = s[(s.run_id == together_run.run_id) & (s.time >= t0) & (s.time <= t1)].set_index("step").step_s
            a = s[s.run_id == alone_run.run_id].set_index("step").step_s
            ids = w.index.intersection(a.index)
            if len(ids) < 10:
                continue
            rows.append({"draw": draw, "node_set": node_set, "steps": len(ids), "window_s": t1 - t0, "alone_s": a[ids].mean(), "together_s": w[ids].mean(),
                         "change_pct": 100 * (w[ids].mean() / a[ids].mean() - 1), "change_median_pct": 100 * (w[ids].median() / a[ids].median() - 1),
                         "tok_s_gpu_alone": alone_run.tokens_per_gpu / a[ids].mean(), "tok_s_gpu_together": together_run.tokens_per_gpu / w[ids].mean(),
                         "slow_nodes": alone_run.slow_nodes if isinstance(alone_run.slow_nodes, str) else "", "bw_groups": alone_run.bw_groups,
                         "synthetic": bool(alone_run.synthetic or together_run.synthetic)})
    return pd.DataFrame(rows)


def with_energy(runs: pd.DataFrame, power: pd.DataFrame) -> pd.DataFrame:
    """Runs with GPU power -> + mtok_kwh: million tokens per kWh of GPU energy (GPU power only).
    The mean power of the GPUs sampled stands for every GPU of the run."""
    if runs.empty or power.empty:
        return pd.DataFrame()
    e = runs.merge(power.drop(columns="synthetic"), on="run_id")
    return e.assign(mtok_kwh=e.tok_s_gpu / e.power_w_gpu * 3.6e6 / 1e6)
