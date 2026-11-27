"""Figures of the report, one function per figure (tables from build.py in, PDF out).

Every function takes the tables dict and returns a matplotlib figure, or None
when its data is missing. See style.py for the visual conventions.
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import style
from .topology import compact
from .style import (CLUSTER_COLOR, CLUSTER_MARKER, FAMILY_COLOR, FAMILY_LABEL, FAMILY_MARKER, GRAY,
                    LIGHT, OKABE_ITO, TEXT_WIDTH)

QWEN = "qwen3_8-27B"
CURRENT_TOKENS = 12288  # 3 x 4,096 tokens per GPU per step (current default)


def _qwen_full(runs: pd.DataFrame, tokens=CURRENT_TOKENS) -> pd.DataFrame:
    r = runs[(runs.status == "completed") & (runs.model == QWEN) & (runs.n_layers == 64)
             & (runs.tokens_per_gpu == tokens) & (runs.grad_accum == 1) & (runs.adam_moments == "fp32")]
    return r[~r.run_name.str.contains("long")]


def _ceiling(runs: pd.DataFrame, cluster: str) -> float | None:
    """1-node MFU (16 layers, same tokens per GPU): compute ceiling without the network."""
    r = runs[(runs.status == "completed") & (runs.cluster == cluster) & (runs.nodes == 1)
             & (runs.tokens_per_gpu == CURRENT_TOKENS) & (runs.grad_accum == 1) & ~runs.synthetic]
    return float(r.mfu_pct.median()) if len(r) else None


def _hollow(ax, start: int) -> None:
    """Hollow markers for the scatter collections added since index `start` (synthetic)."""
    for coll in ax.collections[start:]:
        fc = coll.get_facecolors()
        if len(fc):
            coll.set_edgecolors(fc)
            coll.set_facecolors("white")
            coll.set_linewidths(0.8)


def _scatter(ax, df, x, y, hue=None, style_col=None, palette=None, markers=None, size=22, **kw):
    """sns.scatterplot, measured rows filled and synthetic rows hollow."""
    for syn in (False, True):
        part = df[df.synthetic == syn]
        if part.empty:
            continue
        start = len(ax.collections)
        sns.scatterplot(data=part, x=x, y=y, hue=hue, style=style_col, palette=palette,
                        markers=markers, s=size, linewidth=0, legend=False, ax=ax, zorder=3, **kw)
        if syn:
            _hollow(ax, start)


def _gpu_axis(ax, values) -> None:
    ax.set_xscale("log", base=2)
    lo, hi = min(values), max(values)
    ticks = [2**k for k in range(0, 10) if lo / 1.5 <= 2**k <= hi * 1.5]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.minorticks_off()


# -- 1. Hardware baseline ----------------------------------------------------

def health_tt(t):
    """Per-GPU matmul and per-node intra-node all-reduce, by leaf (tt_healthcheck, measured)."""
    h = t["node_health"]
    h = h[(h.source == "tt_health") & (h.cluster == "rorqual")]
    if h.empty:
        return None
    h = h.assign(leaf_short=h.leaf.str.replace("ihpc", "").str.replace("u25", ""))
    order = sorted(h.leaf_short.unique())
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 1.9))
    for ax, metric, label, frac in (
        (axes[0], "matmul_tflops", "bf16 matmul (TFLOPS per GPU)", 0.95),
        (axes[1], "intra_busbw_gbs", "intra-node all-reduce bus BW (GB/s)", 0.90),
    ):
        d = h[h.metric == metric]
        sns.stripplot(data=d, x="leaf_short", y="value", order=order, color=GRAY, size=2.5,
                      jitter=0.25, ax=ax, zorder=3)
        med = d.value.median()
        ax.axhline(med, color=LIGHT, lw=0.8, zorder=1)
        ax.axhline(frac * med, color=OKABE_ITO["vermillion"], lw=0.8, ls=(0, (1, 2)))
        ax.text(len(order) - 0.5, frac * med, f" exclusion threshold ({frac:.0%} of median)",
                fontsize=6, color=OKABE_ITO["vermillion"], va="bottom", ha="right")
        ax.text(len(order) - 0.5, med, f" median {med:.0f}", fontsize=6, color=GRAY, va="bottom", ha="right")
        ax.set_xlabel("leaf switch (ihpc…u25)")
        ax.set_ylabel(label)
        style.finish(ax)
    fig.tight_layout()
    return fig


def health_srygaard(t):
    """Window health check (srygaard): per-node bandwidths by leaf, with the tool's thresholds."""
    h = t["node_health"]
    h = h[(h.source == "srygaard") & (h.cluster == "rorqual")]
    if h.empty:
        return None
    # One health-check run: the one covering the most nodes (earliest if tied).
    cover = h.groupby("job_id").host.nunique()
    h = h[h.job_id == cover[cover == cover.max()].index.min()]
    h = h[h.metric != "verdict"].assign(
        leaf_short=lambda d: d.leaf.str.replace("ihpc", "").str.replace("u25", ""))
    order = sorted(h.leaf_short.unique())
    # slurm-healthcheck defaults (healthcheck/config.py, Nccl): absolute floors, and for pairs
    # also a fraction of the median.
    metrics = [("intra_busbw_gbs", "intra-node (NVLink)", lambda med: 300),
               ("loopback_busbw_gbs", "through the node's NIC (loopback)", lambda med: 5),
               ("pair_busbw_gbs", "node pairs (ring)", lambda med: max(20, 0.7 * med))]
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 1.9))
    for ax, (metric, title, floor_of) in zip(axes, metrics):
        d = h[h.metric == metric]
        med = d.value.median()
        floor = floor_of(med)
        start = len(ax.collections)
        sns.stripplot(data=d[d.value >= floor], x="leaf_short", y="value", order=order, color=GRAY,
                      size=2.2, jitter=0.25, ax=ax)
        if d.synthetic.any():
            _hollow(ax, start)
        ax.axhline(floor, color=OKABE_ITO["vermillion"], lw=0.8, ls=(0, (1, 2)))
        ax.text(len(order) - 0.5, floor, f"fail < {floor:.0f}", fontsize=6, color=OKABE_ITO["vermillion"],
                va="bottom", ha="right")
        ax.axhline(med, color=LIGHT, lw=0.8, zorder=1)
        low = d[d.value < floor]
        # A slow pair marks both nodes; the culprit is slow in both of its pairs (ring pairs).
        culprits = set(low.host.value_counts().loc[lambda v: v >= 2].index) if metric == "pair_busbw_gbs" \
            else set(low.host)
        for _, b in low.iterrows():
            xi = order.index(b.leaf_short)
            ax.plot(xi, b.value, "x", color=OKABE_ITO["vermillion"], ms=4, mew=1)
            if b.host in culprits:
                culprits.discard(b.host)
                ax.annotate(b.host, (xi, b.value), xytext=(3, -2), textcoords="offset points",
                            fontsize=6, color=OKABE_ITO["vermillion"])
        ax.set_ylim(0.9 * min(floor, d.value.min()), None)
        ax.set_title(title)
        ax.set_xlabel("leaf switch")
        ax.set_ylabel("bus BW (GB/s)" if ax is axes[0] else "")
        ax.tick_params(axis="x", labelrotation=90, labelsize=5)
        style.finish(ax)
    style.synthetic_tag(fig, h.synthetic.any())
    fig.tight_layout()
    return fig


def health_gpus(t):
    """Per-GPU compute on every cluster: is an H100 an H100 everywhere? (health-check matmul)"""
    h = t["node_health"]
    h = h[h.metric == "matmul_tflops"]
    if h.empty:
        return None
    clusters = [c for c in CLUSTER_COLOR if c in set(h.cluster)]
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH * 0.62, 1.9))
    start = len(ax.collections)
    sns.stripplot(data=h, x="cluster", y="value", order=clusters, hue="cluster", palette=CLUSTER_COLOR,
                  size=2.2, jitter=0.3, legend=False, ax=ax)
    if h.synthetic.any():
        _hollow(ax, start)
    for i, c in enumerate(clusters):
        d = h[h.cluster == c]
        med = d.value.median()
        ax.plot([i - 0.4, i + 0.4], [med, med], color="black", lw=0.8)
        ax.plot([i - 0.4, i + 0.4], [0.95 * med] * 2, color=OKABE_ITO["vermillion"], lw=0.8, ls=(0, (1, 2)))
        ax.text(i + 0.42, med, f"{med:.0f}", fontsize=6, va="center")
        ax.text(i + 0.42, 0.95 * med, "−5 %", fontsize=6, va="center", color=OKABE_ITO["vermillion"])
        ax.text(i, d.value.max() + 6, f"{d.host.nunique()} nodes, {len(d)} GPUs", fontsize=6,
                ha="center", color=GRAY)
    ax.set_xlabel("")
    ax.set_ylabel("bf16 matmul (TFLOPS per GPU)")
    style.finish(ax)
    style.synthetic_tag(fig, h.synthetic.any())
    fig.tight_layout()
    return fig


# -- 2. Fabric and topology --------------------------------------------------

def placement_at_scale(t):
    """At scale, balance or pack? Beyond one switch (> 12 nodes on Rorqual), 3 per switch (balanced)
    vs. fewest switches (packed), on the plane total throughput (x) vs. throughput per GPU (y). Runs
    with the same GPU count lie on a ray through the origin (total = per GPU x GPUs): along a ray,
    placement; from ray to ray, scale."""
    r = _qwen_full(t["runs"])
    r = r[(r.cluster == "rorqual") & r.family.isin(["H", "P"]) & (r.nodes > 12)].copy()
    if r.empty:
        return None
    camp = r.groupby("campaign").size().idxmax()
    r = r[r.campaign == camp]
    r["total"] = r.tok_s_gpu * r.gpus / 1e3
    fam_color = {"H": OKABE_ITO["vermillion"], "P": OKABE_ITO["green"]}
    fam_marker = {"H": "^", "P": "s"}
    fam_label = {"H": "3 per switch (balanced)", "P": "fewest switches (packed)"}
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 2.8))
    y_lo, y_hi = r.tok_s_gpu.min() * 0.985, r.tok_s_gpu.max() * 1.015
    x_hi = r.total.max() * 1.12
    ys = np.array([y_lo, y_hi])
    groups = []  # rays whose labels would collide share one label
    for gpus in sorted(r.gpus.unique()):  # iso-GPU rays: total = per GPU x GPUs
        ax.plot(ys * gpus / 1e3, ys, color=LIGHT, lw=0.6, zorder=0)
        if groups and (gpus - groups[-1][-1]) * y_hi / 1e3 < 0.06 * x_hi:
            groups[-1].append(gpus)
        else:
            groups.append([gpus])
    for g in groups:
        ax.text(y_hi * np.mean(g) / 1e3, y_hi, " / ".join(str(int(x // 4)) for x in g) + " nodes", fontsize=6,
                color=GRAY, ha="center", va="bottom")
    for f, g in r.groupby("family"):
        med = g.groupby("nodes")[["total", "tok_s_gpu"]].median()
        ax.plot(med.total, med.tok_s_gpu, color=fam_color[f], lw=0.8, zorder=2)
        for syn in (False, True):
            part = g[g.synthetic == syn]
            ax.plot(part.total, part.tok_s_gpu, ls="", marker=fam_marker[f], ms=4.5, zorder=3,
                    **style.marker_kw(fam_color[f], syn))
    # Placement gap at equal node count, and scale step between consecutive node counts (packed).
    for n, g in r.groupby("nodes"):
        if set(g.family) >= {"H", "P"}:
            h, p = (g[g.family == f].tok_s_gpu.median() for f in ("H", "P"))
            ax.annotate(f"balanced {100 * (h / p - 1):+.1f} %", (g.total.max(), g.tok_s_gpu.max()), xytext=(6, 2),
                        textcoords="offset points", fontsize=6, color=GRAY)
    pk = r[r.family == "P"].groupby("nodes")[["total", "tok_s_gpu"]].median()
    for (n0, a), (n1, b) in zip(pk.iterrows(), list(pk.iterrows())[1:]):
        ax.annotate(f"{int(n0)}→{int(n1)} nodes: per GPU {100 * (b.tok_s_gpu / a.tok_s_gpu - 1):+.1f} %, "
                    f"total {100 * (b.total / a.total - 1):+.0f} %", ((a.total + b.total) / 2, (a.tok_s_gpu + b.tok_s_gpu) / 2),
                    xytext=(0, -22), textcoords="offset points", ha="center", fontsize=6, color=fam_color["P"],
                    bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
    unpaired = sorted(f"{f}{int(n)}" for n, g in r.groupby("nodes") if g.family.nunique() < 2 for f in g.family.unique())
    ax.set_xlim(0, x_hi)
    ax.set_ylim(y_lo, y_hi * 1.01)
    ax.set_xlabel("total throughput (thousand tokens/s)")
    ax.set_ylabel("tokens/s per GPU")
    ax.set_title(f"rorqual, {camp}: beyond one switch", loc="left")
    handles = [plt.Line2D([], [], ls="", marker=fam_marker[f], color=fam_color[f], markerfacecolor="white",
                          label=fam_label[f]) for f in ("H", "P")]
    handles.append(plt.Line2D([], [], color=LIGHT, lw=0.8, label="same GPU count"))
    ax.legend(handles=handles, loc="lower left", handletextpad=0.4)
    style.finish(ax, trim=False)
    if unpaired:
        fig.text(0.01, 0.01, "Not paired (different node counts): " + ", ".join(unpaired), fontsize=6,
                 style="italic", color=GRAY, va="bottom")
    style.synthetic_tag(fig, r.synthetic.any())
    fig.tight_layout(rect=(0, 0.05 if unpaired else 0, 1, 1))
    return fig


def batch_effect(t):
    """Tokens per GPU per step vs throughput: is the step dominated by per-step communication?"""
    r = t["runs"]
    r = r[(r.status == "completed") & ~r.synthetic & (r.model == QWEN) & (r.grad_accum == 1)
          & (r.tp == 1) & (r.dp_replicate == 1)
          & (((r.nodes == 1) & (r.n_layers == 16)) | ((r.nodes > 1) & (r.n_layers == 64)))]
    r = r[r.nodes <= 3]
    if r.empty:
        return None
    r = r.assign(label=r.mb_seqs.astype(int).astype(str) + "×" + (r.seq_len // 1024).astype(int).astype(str) + "k")
    order = r.sort_values("tokens_per_gpu").drop_duplicates("tokens_per_gpu")
    xpos = {tok: i for i, tok in enumerate(order.tokens_per_gpu)}
    panels = sorted(r.nodes.unique())
    fig, axes = plt.subplots(1, len(panels), figsize=(TEXT_WIDTH, 1.8))
    for ax, n in zip(np.atleast_1d(axes), panels):
        d = r[r.nodes == n].assign(x=lambda d: d.tokens_per_gpu.map(xpos))
        sns.lineplot(data=d, x="x", y="tok_s_gpu", hue="cluster", palette=CLUSTER_COLOR,
                     estimator="median", errorbar=None, legend=False, ax=ax, lw=0.8)
        _scatter(ax, d, "x", "tok_s_gpu", hue="cluster", style_col="cluster", palette=CLUSTER_COLOR,
                 markers=CLUSTER_MARKER)
        for c, g in d.groupby("cluster"):
            last = g.loc[g.x.idxmax()]
            ax.annotate(c, (last.x, last.tok_s_gpu), xytext=(4, 0), textcoords="offset points",
                        fontsize=6, va="center", color=CLUSTER_COLOR[c])
        ax.set_xticks(list(xpos.values()))
        ax.set_xticklabels([f"{l}\n{tok // 1024}k" for l, tok in zip(order.label, order.tokens_per_gpu)])
        ax.set_xlim(-0.3, len(xpos) - 0.4)
        ax.set_title(f"{int(n)} node{'s' if n > 1 else ''} ({'16 layers' if n == 1 else 'full model'})"
                     + (", bf16 moments" if n == 2 else ""))
        ax.set_xlabel("sequences × length, tokens per GPU")
        ax.set_ylabel("tokens/s per GPU" if n == panels[0] else "")
        style.finish(ax)
    fig.tight_layout()
    return fig


def comm_scopes(t):
    """Bus bandwidth by scope: where does bandwidth drop (NVLink, NIC, leaf, spine)?

    Only the scopes the health checks measure: one node, NIC loopback, node pairs, and the
    all-reduce / all-to-all of all the job's nodes (one leaf with --job-per switch, every
    leaf with --job-per cluster; tt_healthcheck: the allocation's nodes). Peak bus bandwidth
    over the tested sizes (128 MiB-1 GiB).
    """
    c = t["comm"]
    c = c[(c.cluster == "rorqual") & c.size_bytes.isna() | (c.source == "tt_health")]
    c = c[c.cluster == "rorqual"]
    if c.empty:
        return None

    def scope(row):
        if row.kind == "intra":
            return "one node (NVLink)"
        if row.kind == "loopback":
            return "one node, through its NIC"
        if row.kind == "pair":
            if row.source == "nccl_tests":
                leaves = " ↔ ".join(l.replace("ihpc", "").replace("u25", "") for l in row.leaves.split(";"))
                return f"2 nodes, {leaves}, {row.condition}"
            return "2 nodes"
        short = [l.replace("ihpc", "").replace("u25", "") for l in row.leaves.split(";")]
        where = ("one leaf" if row.n_switches == 1 else
                 "+".join(sorted(short)) if row.n_switches <= 3 else f"{row.n_switches} leaves")
        coll = "all-to-all" if row.collective == "alltoall" else "all-reduce"
        return f"{coll}, {where}"

    c = c[~((c.kind == "group") & (c.nodes == 1))]  # 1-node allocation: same as intra
    c = c.assign(scope=c.apply(scope, axis=1))
    span = c.groupby("scope").nodes.agg(["min", "max"])
    names = {s: s + (f" ({lo}–{hi} nodes)" if lo != hi else f" ({lo} nodes)" if s.startswith("all-") else "")
             for s, (lo, hi) in span.iterrows()}
    c = c.assign(scope=c.scope.map(names))
    order = (c.groupby("scope").agg(coll=("collective", "first"), bw=("busbw_gbs", "median"))
             .sort_values(["coll", "bw"], ascending=[True, False]).index.tolist())
    c = c.assign(y=c.scope.map({s: i for i, s in enumerate(order)}))
    rng = np.random.default_rng(0)
    c = c.assign(yj=c.y + rng.uniform(-0.15, 0.15, len(c)))
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH * 0.75, 0.6 + 0.17 * len(order)))
    _scatter(ax, c, "busbw_gbs", "yj", hue="source",
             palette={"tt_health": GRAY, "srygaard": OKABE_ITO["blue"], "nccl_tests": OKABE_ITO["vermillion"]},
             size=10, alpha=0.8)
    ax.set_xscale("log")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.invert_yaxis()
    ax.set_xlabel("peak bus bandwidth (GB/s, log scale)")
    ax.set_ylabel("")
    ax.axvline(25, color=LIGHT, lw=0.8, ls=(0, (1, 2)))
    ax.text(25, -0.6, "200 Gb/s", fontsize=6, color=GRAY, ha="center", va="bottom")
    style.finish(ax, trim=False)
    style.synthetic_tag(fig, c.synthetic.any())
    fig.tight_layout()
    return fig


# -- 3. Scaling ----------------------------------------------------------------

def weak_scaling(t):
    """Throughput per GPU from one switch to the full machine (weak scaling), with the
    no-network ceiling: single-node MFU (16 layers) converted to full-model tokens/s (est.)."""
    runs = t["runs"]
    r = _qwen_full(runs)
    if r.empty:
        return None
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 2.4))
    flops, peak = r.flops_per_token.iloc[0], r.peak_flops_per_gpu.iloc[0]
    top = r.tok_s_gpu.max()
    labels = []  # (y, x, text, color): placed after the loop, pushed apart vertically
    for c, d in r.groupby("cluster"):
        color = CLUSTER_COLOR[c]
        med = d.groupby("gpus").tok_s_gpu.median()
        sns.lineplot(x=med.index, y=med.values, color=color, lw=0.9, ax=ax, legend=False, zorder=2)
        _scatter(ax, d, "gpus", "tok_s_gpu", hue="cluster", style_col="cluster", palette=CLUSTER_COLOR,
                 markers=CLUSTER_MARKER, size=14, alpha=0.9)
        first, last = med.iloc[0], med.iloc[-1]
        text = c if len(med) < 2 else (f"{c}: {100 * (last / first - 1):+.0f} % from {int(med.index[0])} "
                                       f"to {int(med.index[-1])} GPUs")
        labels.append((last, med.index[-1], text, color))
        ceiling = _ceiling(runs, c)
        if ceiling:
            y = ceiling / 100 * peak / flops
            top = max(top, y)
            ax.axhline(y, color=color, lw=0.7, ls=(0, (4, 3)), zorder=1)
            ax.text(r.gpus.max() * 1.3, y, f"{c}: no inter-node traffic (est. from 1 node, 16 layers)",
                    fontsize=6, color=color, va="bottom", ha="right")
    gap = 0.05 * 1.1 * top
    placed = []
    for y, x, text, color in sorted(labels, reverse=True):  # downward: ceilings are above the lines
        y_text = min([y - 0.5 * gap] + [p - gap for p in placed])
        placed.append(y_text)
        ax.annotate(text, (x, y), xytext=(x * 1.12, y_text), textcoords="data", va="center",
                    fontsize=6.5, color=color)
    _gpu_axis(ax, r.gpus)
    ax.set_xlim(r.gpus.min() / 1.3, r.gpus.max() * 1.3)
    ax.set_ylim(0, 1.1 * top)
    ax.set_xlabel("GPUs (4 per node), log scale")
    ax.set_ylabel("tokens/s per GPU")
    style.finish(ax, trim=False)
    style.synthetic_tag(fig, r.synthetic.any())
    fig.tight_layout()
    return fig


def placement_scaling(t):
    """More nodes vs. crossing switches (3-16 nodes): the one-switch line is the cost of more nodes
    alone (band: run-to-run spread); the spread placements on the same axis show what crossing switches
    adds. Relative to each cluster's one-switch run at 3 nodes; one campaign per cluster."""
    runs = t["runs"]
    r = _qwen_full(runs)
    r = r[r.family.isin(["S", "H", "O"]) & (r.nodes <= 16)].copy()
    if r.empty:
        return None
    # One campaign per cluster (the one with the most placement runs): no mixing of traffic conditions.
    best = r.groupby(["cluster", "campaign"]).size().reset_index().sort_values(0).drop_duplicates("cluster", keep="last")
    r = r.merge(best[["cluster", "campaign"]], on=["cluster", "campaign"])
    rep = r.groupby(["cluster", "variant"]).tok_s_gpu
    count = rep.transform("count")
    spread = 100 * (rep.transform("max") - rep.transform("min")) / rep.transform("median")
    noise = float(spread[count > 1].median()) if (count > 1).any() else 0.0
    fam_marker = {"S": "o", "H": "^", "O": "D"}
    fam_label = {"S": "one switch", "H": "3 per switch", "O": "1 per switch"}
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 2.9))
    ax.axhline(0, color=LIGHT, lw=0.6, zorder=0)
    for c, d in r.groupby("cluster"):
        col = CLUSTER_COLOR[c]
        s3 = d[(d.family == "S") & (d.nodes == d[d.family == "S"].nodes.min())].tok_s_gpu.median()
        d = d.assign(rel=100 * (d.tok_s_gpu / s3 - 1))
        med = {f: g.groupby("nodes").rel.median() for f, g in d.groupby("family")}
        if "S" not in med:
            continue
        s = med["S"]
        ax.fill_between(s.index, s - noise, s + noise, color=col, alpha=0.18, lw=0)
        ax.plot(s.index, s.values, color=col, lw=1.6, zorder=3)
        for f, m in med.items():
            if f == "H":
                m = m[(m.index > 3) & (m.index <= 12)]  # 3 nodes on one 3-node leaf is one switch: a control
            if f != "S":
                ax.plot(m.index, m.values, color=col, lw=0.7, zorder=3)
            for syn in (False, True):
                part = d[(d.family == f) & d.nodes.isin(m.index) & (d.synthetic == syn)]
                pts = part.groupby("nodes").rel.median()
                ax.plot(pts.index, pts.values, ls="", marker=fam_marker[f], ms=3.8 if f != "S" else 4.2, zorder=4,
                        **style.marker_kw(col, syn))
        if "O" in med:
            common = [n for n in med["O"].index if n in s.index]
            if c == "rorqual" and common:
                ax.fill_between(common, s[common], med["O"][common], color=col, alpha=0.10, lw=0)
                n_mid = common[len(common) // 2]
                ax.annotate("what crossing switches adds\n(1 per switch)", (n_mid, (s[n_mid] + med["O"][n_mid]) / 2),
                            xytext=(-70, -40), textcoords="offset points", fontsize=6, color=col,
                            arrowprops=dict(arrowstyle="-", lw=0.4, color=col))
        if c == "rorqual":
            ax.annotate("one switch: more nodes alone", (s.index.max(), s.iloc[-1]), xytext=(5, 4),
                        textcoords="offset points", fontsize=6, color=col)
    # Leaf-to-leaf control: two one-switch runs on different leaves, same campaign and isolation.
    ctl = runs[(runs.status == "completed") & (runs.family == "S") & (runs.isolation == "reserved")]
    for (c, camp, n), g in ctl.groupby(["cluster", "campaign", "nodes"]):
        if g.leaves.nunique() < 2:
            continue
        by_leaf = g.groupby("leaves").tok_s_gpu.median()
        delta = 100 * (by_leaf.max() / by_leaf.min() - 1)
        leaves = " vs ".join(l.replace("u25", "") for l in by_leaf.index)
        base = r[(r.cluster == c) & (r.family == "S")]
        if base.empty:
            continue
        s3 = base[base.nodes == base.nodes.min()].tok_s_gpu.median()
        y0 = 100 * (base[base.nodes == n].tok_s_gpu.median() / s3 - 1) if (base.nodes == n).any() else 0
        ax.plot([n, n], [y0, y0 + delta], color=GRAY, lw=0.8)
        ax.plot(n, y0 + delta, "X", ms=5, color=GRAY, zorder=5)
        ax.annotate(f"control: {leaves}, one switch each ({camp}, measured): {delta:+.1f} %", (n, y0 + delta),
                    xytext=(8, 4), textcoords="offset points", ha="left", va="bottom", fontsize=6, color=GRAY)
    ax.set_xticks(sorted(r.nodes.unique()))
    ax.set_xlim(r.nodes.min() - 0.6, r.nodes.max() + 3.5)
    ax.set_xlabel("nodes")
    ax.set_ylabel("throughput per GPU vs.\none switch, 3 nodes (%)")
    fams = [f for f in fam_marker if f in set(r.family)]
    clusters = [c for c in CLUSTER_COLOR if c in set(r.cluster)]
    handles = ([plt.Line2D([], [], ls="", marker=fam_marker[f], color="black", markerfacecolor="white", label=fam_label[f])
                for f in fams] + [plt.Line2D([], [], color=CLUSTER_COLOR[c], lw=1.2, label=c) for c in clusters]
               + [plt.Line2D([], [], color=LIGHT, lw=5, alpha=0.6, label=f"run-to-run ±{noise:.2f} %")])
    ax.legend(handles=handles, loc="lower left", ncol=3, handletextpad=0.4, columnspacing=1.0)
    style.finish(ax, trim=False)
    style.synthetic_tag(fig, r.synthetic.any())
    fig.tight_layout()
    return fig


def step_decomposition(t):
    """Step time = compute (est. from 1 node, scaled to 64 layers) + exposed communication."""
    runs = t["runs"]
    r = _qwen_full(runs)
    if r.empty:
        return None
    rows = []
    for (c, gpus), d in r.groupby(["cluster", "gpus"]):
        one = runs[(runs.cluster == c) & (runs.nodes == 1) & (runs.status == "completed") & ~runs.synthetic
                   & (runs.tokens_per_gpu == CURRENT_TOKENS) & (runs.grad_accum == 1)]
        if one.empty:
            continue
        compute = one.step_s.median() * d.flops_per_token.iloc[0] / one.flops_per_token.median()
        rows.append({"cluster": c, "gpus": gpus, "step": d.step_s.median(), "compute": compute,
                     "synthetic": d.synthetic.all()})
    if not rows:
        return None
    b = pd.DataFrame(rows).sort_values(["cluster", "gpus"], ascending=[True, False])
    b["label"] = b.cluster + ", " + b.gpus.astype(int).astype(str) + " GPUs"
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH * 0.8, 0.5 + 0.16 * len(b)))
    y = np.arange(len(b))
    colors = b.cluster.map(CLUSTER_COLOR)
    ax.barh(y, b.compute, color="white", edgecolor=colors, hatch="////", lw=0.6, height=0.6)
    ax.barh(y, b.step - b.compute, left=b.compute, color=colors, alpha=0.35, edgecolor=colors,
            lw=0.6, height=0.6)
    for yi, (_, row) in zip(y, b.iterrows()):
        ax.text(row.step, yi, f" {row.step:.1f} s ({100 * (row.step - row.compute) / row.step:.0f}% comm.)",
                va="center", fontsize=6, color=GRAY, style="italic" if row.synthetic else "normal")
    ax.set_yticks(y)
    ax.set_yticklabels(b.label)
    ax.set_xlabel("step time (s): hatched = compute (est.), shaded = exposed communication (est.)")
    style.finish(ax)
    style.synthetic_tag(fig, b.synthetic.any(), "includes synthetic data (italic labels)")
    fig.tight_layout()
    return fig


# -- 4. MoE --------------------------------------------------------------------

def moe_scaling(t):
    """GLM-5: throughput by scale, expert-parallel size and precision."""
    r = t["runs"]
    r = r[(r.status == "completed") & (r.model == "glm5")]
    if r.empty:
        return None
    ep_color = {4: OKABE_ITO["blue"], 8: OKABE_ITO["orange"], 16: OKABE_ITO["vermillion"]}
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.0), sharey=False)
    for ax, (prec, d) in zip(axes, r.groupby("precision")):
        for ep, g in d.groupby("ep"):
            sns.lineplot(data=g, x="gpus", y="mfu_pct", color=ep_color[int(ep)], lw=0.8, ax=ax,
                         legend=False, estimator="median", errorbar=None)
            last = g.loc[g.gpus.idxmax()]
            ax.annotate(f"EP{int(ep)}", (last.gpus, last.mfu_pct), xytext=(4, 0), textcoords="offset points",
                        fontsize=6, va="center", color=ep_color[int(ep)])
        _scatter(ax, d.assign(ep=d.ep.astype(int)), "gpus", "mfu_pct", hue="ep", palette=ep_color, size=14)
        full = d[d.n_layers == d.n_layers_full].gpus.min()
        ax.axvline(full, color=LIGHT, lw=0.7, ls=(0, (1, 2)))
        ax.text(full, ax.get_ylim()[1], "full model → ", fontsize=6, color=GRAY, va="top", ha="right")
        _gpu_axis(ax, d.gpus)
        ax.set_title(prec.upper())
        ax.set_xlabel("GPUs (truncated models left of the line)")
        ax.set_ylabel("MFU on active parameters (%)" if ax is axes[0] else "")
        style.finish(ax)
    style.synthetic_tag(fig, r.synthetic.any())
    fig.tight_layout()
    return fig


def _expert_runs(t):
    """EP8 layout runs of the reference model (EP8 aligned), and the runs left out."""
    r = t["runs"]
    r = r[(r.status == "completed") & (r.family == "EP")].copy()
    if r.empty or "EP8-aligned" not in set(r.variant):
        return None, None
    r["params_trainable_fraction"] = r.params_trainable_fraction.fillna(1.0)
    key = ["cluster", "model", "n_layers", "params_trainable_fraction"]
    ref_rows = r[r.variant == "EP8-aligned"][key].drop_duplicates()
    same = r.merge(ref_rows, on=key, how="left", indicator=True)["_merge"].eq("both").to_numpy()
    return r[same].copy(), r[~same].drop_duplicates("variant")


def _expert_plane(ax, r):
    """More GPUs vs. more total throughput than the largest aligned run, equal scales. Same-node
    layouts (mixed, crossed) sit at x = 0; diagonal = proportional to GPUs; below 0 = less total."""
    aligned = r[r.variant == "EP8-aligned"]
    n_ref = aligned.nodes.iloc[0]
    total_ref = (aligned.tok_s_gpu * aligned.gpus).median()
    r = r.assign(x=100 * (r.nodes / n_ref - 1), y=100 * (r.tok_s_gpu * r.gpus / total_ref - 1))
    first = r.drop_duplicates("variant").set_index("variant")
    name = {"EP8-aligned": "aligned", "EP8-mixed": "mixed", "EP8-crossed": "crossed",
            "EP8-mixed-all": "every node"}
    color = {"EP8-aligned": OKABE_ITO["blue"], "EP8-mixed": OKABE_ITO["orange"],
             "EP8-crossed": OKABE_ITO["vermillion"], "EP8-mixed-all": OKABE_ITO["green"]}
    marker = {"EP8-aligned": "o", "EP8-mixed": "s", "EP8-crossed": "^", "EP8-mixed-all": "D"}
    top = max(r.x.max(), 10) * 1.2
    hi = max(r.x.max(), 10) * 1.55
    lo = min(r.y.min(), 0) - 4
    ax.fill_between([-3, hi], [0, 0], [lo, lo], color=OKABE_ITO["vermillion"], alpha=0.06, lw=0)
    ax.axhline(0, color=GRAY, lw=0.6)
    ax.text(hi, 0.4, "break-even", fontsize=6, color=GRAY, ha="right", va="bottom")
    ax.text(hi, lo / 2, "less total\nthan aligned", fontsize=6, color=OKABE_ITO["vermillion"], ha="right",
            va="center")
    end = top / 1.12
    ax.plot([0, end], [0, end], color=LIGHT, lw=0.8)
    ax.text(end, end, " proportional\n to GPUs", fontsize=6, color=GRAY, ha="left", va="center")
    for v, g in r.groupby("variant"):
        for syn in (False, True):
            part = g[g.synthetic == syn]
            ax.plot(part.x, part.y, ls="", marker=marker[v], ms=4.5, zorder=3, **style.marker_kw(color[v], syn))
        row = first.loc[v]
        med_x, med_y = g.x.median(), g.y.median()
        sep = "\n" if v in ("EP8-aligned", "EP8-mixed-all") else ": "  # x = 0 rows are close: one line
        text = (f"{name[v]} ({int(row.nodes)}){sep}{int(row.ep_groups_crossing)}/{int(row.ep_groups)} cross"
                + (f", {med_y:+.1f} %" if v != "EP8-aligned" else ""))
        dy = {"EP8-aligned": 5, "EP8-mixed-all": -6}.get(v, 0)  # around the zero line
        ax.annotate(text, (med_x, med_y), xytext=(6, dy), textcoords="offset points", ha="left",
                    va="bottom" if dy > 0 else "top" if dy < 0 else "center", fontsize=6, color=color[v])
    if "EP8-mixed-all" in first.index:
        g = r[r.variant == "EP8-mixed-all"]
        x, y = g.x.median(), g.y.median()
        ax.plot([x, x], [y, x], color=OKABE_ITO["green"], lw=0.6, ls=(0, (1, 2)))
        ax.text(x + 0.8, (x + y) / 2, f"{x - y:.0f} points\nbelow\nproportional", fontsize=6,
                color=OKABE_ITO["green"], va="center")
    ax.set_xlim(-3, hi)
    ax.set_ylim(lo, top)
    ax.set_aspect("equal")
    ax.set_xlabel("more GPUs than aligned (%)")
    ax.set_ylabel("more total throughput than aligned (%)")
    ax.set_title(f"vs. the largest aligned run ({int(n_ref)} nodes)")
    style.finish(ax, trim=False)


def _expert_waterfall(ax, r):
    """Total throughput of every node vs. aligned, step by step: proportional gain of the extra nodes,
    cost of crossing groups (mixed vs. aligned on the same nodes, applied to every node: est.), rest
    (larger scale, more simultaneous crossings: est., the remainder)."""
    med = r.groupby("variant").agg(nodes=("nodes", "first"), tok=("tok_s_gpu", "median"))
    n0, t0 = med.loc["EP8-aligned", ["nodes", "tok"]]
    n1, t1 = med.loc["EP8-mixed-all", ["nodes", "tok"]]
    tm = med.loc["EP8-mixed", "tok"]
    ideal = 100 * n1 / n0
    after_cross = ideal * tm / t0
    final = 100 * n1 * t1 / (n0 * t0)
    steps = [  # label, start, length, kind (top to bottom)
        (f"aligned ({int(n0)} nodes)", 0, 100, "measured"),
        (f"+{int(n1 - n0)} nodes (proportional)", 100, ideal - 100, "arithmetic"),
        ("crossing groups", after_cross, ideal - after_cross, "estimated"),
        ("rest: scale, crossings", min(final, after_cross), abs(after_cross - final), "estimated"),
        (f"every node ({int(n1)} nodes)", 0, final, "measured"),
    ]
    deltas = [None, ideal - 100, after_cross - ideal, final - after_cross, None]
    for i, ((label, left, length, kind), delta) in enumerate(zip(steps, deltas)):
        if kind == "measured":
            ax.barh(i, length, left=left, height=0.6, color=OKABE_ITO["blue"] if i == 0 else OKABE_ITO["green"])
            ax.text(left + length + 1.5, i, f"{length:.1f}", va="center", fontsize=6)
        else:
            c = OKABE_ITO["green"] if delta > 0 else OKABE_ITO["vermillion"]
            ax.barh(i, length, left=left, height=0.6, color="white" if kind == "estimated" else c,
                    edgecolor=c, hatch="////" if kind == "estimated" else None, lw=0.6)
            ax.text(max(left + length, left) + 1.5, i, f"{delta:+.1f}" + (" (est.)" if kind == "estimated" else ""),
                    va="center", fontsize=6)
        if i < len(steps) - 1:  # connector to the next bar
            level = 100 + sum(d for d in deltas[1:i + 1])
            ax.plot([level, level], [i + 0.3, i + 0.7], color=LIGHT, lw=0.6)
    ax.axvline(100, color=GRAY, lw=0.5, ls=(0, (1, 2)), zorder=0)
    ax.set_yticks(range(len(steps)))
    ax.set_yticklabels([s[0] for s in steps])
    ax.set_ylim(len(steps) - 0.5, -0.5)
    ax.set_xlim(0, 1.3 * ideal)
    ax.set_xlabel("total throughput (aligned = 100)")
    ax.set_title("where the extra GPUs go")
    style.finish(ax, trim=False)


def expert_placement(t):
    """Expert-group placement (GLM-5): is using every node worth it compared with the largest aligned
    subset? Left: every run on the GPUs / total-throughput plane. Right: breakdown for every node."""
    r, left_out = _expert_runs(t)
    if r is None:
        return None
    full = {"EP8-mixed", "EP8-mixed-all"} <= set(r.variant)
    fig, axes = plt.subplots(1, 2 if full else 1, figsize=(TEXT_WIDTH, 3.0), squeeze=False,
                             gridspec_kw={"width_ratios": [1, 1] if full else [1]})
    _expert_plane(axes[0, 0], r)
    if full:
        _expert_waterfall(axes[0, 1], r)
    n = r.iloc[0]
    fig.suptitle(f"{n.cluster}, {n.model}, EP8 (groups of 2 nodes)", fontsize=8, x=0.01, ha="left")
    if len(left_out):
        notes = "; ".join(f"{v.variant} ({int(v.n_layers)} layers)" for v in left_out.itertuples())
        fig.text(0.99, 0.01, f"Not shown, different model: {notes}", fontsize=6, style="italic", color=GRAY,
                 ha="right", va="bottom")
    style.synthetic_tag(fig, r.synthetic.any())
    fig.tight_layout()
    return fig


def expert_groups(t):
    """All-to-all time per EP group in the mixed layouts: the slowest (crossing) group gates the step."""
    r, _ = _expert_runs(t)
    groups = t["ep_groups"]
    if r is None or groups.empty:
        return None
    g = groups.merge(r[["run_id", "variant", "nodes", "synthetic"]], on="run_id")
    g = g[g.variant.str.startswith("EP8-mixed")]
    g = g[g.run_id.isin(g.groupby("variant").run_id.min())]
    if g.empty:
        return None
    g = g.assign(where=np.where(g.n_switches > 1, "across switches", "inside one switch"),
                 run=g.variant.map({"EP8-mixed": "mixed", "EP8-mixed-all": "mixed, every node"})
                 + " (" + g.nodes.astype(int).astype(str) + " nodes)")
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH * 0.6, 1.9))
    start = len(ax.collections)
    sns.stripplot(data=g, x="run", y="a2a_s", hue="where", dodge=True, size=3, jitter=0.15, ax=ax,
                  palette={"inside one switch": GRAY, "across switches": OKABE_ITO["vermillion"]})
    if g.synthetic.any():
        _hollow(ax, start)
    ax.legend(loc="lower right", handletextpad=0.2)
    ax.set_ylim(0, 1.2 * g.a2a_s.max())
    ax.set_xlabel("")
    ax.set_ylabel("all-to-all per EP group\n(s per step)")
    style.finish(ax)
    style.synthetic_tag(fig, g.synthetic.any())
    fig.tight_layout()
    return fig


# -- 6. Operational readiness ----------------------------------------------------

def startup(t):
    """Time to an initialized model and to the first completed step, by scale."""
    r = t["runs"]
    r = r[(r.status == "completed") & (r.model == QWEN) & (r.n_layers == 64)]
    if r.empty:
        return None
    long = r.melt(id_vars=["cluster", "nodes", "synthetic"], value_vars=["init_s", "first_step_s"],
                  var_name="phase", value_name="seconds")
    long["phase"] = long.phase.map({"init_s": "process start → model initialized",
                                    "first_step_s": "model initialized → first step done"})
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 1.8), sharey=False)
    for ax, (phase, d) in zip(axes, long.groupby("phase", sort=False)):
        _scatter(ax, d, "nodes", "seconds", hue="cluster", style_col="cluster", palette=CLUSTER_COLOR,
                 markers=CLUSTER_MARKER, size=12, alpha=0.9)
        ax.set_xscale("log", base=2)
        ticks = [1, 2, 4, 8, 16, 32, 64]
        ax.set_xticks([x for x in ticks if x <= d.nodes.max() * 1.5])
        ax.set_xticklabels([str(x) for x in ticks if x <= d.nodes.max() * 1.5])
        ax.minorticks_off()
        ax.set_ylim(0, None)
        ax.set_title(phase)
        ax.set_xlabel("nodes")
        ax.set_ylabel("seconds" if ax is axes[0] else "")
        style.finish(ax)
    style.synthetic_tag(fig, r.synthetic.any())
    fig.tight_layout()
    return fig


def stability(t):
    """Throughput per step over a long run, with GPU clocks (nodemon) on the same time axis."""
    steps, runs, mon = t["steps"], t["runs"], t["gpu_monitor"]
    long = runs[runs.bench.str.contains("long") | runs.run_name.str.contains("long")]
    if long.empty or mon.empty:
        return None
    run = long.iloc[0]
    s = steps[steps.run_id == run.run_id].sort_values("t_s")
    m = mon[(mon.source == run.run_name) & (mon.metric == "sm_clock_mhz")].copy()
    m["t_s"] = m.time - m.time.min()
    m["gpu_id"] = m.host + ":" + m.gpu.astype(str)
    fig, axes = plt.subplots(2, 1, figsize=(TEXT_WIDTH, 2.6), sharex=True,
                             gridspec_kw={"height_ratios": [1.3, 1]})
    ax = axes[0]
    sns.lineplot(x=s.t_s / 60, y=s.tok_s_gpu, color=LIGHT, lw=0.5, ax=ax)
    sns.lineplot(x=s.t_s / 60, y=s.tok_s_gpu.rolling(15, center=True).median(), color=CLUSTER_COLOR["rorqual"],
                 lw=1.0, ax=ax)
    ax.set_ylabel("tokens/s per GPU")
    ax.set_title(f"{run.cluster}, {int(run.gpus)} GPUs: per step (gray) and 15-step rolling median")
    style.finish(ax)
    ax = axes[1]
    slowest = m.groupby("gpu_id").value.min().idxmin()
    others = m[m.gpu_id != slowest]
    band = others.groupby("t_s").value.agg(["min", "max", "median"]).reset_index()
    ax.fill_between(band.t_s / 60, band["min"], band["max"], color=LIGHT, alpha=0.6, lw=0)
    sns.lineplot(x=band.t_s / 60, y=band["median"], color=GRAY, lw=0.7, ax=ax)
    sl = m[m.gpu_id == slowest]
    sns.lineplot(x=sl.t_s / 60, y=sl.value, color=OKABE_ITO["vermillion"], lw=1.0, ax=ax)
    ax.annotate(slowest, (sl.t_s.iloc[sl.value.argmin()] / 60, sl.value.min()), xytext=(4, -2),
                textcoords="offset points", fontsize=6, color=OKABE_ITO["vermillion"])
    ax.text(band.t_s.max() / 60, band["max"].max(), "other GPUs: range and median", fontsize=6,
            color=GRAY, ha="right", va="bottom")
    ax.set_ylabel("SM clock (MHz)")
    ax.set_xlabel("minutes since the start of the measurement")
    style.finish(ax)
    style.synthetic_tag(fig, bool(run.synthetic))
    fig.tight_layout()
    return fig


def _cpu_drop(cpu: pd.DataFrame) -> pd.DataFrame:
    """Per host, the lowest 1-minute median busy-core clock, relative to the first minute (%)."""
    c = cpu.copy()
    c["win"] = ((c.time - c.time.min()) // 60).astype(int)
    per = c.groupby(["host", "win"]).value.median().reset_index().sort_values("win")
    first = per.groupby("host").value.first()
    low = per.groupby("host").value.min()
    return (100 * (low / first - 1)).rename("cpu_drop_pct").reset_index()


def _activity(mon):
    """The sharpest GPU-activity signal the telemetry carries, and the label that names it."""
    for metric, label in (("tensor_util", "Tensor-pipe activity"),
                          ("util_mean", "GPU busy fraction"),
                          ("util", "GPU busy fraction\n(point samples)")):
        rows = mon[mon.metric == metric]
        if not rows.empty:
            return rows.copy(), label
    return None, ""


def stragglers(t):
    """Per-GPU activity and per-node CPU clock (nodemon): the slow GPU is the one that waits least."""
    mon = t["gpu_monitor"]
    if mon.empty:
        return None
    util, activity = _activity(mon)
    if util is None:
        return None
    leaf_of = t["ranks"].set_index("host").leaf.to_dict() if not t["ranks"].empty else {}
    nh = t["node_health"]
    leaf_of.update(nh.drop_duplicates("host").set_index("host").leaf.to_dict())
    # Per GPU, the 10-minute window where it was most active (straggler episodes are short).
    util["win"] = ((util.time - util.time.min()) // 600).astype(int)
    peak = (util.groupby(["host", "gpu", "win"]).value.median()
            .groupby(["host", "gpu"]).max().rename("util_peak").reset_index())
    agg = peak.merge(util.groupby(["host", "gpu"]).synthetic.any().reset_index(), on=["host", "gpu"])
    agg["leaf"] = agg.host.map(leaf_of).fillna("?").str.replace("ihpc", "").str.replace("u25", "")
    cpu = mon[mon.metric == "cpu_mhz_busy_mean"]
    drop = _cpu_drop(cpu) if not cpu.empty else None
    order = sorted(agg.leaf.unique())

    height = 3.2 if drop is not None else 1.8
    fig, axes = plt.subplots(2 if drop is not None else 1, 1, figsize=(TEXT_WIDTH, height),
                             sharex=True, squeeze=False)
    ax = axes[0][0]
    start = len(ax.collections)
    sns.stripplot(data=agg, x="leaf", y="util_peak", order=order, color=GRAY, size=2, jitter=0.3, ax=ax)
    if agg.synthetic.any():
        _hollow(ax, start)
    out = agg.loc[agg.util_peak.idxmax()]
    ax.plot(order.index(out.leaf), out.util_peak, "o", color=OKABE_ITO["vermillion"], ms=3.5)
    ax.annotate(f"{out.host}:GPU{int(out.gpu)}", (order.index(out.leaf), out.util_peak), xytext=(5, -2),
                textcoords="offset points", fontsize=6, color=OKABE_ITO["vermillion"])
    ax.set_ylabel(f"{activity}\n(max over 10-min windows)")
    style.finish(ax)

    if drop is not None:
        ax = axes[1][0]
        drop["leaf"] = drop.host.map(leaf_of).fillna("?").str.replace("ihpc", "").str.replace("u25", "")
        start = len(ax.collections)
        sns.stripplot(data=drop, x="leaf", y="cpu_drop_pct", order=order, color=GRAY, size=2,
                      jitter=0.3, ax=ax)
        if cpu.synthetic.any():
            _hollow(ax, start)
        worst = drop.loc[drop.cpu_drop_pct.idxmin()]
        ax.plot(order.index(worst.leaf), worst.cpu_drop_pct, "o", color=OKABE_ITO["vermillion"], ms=3.5)
        ax.annotate(worst.host, (order.index(worst.leaf), worst.cpu_drop_pct), xytext=(5, -2),
                    textcoords="offset points", fontsize=6, color=OKABE_ITO["vermillion"])
        ax.axhline(-15, color=LIGHT, lw=0.8, zorder=0)
        ax.set_ylabel("CPU busy-core clock\n(lowest minute vs. first, %)")
        style.finish(ax)
    axes[-1][0].set_xlabel("leaf switch")
    axes[-1][0].tick_params(axis="x", labelrotation=90, labelsize=6)
    style.synthetic_tag(fig, bool(agg.synthetic.any()))
    fig.tight_layout()
    return fig


FIGURES = {
    "health_tt": health_tt,
    "health_srygaard": health_srygaard,
    "health_gpus": health_gpus,
    "batch_effect": batch_effect,
    "comm_scopes": comm_scopes,
    "weak_scaling": weak_scaling,
    "placement_scaling": placement_scaling,
    "placement_at_scale": placement_at_scale,
    "step_decomposition": step_decomposition,
    "moe_scaling": moe_scaling,
    "expert_placement": expert_placement,
    "expert_groups": expert_groups,
    "startup": startup,
    "stability": stability,
    "stragglers": stragglers,
}


def render_all(tables, out_dir, only=None) -> dict[str, bool]:
    style.apply()
    done = {}
    flags = []
    for name, fn in FIGURES.items():
        if only and name not in only:
            continue
        style.USED_SYNTHETIC = False
        fig = fn(tables)
        flags.append(rf"\expandafter\def\csname syn@{name}\endcsname{{{int(style.USED_SYNTHETIC and fig is not None)}}}")
        done[name] = fig is not None
        if fig is not None:
            style.save(fig, out_dir / f"{name}.pdf")
        else:
            (out_dir / f"{name}.pdf").unlink(missing_ok=True)  # never leave a stale figure
    if not only:
        (out_dir / "synthetic.tex").write_text("\n".join(flags) + "\n")
    return done
