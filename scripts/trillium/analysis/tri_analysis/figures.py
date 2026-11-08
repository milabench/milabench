"""Figures of the Trillium report, one function per figure ID of DESIGN.md (tables in, figure out).

Every function takes the tables dict and returns a matplotlib figure, or None when its data is
missing. Conventions (style.py): measured points filled, synthetic points hollow, and a
SYNTHETIC watermark in every figure drawn wholly or partly from synthetic data.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter, LogLocator, ScalarFormatter
from tt_analysis.figures import _activity, _cpu_drop

from . import style
from .machine import (GLM_COPY_NODES, GLM_SEQS_PER_COPY, GLM_STRONG_SEQS, GPU_MEMORY_GIB, GPUS_PER_NODE, NET_D2_JOBS, NET_D2_AGAIN_JOB, NET_D3_INSIDE_JOB, NET_D3_JOB, NET_FULL_JOB,
                      NODE_NET_GBS, NODES, SUBNET_OF)
from .style import (BLUE, COLLECTIVE_COLOR, COLLECTIVE_MARKER, CRITICAL, GLM, INK2, LIGHT, MUTED, QWEN,
                    RECIPE_COLOR, SCALE_COLOR, SEQUENTIAL, SERIES, TEXT_WIDTH, TOKENS_COLOR, WORKLOAD_COLOR,
                    WORKLOAD_LABEL, marker_kw)
from .select import (CAPPED_NODES, REF_NODES, SLOW_NODES, checkpoints, completed, drift, e1_changes, failed_runs, loss_gaps, pinned_blocked, qwen_base,
                     with_energy)

QWEN_LAYOUTS = ["fsdp", "hsdp8", "hsdp16", "tp2-fsdp", "tp4-fsdp", "pp2-fsdp", "pp4-fsdp"]
GLM_LAYOUTS = ["pp13-ep8", "pp13-ep16", "pp13-ep32", "pp26-ep8", "pp10-ep8", "pp25-ep8", "pp6-ep16"]
POS = {h: i for i, h in enumerate(NODES)}
REF = {"layout": "fsdp", "tokens_per_gpu": 12288, "grad_reduce": "fp32", "ac": "selective", "precision": "bf16"}


def _runs(t, *experiments) -> pd.DataFrame:
    return completed(t["runs"], *experiments)


def _median(r: pd.DataFrame, keys, col="tok_s_gpu") -> pd.DataFrame:
    """Median over repetitions; a configuration is synthetic if any of its runs is."""
    return r.groupby(keys, dropna=False).agg(value=(col, "median"), lo=(col, "min"), hi=(col, "max"),
                                             n=(col, "size"), synthetic=("synthetic", "any")).reset_index()


def _pt(ax, x, y, color, synthetic, marker="o", ms=4.5, **kw):
    style_kw = marker_kw(color, bool(synthetic))
    style_kw["markeredgewidth"] = kw.pop("mew", style_kw["markeredgewidth"])
    ax.plot(x, y, ls="", marker=marker, ms=ms, zorder=3, color=color, **style_kw, **kw)


def _handle(color, label, synthetic, marker="o", ls="", **kw):
    """Legend entry drawn like the data it names: hollow when that data is synthetic."""
    return plt.Line2D([], [], ls=ls, marker=marker, color=color, label=label, **marker_kw(color, bool(synthetic)),
                      **kw)


def _failed_x(ax, x, label=None):
    """A failed measured run with no successful one: a red cross on the x axis (it has no value)."""
    ax.plot(x, 0.04, "x", color=CRITICAL, ms=5, mew=1.2, zorder=6, clip_on=False, transform=ax.get_xaxis_transform())
    if label:
        ax.annotate(label, (x, 0.04), xycoords=ax.get_xaxis_transform(), xytext=(0, 5), textcoords="offset points",
                    fontsize=5.5, color=CRITICAL, ha="center")


def _failed_y(ax, y):
    """Same, for a row of a horizontal plot."""
    ax.plot(0.02, y, "x", color=CRITICAL, ms=5, mew=1.2, zorder=6, clip_on=False, transform=ax.get_yaxis_transform())


def _subnets():
    """(subnet of each node position, positions where a subnet starts, tick centres)."""
    groups = pd.Series(SUBNET_OF).reindex(NODES)
    edges = [i for i in range(1, len(NODES)) if groups.iloc[i] != groups.iloc[i - 1]]
    bounds = [0, *edges, len(NODES)]
    return groups, edges, [(a + b - 1) / 2 for a, b in zip(bounds, bounds[1:])], bounds[:-1]


def _subnet_axis(axes) -> None:
    """x = node position in host order; light lines separate the subnets, the ticks (last axis) name them."""
    axes = list(np.ravel(axes))
    groups, edges, centers, starts = _subnets()
    for a in axes:
        for e in edges:
            a.axvline(e - 0.5, color=LIGHT, lw=0.5, zorder=0)
    ax = axes[-1]
    ax.set_xticks(centers)
    ax.set_xticklabels([groups.iloc[a] for a in starts])
    ax.set_xlim(-0.8, len(NODES) - 0.2)
    ax.tick_params(axis="x", length=0)


def _label_outliers(ax, d, x, y, threshold, label) -> None:
    """Cross the points under the threshold; one label per host (lowest point), naming the whole
    node when all its points are low."""
    low = d[d[y] < threshold]
    for _, row in low.iterrows():
        ax.plot(row[x], row[y], "x", color=CRITICAL, ms=4, mew=1, zorder=4)
    for host, g in low.groupby("host"):
        row = g.loc[g[y].idxmin()]
        text = f"{host}: all" if len(g) == (d.host == host).sum() > 1 else label(row)
        ax.annotate(text, (row[x], row[y]), xytext=(4, -3), textcoords="offset points", fontsize=6, color=CRITICAL)


# -- D. Health and network ---------------------------------------------------

PERIOD_STYLE = {"before": (SERIES[0], "o", "before the kernel change"), "after": (SERIES[1], "s", "after")}
GROUP_EDGE = POS["trig0129"] - 0.5  # the two bandwidth groups (D2)


def _group_edge(axes) -> None:
    for ax in np.ravel(axes):
        ax.axvline(GROUP_EDGE, color=INK2, lw=0.8, ls=(0, (4, 2)), zorder=1)


def _periods(ax, d, ms=2.5) -> None:
    for period, (col, marker, _) in PERIOD_STYLE.items():
        part = d[d.period == period]
        if len(part):
            _pt(ax, part.x, part.value, col, False, marker, ms=ms, mew=0.5)


def _period_legend(fig_or_ax, **kw):
    fig_or_ax.legend(handles=[_handle(c, label, False, m, ms=3) for c, m, label in PERIOD_STYLE.values()], **kw)


def health_gpus(t):
    """Per-GPU compute rate by node: DCGM's FP16 diagnostic (D1; the tool has no matmul check), before and after the kernel change,
    with the median and a -5 % line. The capped nodes' tests are in the machine block."""
    h = t["node_health"]
    if h.empty:
        return None
    d = h[(h.metric == "dcgm_tflops") & h.period.isin(PERIOD_STYLE) & h.host.isin(POS)].copy()
    if d.empty:
        return None
    d["x"] = d.host.map(POS) + (d.gpu.astype(float) - 3.5) * 0.06 + d.period.map({"before": -0.12, "after": 0.12})
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 1.7))
    _periods(ax, d, ms=2.2)
    med = d.value.median()
    ax.axhline(0.95 * med, color=CRITICAL, lw=0.7, ls=(0, (1, 2)), zorder=1)
    ax.text(-0.5, 0.95 * med, "\u22125 %", fontsize=6, color=CRITICAL, ha="left", va="bottom")
    ax.set_ylim(0.93 * med, 1.03 * med)
    ax.set_ylabel("DCGM diagnostic, FP16\n(TFLOP/s per GPU)")
    style.finish(ax, trim=False)
    _subnet_axis(ax)
    _group_edge(ax)
    ax.set_xlabel(f"node ({d.host.nunique()} nodes), grouped by subnet; dashed: the two bandwidth groups")
    _period_legend(ax, loc="lower right", ncol=2, fontsize=6)
    fig.tight_layout()
    return fig


def health_links(t):
    """Per-node NCCL all-reduce: within the node (NVLink), with a partner node, and the NIC loopback (D1): before and after the kernel change."""
    h = t["node_health"]
    if h.empty:
        return None
    h = h[h.host.isin(POS) & h.period.isin(PERIOD_STYLE)]
    if h.empty:
        return None
    panels = (("intra_busbw_gbs", "within a node\n(GB/s)", 0.9), ("pair_busbw_gbs", "with a partner\n(GB/s)", 0.9),
              ("loopback_busbw_gbs", "NIC loopback\n(GB/s)", 0.9))
    fig, axes = plt.subplots(3, 1, figsize=(TEXT_WIDTH, 3.3), sharex=True)
    for ax, (metric, label, frac) in zip(axes, panels):
        d = h[h.metric == metric].copy()
        d["x"] = d.host.map(POS) + d.period.map({"before": -0.12, "after": 0.12})
        _periods(ax, d)
        med = d.value.median()
        ax.axhline(med, color=MUTED, lw=0.7, zorder=1)
        ax.text(-0.5, med, f"median {med:,.3g}", fontsize=6, color=INK2, ha="left", va="bottom",
                bbox={"fc": "white", "ec": "none", "pad": 0.3}, zorder=5)
        low = d[d.value < frac * med]
        if len(low):  # a pair test names both nodes: the node in most low tests is the one to name
            host = low.host.value_counts().idxmax()
            g = low[low.host == host]
            ax.annotate(f"{host} and its partners: {g.value.min():.0f}\u2013{g.value.max():.0f}", (g.x.iloc[0], g.value.min()), xytext=(8, -2),
                        textcoords="offset points", fontsize=6, color=CRITICAL)
        ax.set_ylim(0.75 * med, 1.05 * d.value.max())
        ax.set_ylabel(label)
        style.finish(ax, trim=False)
    _subnet_axis(axes)
    _group_edge(axes)
    axes[-1].set_xlabel("node, grouped by subnet; dashed: the two bandwidth groups")
    _period_legend(axes[0], loc="lower right", ncol=2, fontsize=6)
    fig.tight_layout()
    return fig


def machine_status(t):
    """What differs between nodes. Top: CPU clock at full GPU power in the health check, before and after the kernel change. Bottom: share of CPU
    samples below 500 MHz in 13-node Qwen training, one point per run: E1 and the layout runs. Shaded: 200 W cap, out."""
    h = t["node_health"]
    n = t.get("node_telemetry", pd.DataFrame())
    if h.empty:
        return None
    d = h[(h.metric == "cpu_mhz_full_power") & h.period.isin(PERIOD_STYLE) & h.host.isin(POS)].copy()
    if d.empty:
        return None
    d["x"] = d.host.map(POS) + d.period.map({"before": -0.12, "after": 0.12})
    has_n = len(n) > 0
    fig, axes = plt.subplots(2 if has_n else 1, 1, figsize=(TEXT_WIDTH, 3.4 if has_n else 1.9), sharex=True, squeeze=False)
    axes = axes[:, 0]
    ax = axes[0]
    _periods(ax, d)
    row = d[(d.host == "trig0114") & (d.period == "before")].iloc[0]
    ax.annotate("trig0114", (row.x, row.value), xytext=(5, -2), textcoords="offset points", fontsize=6, color=INK2)
    ax.set_ylim(1200, 3400)
    ax.set_ylabel("CPU clock at full\nGPU power (MHz)")
    style.finish(ax, trim=False)
    _period_legend(ax, loc="upper right", ncol=2, fontsize=6)
    if has_n:
        ax = axes[1]
        runs = t["runs"]
        fsdp = runs[(runs.layout == "fsdp") & (runs.tokens_per_gpu == 12288) & (runs.ac == "selective") & (runs.nodes == 13) & (runs.status == "completed")
                    & runs.experiment.isin(["E1", "A-Q", "post-pstate"])]
        c = n[n.run_id.isin(fsdp.run_id) & n.host.isin(POS)].dropna(subset=["cpu_low_share"]).copy()
        c["x"] = c.host.map(POS)
        _pt(ax, c.x, 100 * c.cpu_low_share, INK2, False, ms=2.2, mew=0.5)
        for host in sorted(SLOW_NODES):
            g = c[c.host == host]
            if len(g):
                ax.annotate(host, (POS[host], 100 * g.cpu_low_share.max()), xytext=(0, 3), textcoords="offset points", fontsize=6, color=CRITICAL, ha="center")
        ax.set_ylim(0, None)
        ax.set_ylabel("CPU samples below\n500 MHz, FSDP 12k (%)")
        style.finish(ax, trim=False)
    for ax in axes:
        for host in CAPPED_NODES:
            ax.axvspan(POS[host] - 0.5, POS[host] + 0.5, color=CRITICAL, alpha=0.13, lw=0, zorder=0)
        ax.axvspan(POS["trig0146"] - 0.5, POS["trig0146"] + 0.5, color=MUTED, alpha=0.3, lw=0, zorder=0)
    _subnet_axis(axes)
    _group_edge(axes)
    axes[-1].set_xlabel("node, grouped by subnet; dashed: the two bandwidth groups; red: 200 W cap from Nov 5 to Nov 6; gray: out until Nov 6")
    fig.tight_layout()
    return fig


def pair_matrix(t):
    """Bandwidth of every node pair, nodes ordered by subnet (D2): November 4, and the two November 7 sweeps, on one log scale."""
    p = t["pairs"]
    if p.empty:
        return None
    p = p[(p.status == "ok") & p.a.isin(POS) & p.b.isin(POS)]
    if p.synthetic.any():
        panels = [(p, "bw_gbs", "bandwidth"), (p, "lat_us", "latency")]
    else:  # GPUs 0, 2, 4, 6: what the fabric carries; the others are bring-up tests
        panels = [(p[p.job_id.isin(NET_D2_JOBS)], "bw_gbs", "Nov 4, 2 jobs of 25 nodes"),
                  (p[p.job_id == NET_FULL_JOB], "bw_gbs", "Nov 7 11:25, 50 nodes"),
                  (p[p.job_id == NET_D2_AGAIN_JOB], "bw_gbs", "Nov 7 15:08, 48 nodes, 2 passes")]
        panels = [x for x in panels if len(x[0])]
    bw = pd.concat([d.bw_gbs for d, col, _ in panels if col == "bw_gbs"])
    shared = all(col == "bw_gbs" for _, col, _ in panels)  # one colorbar for all
    norm = LogNorm(vmin=bw.min(), vmax=bw.max())  # 45, 64, 96 and 189 GB/s all apart
    n = len(NODES)
    fig, axes = plt.subplots(1, len(panels), figsize=(TEXT_WIDTH, 2.45), squeeze=False, layout="constrained")
    axes = axes[0]
    groups, edges, centers, starts = _subnets()
    for ax, (d, col, title) in zip(axes, panels):
        m = np.full((n, n), np.nan)
        agg = d.groupby(["a", "b"])[col].median()
        for (a, b), v in agg.items():
            m[POS[a], POS[b]] = m[POS[b], POS[a]] = v
        im = ax.imshow(m, cmap=SEQUENTIAL, interpolation="nearest", norm=norm if col == "bw_gbs" else None)  # darker = more
        for e in edges:
            ax.axhline(e - 0.5, color="white", lw=0.6)
            ax.axvline(e - 0.5, color="white", lw=0.6)
        ax.axhline(GROUP_EDGE, color=INK2, lw=0.7, ls=(0, (4, 2)))
        ax.axvline(GROUP_EDGE, color=INK2, lw=0.7, ls=(0, (4, 2)))
        ax.set_xticks(centers)
        ax.set_xticklabels([groups.iloc[a] for a in starts], fontsize=6)
        ax.set_yticks(centers)
        ax.set_yticklabels([groups.iloc[a] for a in starts], fontsize=6)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        if not shared:
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cb.set_label("GB/s, one direction" if col == "bw_gbs" else "µs, 8 B", fontsize=7)
            cb.ax.tick_params(labelsize=6)
            cb.outline.set_visible(False)
        ax.set_title(f"{title}\nmedian {np.nanmedian(m):.3g}, range {np.nanmin(m):.3g}–{np.nanmax(m):.3g}", fontsize=6.5)
    if shared:
        cb = fig.colorbar(im, ax=list(axes), fraction=0.02, pad=0.01)
        cb.set_label("GB/s per node, one direction (log scale)", fontsize=7)
        cb.ax.minorticks_off()
        cb.set_ticks([v for v in (45, 64, 96, 189) if bw.min() <= v <= bw.max()])
        cb.ax.yaxis.set_major_formatter(ScalarFormatter())
        cb.ax.tick_params(labelsize=6)
        cb.outline.set_visible(False)
    style.synthetic_mark(fig, p.synthetic.any(), "includes synthetic data")
    return fig


def collectives_scale(t):
    """NCCL bus bandwidth vs nodes (1 GiB) per collective, bisection cuts; bus BW vs size (D3, D4)."""
    c = t["collectives"]
    if c.empty:
        return None
    c = c[c.status == "ok"]
    inside = c[(c.job_id == NET_D3_INSIDE_JOB) & (c.experiment == "D3") & (c.size_bytes == 2**30) & c.nodes.isin([8, 13, 26])]
    if not c.synthetic.any():
        c = c[c.job_id == NET_D3_JOB]
    d3 = c[(c.experiment == "D3") & (c.size_bytes == 2**30)]
    d4 = c[c.experiment == "D4"]
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.5), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    for coll, g in d3.groupby("collective"):
        med = _median(g, ["nodes"], "busbw_gbs")
        col = COLLECTIVE_COLOR[coll]
        ax.plot(med.nodes, med.value, color=col, lw=1.0, zorder=2)
        for _, row in med.iterrows():
            _pt(ax, row.nodes, row.value, col, row.synthetic, COLLECTIVE_MARKER[coll], ms=3.5)
    for y, text in ((NODE_NET_GBS, "1,600 Gb/s per node"), (NODE_NET_GBS / GPUS_PER_NODE, "its share per GPU")):
        ax.axhline(y, color=MUTED, lw=0.7, ls=(0, (4, 3)), zorder=1)
        ax.text(1, y, text, fontsize=6, color=INK2, va="bottom")
    handles = [_handle(COLLECTIVE_COLOR[k], k.replace("_", "-"), d3[d3.collective == k].synthetic.any(),
                       COLLECTIVE_MARKER[k], "-", ms=3.5) for k in COLLECTIVE_COLOR if k in set(d3.collective)]
    for coll, g in inside.groupby("collective"):
        med = g.groupby("nodes").busbw_gbs.median()
        ax.plot(med.index, med.values, ls="", marker=COLLECTIVE_MARKER[coll], mfc="white", mec=COLLECTIVE_COLOR[coll], mew=0.7, ms=4.5, zorder=3)
    if len(inside):
        handles.append(plt.Line2D([], [], ls="", marker="o", mfc="white", mec=INK2, mew=0.7, ms=4, label="inside one group"))
    ax.legend(handles=handles, loc="upper right", fontsize=6)
    if not d4.empty:  # not a bus bandwidth: median node-pair bandwidth across the cut, per node
        x = d4.nodes.max() * 1.6
        for _, row in d4.iterrows():
            _pt(ax, x, row.busbw_gbs, INK2, row.synthetic, "X", ms=5)
        ax.annotate(f"bisection: pair BW\nper node, {len(d4)} cuts\n{d4.busbw_gbs.median():.0f} GB/s",
                    (x, d4.busbw_gbs.min()), xytext=(0, -6), textcoords="offset points", fontsize=6, color=INK2,
                    ha="center", va="top")
    ax.set_xscale("log", base=2)
    ticks = sorted(d3.nodes.unique())
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{int(x)}, 26" if x == 25 and len(inside) else str(int(x)) for x in ticks])
    ax.minorticks_off()
    ax.set_xlim(0.8, max(ticks) * 2.4)
    ax.set_yscale("log")
    ax.set_xlabel("nodes (8 GPUs each), 1 GiB messages")
    ax.set_ylabel("bus bandwidth (GB/s)")
    style.finish(ax, trim=False)

    ax = axes[1]
    sweep = c[(c.experiment == "D3") & c.collective.isin(["all_reduce", "alltoall"]) & (c.repetition == 1)]
    sizes = sweep.groupby(["collective", "nodes"]).size_bytes.nunique()
    for (coll, nodes), g in sweep.groupby(["collective", "nodes"]):
        if sizes[(coll, nodes)] < 4:
            continue
        g = g.sort_values("size_bytes")
        big = nodes == sweep.nodes.max()
        ax.plot(g.size_bytes, g.busbw_gbs, color=COLLECTIVE_COLOR[coll], lw=1.0, ls="-" if big else (0, (3, 2)))
        ax.annotate(f"{coll.replace('_', '-')}, {int(nodes)} nodes", (g.size_bytes.iloc[-1], g.busbw_gbs.iloc[-1]),
                    xytext=(-2, 3), textcoords="offset points", fontsize=6, color=INK2, ha="right")
    ax.set_xscale("log", base=2)
    ax.set_xticks([2**10, 2**20, 2**30])
    ax.set_xticklabels(["1 KiB", "1 MiB", "1 GiB"])
    ax.minorticks_off()
    ax.set_xlabel("message size")
    ax.set_yscale("log")
    ax.set_ylabel("bus bandwidth (GB/s)")
    style.finish(ax, trim=False)
    style.synthetic_mark(fig, c.synthetic.any())
    fig.tight_layout()
    return fig


# -- E, R. Jobs side by side, background load -------------------------------

def interference(t):
    """Throughput of each 13-node set alone and with the 2 others of its draw, on the same steps (E1; select.e1_changes)."""
    m = e1_changes(t["runs"], t["steps"])
    if m.empty:
        return None
    m = m.sort_values("node_set").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 2.3))
    col = {"alone": SERIES[0], "together": SERIES[1]}
    for i, row in m.iterrows():
        ax.plot([i, i], [row.tok_s_gpu_alone, row.tok_s_gpu_together], color=LIGHT, lw=1.2, zorder=1)
        _pt(ax, i, row.tok_s_gpu_alone, col["alone"], row.synthetic, "o")
        _pt(ax, i, row.tok_s_gpu_together, col["together"], row.synthetic, "s")
        ax.annotate(f"{row.change_pct:+.2f} %".replace("-", "\u2212"), (i, max(row.tok_s_gpu_alone, row.tok_s_gpu_together)), xytext=(0, 5),
                    textcoords="offset points", fontsize=6, color=INK2, ha="center")
    labels = []
    for row in m.itertuples():
        slow = f"\n{row.slow_nodes.replace('trig', '').replace(';', ', ')} in set" if row.slow_nodes else ""
        labels.append(row.node_set.replace("d", "draw ").replace("-s", ", set ") + slow)
    ax.set_xticks(range(len(m)))
    ax.set_xticklabels(labels, fontsize=5.5)
    handles = [_handle(col["alone"], "alone", m.synthetic.any(), "o"), _handle(col["together"], "together", m.synthetic.any(), "s")]
    ax.legend(handles=handles, loc="lower left", ncol=2)
    ax.set_title(f"step time together against alone: mean {m.change_pct.mean():+.2f} %, range {m.change_pct.min():+.2f} to {m.change_pct.max():+.2f} %".replace("-", "\u2212"))
    ax.set_ylim(m[["tok_s_gpu_alone", "tok_s_gpu_together"]].min().min() * 0.96, m[["tok_s_gpu_alone", "tok_s_gpu_together"]].max().max() * 1.03)
    ax.set_ylabel("tokens/s per GPU")
    style.finish(ax, trim=False)
    style.synthetic_mark(fig, m.synthetic.any())
    fig.tight_layout()
    return fig


def reference_drift(t):
    """The same reference run (Qwen, FSDP, 12k tokens per GPU, 4 nodes): fixed nodes from its third run on (R). The first two ran on other
    nodes, one of them slow. Line: when the kernel change reached these nodes."""
    r = _runs(t, "R")
    if r.empty:
        return None
    hours, trend = drift(r)
    r = r.assign(h=hours)
    measured = not r.synthetic.any()
    fixed = r[r.hosts == REF_NODES] if measured else r
    other = r[r.hosts != REF_NODES] if measured else r.iloc[0:0]
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH * 0.8, 2.1))
    med = fixed.tok_s_gpu.median()
    ax.axhspan(0.99 * med, 1.01 * med, color=LIGHT, alpha=0.35, lw=0, zorder=0)
    ax.text(r.h.max(), 1.01 * med, "median \u00b11 %", fontsize=6, color=INK2, ha="right", va="bottom")
    ax.plot(fixed.h, fixed.tok_s_gpu, color=SERIES[0], lw=0.7, zorder=2)
    for _, row in fixed.iterrows():
        _pt(ax, row.h, row.tok_s_gpu, SERIES[0], row.synthetic)
    for _, row in other.iterrows():
        _pt(ax, row.h, row.tok_s_gpu, MUTED, False, "D", ms=3.5)
        nodes = "with " + row.slow_nodes if isinstance(row.slow_nodes, str) and row.slow_nodes else "other nodes"
        ax.annotate(nodes, (row.h, row.tok_s_gpu), xytext=(5, -2), textcoords="offset points", fontsize=6, color=INK2)
    after = r[r.pstate == "after"] if "pstate" in r else r.iloc[0:0]
    if len(after):
        x = after.h.min() - 0.3
        ax.axvline(x, color=MUTED, lw=0.6, ls=(0, (1, 2)))
        ax.annotate("kernel change", (x, 0), xycoords=("data", "axes fraction"), xytext=(-2, 3), textcoords="offset points", fontsize=6, color=INK2, ha="right")
    if not np.isnan(trend):
        ax.set_title(f"fixed nodes: linear trend over the window {trend:+.2f} %".replace("-", "\u2212"))
    ax.set_ylim(0.96 * r.tok_s_gpu.min(), 1.03 * r.tok_s_gpu.max())
    ax.set_xlabel("hours since the first reference run")
    ax.set_ylabel("tokens/s per GPU")
    style.finish(ax, trim=False)
    style.synthetic_mark(fig, r.synthetic.any())
    fig.tight_layout()
    return fig


# -- A-Q. Qwen layouts ---------------------------------------------------------

QWEN_PANELS = (2, 13, 26, 39, 50, 52)  # node counts of the layout panels; hsdp16 runs on 12 nodes, 104 GPUs don't split in 16


def qwen_layouts(t):
    """Throughput per GPU by layout and tokens per GPU, at 2, 13 and the largest node counts (A-Q, with the weak-scaling runs above and below 13
    nodes). Red cross: a configuration that failed, with no successful run. Tokens are valid tokens: a micro-batch of 49,152 slots holds about 15 % padding."""
    r = qwen_base(t["runs"])
    if r.empty:
        return None
    r = r.assign(panel=r.nodes.where(~((r.layout == "hsdp16") & (r.nodes == 12)), 13))
    r = r[r.panel.isin(QWEN_PANELS)]
    scales = sorted(int(n) for n in r.panel.unique())
    m = _median(r, ["panel", "layout", "tokens_per_gpu"])
    failed = failed_runs(t["runs"], "A-Q")
    failed = failed[(failed.ac == "selective") & (failed.grad_reduce == "fp32")] if len(failed) else failed
    layouts = [l for l in QWEN_LAYOUTS if l in {*m.layout, *(failed.layout if len(failed) else ())}]
    fig, axes = plt.subplots(1, len(scales), figsize=(TEXT_WIDTH, 0.9 + 0.24 * len(layouts)), sharey=True, sharex=True, squeeze=False)
    peak, flops = r.peak_flops_per_gpu.iloc[0], r.flops_per_token.iloc[0]
    for ax, n in zip(axes[0], scales):
        d = m[m.panel == n]
        for i, layout in enumerate(layouts):
            g = d[d.layout == layout]
            if len(g):
                ax.plot([g.value.min(), g.value.max()], [i, i], color=LIGHT, lw=0.8, zorder=1)
            for _, row in g.iterrows():
                _pt(ax, row.value, i, TOKENS_COLOR.get(int(row.tokens_per_gpu), MUTED), row.synthetic, ms=4)
            if n == 13 and len(failed) and (failed.layout == layout).any():
                _failed_y(ax, i)
        if len(d):
            best = d.loc[d.value.idxmax()]
            ax.annotate(f"{100 * flops * best.value / peak:.0f} % MFU", (best.value, layouts.index(best.layout)),
                        xytext=(0, 5), textcoords="offset points", fontsize=6, color=INK2, ha="center")
        ax.set_title(f"{n} nodes ({n * GPUS_PER_NODE} GPUs)", fontsize=7)
        ax.set_xlabel("tokens/s per GPU")
        ax.set_xlim(0, None)
        style.finish(ax, trim=False)
    ax = axes[0][0]
    ax.set_yticks(range(len(layouts)))
    ax.set_yticklabels([f"{l} (12 nodes)" if l == "hsdp16" else l for l in layouts])
    ax.invert_yaxis()
    handles = [_handle(c, f"{k // 1024}k", m[m.tokens_per_gpu == k].synthetic.any())
               for k, c in TOKENS_COLOR.items() if k in set(m.tokens_per_gpu)]
    if len(failed):
        handles.append(plt.Line2D([], [], ls="", marker="x", color=CRITICAL, label="failed"))
    fig.legend(handles=handles, title="tokens per GPU per step", loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 1.0 + 0.02), fontsize=6, title_fontsize=6)
    style.synthetic_mark(fig, m.synthetic.any())
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    return fig


def _key(row) -> tuple:
    return tuple(row[k] for k in REF)


def qwen_levers(t):
    """Throughput change of one lever at a time, at 13 nodes (A-Q): each bar compares two runs that differ only in that lever. Selective recompute,
    FP32 reduction, and BF16 unless a bar says otherwise."""
    r = _runs(t, "A-Q")
    if r.empty:
        return None
    r = r[(r.nodes == 13) | ((r.nodes == 12) & (r.layout == "hsdp16"))]
    m = r.groupby(list(REF), dropna=False).agg(value=("tok_s_gpu", "median"), synthetic=("synthetic", "any"))
    def get(layout="fsdp", tokens=12288, reduce="fp32", ac="selective"):
        key = (layout, tokens, reduce, ac, "bf16")
        return m.loc[key] if key in m.index else None
    pairs = [("tokens per GPU", "24k against 12k (fsdp)", get(tokens=24576), get())]
    pairs += [("tokens per GPU", "48k against 24k (fsdp, full recompute: 15 % padding)", get(tokens=49152, ac="full"), get(tokens=24576, ac="full"))]
    for layout in QWEN_LAYOUTS[1:]:
        pairs.append(("layout, 12k", f"{layout} against fsdp", get(layout), get()))
    for layout in QWEN_LAYOUTS[1:]:
        pairs.append(("layout, 24k", f"{layout} against fsdp", get(layout, 24576), get(tokens=24576)))
    for layout in ("fsdp", "hsdp8"):
        pairs.append(("gradient reduction, 24k", f"BF16 against FP32 ({layout})", get(layout, 24576, reduce="bf16"), get(layout, 24576)))
    for layout in ("fsdp", "hsdp8"):
        pairs.append(("recompute, 24k", f"full against selective ({layout})", get(layout, 24576, ac="full"), get(layout, 24576)))
    rows = [{"group": g, "label": label, "pct": 100 * (a.value / b.value - 1), "synthetic": a.synthetic or b.synthetic}
            for g, label, a, b in pairs if a is not None and b is not None]
    if not rows:
        return None
    b = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH * 0.92, 0.6 + 0.17 * len(b)))
    y = np.arange(len(b))
    for yi, row in zip(y, b.itertuples()):
        col = SERIES[list(dict.fromkeys(b.group)).index(row.group) % len(SERIES)]
        ax.barh(yi, row.pct, height=0.62, color="white" if row.synthetic else col, edgecolor=col, lw=0.8)
        ax.text(row.pct, yi, f" {row.pct:+.0f} %".replace("-", "\u2212") if row.pct >= 0 else
                f"{row.pct:+.0f} % ".replace("-", "\u2212"), va="center", ha="left" if row.pct >= 0 else "right",
                fontsize=6, color=INK2)
    ax.axvline(0, color=INK2, lw=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.group}: {r.label}" for r in b.itertuples()], fontsize=6)
    ax.invert_yaxis()
    lim = max(abs(b.pct).max() * 1.2, 5)
    ax.set_xlim(min(b.pct.min() * 1.3, -2), max(b.pct.max() * 1.5, 5))
    ax.set_xlabel("change in valid tokens/s per GPU, 13 nodes (%)")
    style.finish(ax, trim=False)
    style.synthetic_mark(fig, b.synthetic.any(), "includes synthetic data (hollow bars)")
    fig.tight_layout()
    return fig


def step_decomposition(t):
    """Step = compute (est.: the 1-node fsdp step, same tokens per GPU) + exposed communication
    (and pipeline bubble), by layout and scale, 24k tokens per GPU (A-Q)."""
    r = qwen_base(t["runs"])
    if r.empty:
        return None
    r = r[r.tokens_per_gpu == 24576]
    one = r[(r.nodes == 1) & (r.layout == "fsdp")]
    if one.empty:
        return None
    compute = one.step_s.median()
    m = _median(r[r.nodes > 1], ["layout", "nodes"], "step_s")
    m["o"] = m.layout.map({l: i for i, l in enumerate(QWEN_LAYOUTS)})
    m = m.sort_values(["nodes", "o"]).reset_index(drop=True)
    m["label"] = m.layout + ", " + m.nodes.astype(int).astype(str) + " nodes"
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH * 0.85, 0.5 + 0.16 * len(m)))
    y = np.arange(len(m))
    for yi, row in zip(y, m.itertuples()):
        col = SERIES[0]
        ax.barh(yi, compute, height=0.62, color="white", edgecolor=col, hatch="////", lw=0.6)
        ax.barh(yi, row.value - compute, left=compute, height=0.62, color=col, alpha=0.18 if row.synthetic else 0.6,
                edgecolor=col, lw=0.6)
        ax.text(row.value, yi, f" {row.value:.1f} s ({100 * (row.value - compute) / row.value:.0f} %)", va="center",
                fontsize=6, color=INK2, style="italic" if row.synthetic else "normal")
    ax.set_yticks(y)
    ax.set_yticklabels(m.label)
    ax.invert_yaxis()
    ax.set_xlim(0, m.value.max() * 1.25)
    ax.set_xlabel("step time (s): hatched = compute (est., 1-node fsdp), shaded = exposed\n"
                  "communication + pipeline bubble (est.); 24k tokens per GPU")
    style.finish(ax, trim=False)
    style.synthetic_mark(fig, m.synthetic.any() or one.synthetic.any(), "includes synthetic data (italic labels)")
    fig.tight_layout()
    return fig


# -- A-G. GLM-5 layouts ------------------------------------------------------

def _glm_label(row) -> str:
    extra = []
    if row.ac != "full":
        extra.append(f"{row.ac} recompute")
    if row.mb_seqs and row.mb_seqs > 1:
        extra.append(f"micro-batch {int(row.mb_seqs)}")
    copies = row.copies if pd.notna(row.copies) else row.gpus / (row.pp * row.ep)
    per_copy = row.global_batch_tokens / copies / row.seq_len
    if per_copy < GLM_SEQS_PER_COPY / 2:
        extra.append(f"batch {int(per_copy)}")
    return row.layout + (f" ({', '.join(extra)})" if extra else "")


def glm_layouts(t):
    """GLM-5: throughput per GPU and expert all-to-all time per step, by layout, at one model copy
    and at the full machine (A-G)."""
    r = _runs(t, "A-G")
    if r.empty:
        return None
    r = r.assign(label=r.apply(_glm_label, axis=1))
    failed = failed_runs(t["runs"], "A-G")
    failed = failed.assign(label=failed.apply(_glm_label, axis=1)) if len(failed) else failed
    # 10-13 nodes, 20-30, and 36 up: a layout comparison holds at a node count.
    r["scale"] = np.select([r.nodes < 20, r.nodes < 36], ["small", "mid"], "large")
    m = _median(r, ["label", "layout", "nodes", "copies", "scale"])
    ep = t["ep_groups"]
    a2a = (ep.merge(r[["run_id", "label", "nodes"]], on="run_id").groupby(["label", "nodes"]).a2a_s.median()
           if not ep.empty else pd.Series(dtype=float))
    order = {l: i for i, l in enumerate(GLM_LAYOUTS)}
    labels = sorted({*m.label, *(failed.label if len(failed) else ())},
                    key=lambda s: (order.get(s.split(" ")[0], len(order)), s))
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 0.7 + 0.22 * len(labels)), sharey=True)
    for i, lab in enumerate(labels):
        for _, row in m[m.label == lab].iterrows():
            col = SCALE_COLOR[row.scale]
            _pt(axes[0], row.value, i, col, row.synthetic)
            if (lab, row.nodes) in a2a.index:
                _pt(axes[1], a2a[(lab, row.nodes)], i, col, row.synthetic)
        if len(failed) and (failed.label == lab).any():
            _failed_y(axes[0], i)
    axes[0].set_yticks(range(len(labels)))
    axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("tokens/s per GPU")
    axes[1].set_xlabel("expert all-to-all (s per step, median EP group)")
    axes[0].set_xlim(0, m.value.max() * 1.15)
    axes[1].set_xlim(0, None)
    for ax in axes:
        style.finish(ax, trim=False)
    handles = [_handle(SCALE_COLOR[s], f"{'-'.join(str(int(n)) for n in sorted(set(m[m.scale == s].nodes)))} nodes",
                       m[m.scale == s].synthetic.any()) for s in ("small", "mid", "large") if (m.scale == s).any()]
    if len(failed):
        handles.append(plt.Line2D([], [], ls="", marker="x", color=CRITICAL, label="failed, no successful run"))
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=6)
    style.synthetic_mark(fig, m.synthetic.any())
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return fig


# -- B. Scaling --------------------------------------------------------------

def _gpu_axis(ax, values) -> None:
    ax.set_xscale("log", base=2)
    ticks = sorted(set(int(v) for v in values))
    ticks = [v for i, v in enumerate(ticks) if i == len(ticks) - 1 or ticks[i + 1] / v > 1.1]  # 200 and 208 GPUs: one label
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(v) for v in ticks], fontsize=6)
    ax.minorticks_off()


def qwen_scaling(t):
    """Weak scaling (24k tokens per GPU) of the best layout and fsdp, and strong scaling (fixed
    global batch) of the best layout (B-Q)."""
    w, s = _runs(t, "B-Q-weak"), _runs(t, "B-Q-strong")
    if w.empty and s.empty:
        return None
    slow = w[w.slow_nodes.fillna("") != ""] if "slow_nodes" in w else w.iloc[0:0]  # labelled, out of the line
    w = w.drop(slow.index)
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.3), sharey=True)
    ax = axes[0]
    colors = {}
    for layout, g in w.groupby("layout"):
        colors[layout] = SERIES[0] if layout != "fsdp" else SERIES[1]
        med = _median(g, ["gpus"])
        ax.plot(med.gpus, med.value, color=colors[layout], lw=1.0)
        for _, row in med.iterrows():
            _pt(ax, row.gpus, row.value, colors[layout], row.synthetic)
        eff = 100 * med.value.iloc[-1] / med.value.iloc[0]
        ax.annotate(f"{layout}: {eff:.0f} % of {int(med.gpus.iloc[0])} GPUs", (med.gpus.iloc[-1], med.value.iloc[-1]),
                    xytext=(-4, 6 if layout != "fsdp" else -10), textcoords="offset points", fontsize=6,
                    color=INK2, ha="right")
    for _, row in slow.iterrows():
        _pt(ax, row.gpus, row.tok_s_gpu, colors.get(row.layout, MUTED), False, "D", ms=4.5)
        ax.annotate(f"{int(row.nodes)} nodes with\n{row.slow_nodes.count(';') + 1} slow nodes", (row.gpus, row.tok_s_gpu), xytext=(0, -7), textcoords="offset points", fontsize=6,
                    color=CRITICAL, ha="center", va="top")
    _gpu_axis(ax, w.gpus)
    ax.set_title("weak: 24k tokens per GPU")
    ax.set_xlabel("GPUs")
    ax.set_ylabel("tokens/s per GPU")
    ax = axes[1]
    if not s.empty:
        med = _median(s, ["gpus", "layout"])
        col = SERIES[0]
        ax.plot(med.gpus, med.value, color=col, lw=1.0)
        for _, row in med.iterrows():
            _pt(ax, row.gpus, row.value, col, row.synthetic)
            eff = 100 * row.value / med.value.iloc[0]
            ax.annotate(f"{eff:.0f} %", (row.gpus, row.value), xytext=(0, -10), textcoords="offset points",
                        fontsize=6, color=INK2, ha="center")
        batch = s.global_batch_tokens.iloc[0] / 1e6
        ax.set_title(f"strong: {batch:.1f}M tokens per step, {med.layout.iloc[0]}")
        failed = failed_runs(t["runs"], "B-Q-strong")
        for gpus in sorted(set(failed.gpus)) if len(failed) else ():
            _failed_x(ax, gpus, "hsdp8: memory")
        _gpu_axis(ax, [*s.gpus, *(failed.gpus if len(failed) else ())])
    ax.set_xlabel("GPUs (label: efficiency vs fewest GPUs)")
    for ax in axes:
        ax.set_ylim(0, None)
        style.finish(ax, trim=False)
    style.synthetic_mark(fig, pd.concat([w, s]).synthetic.any())
    fig.tight_layout()
    return fig


def glm_scaling(t):
    """GLM-5 model copies: throughput per GPU vs copies, 32 micro-batches per data-parallel rank (weak:
    256 sequences per copy) or 768 sequences in total (strong: the bubble grows), with the 1F1B
    bubble of each point (B-G, counting the runs that serve two experiments)."""
    r = _runs(t, "B-G-weak", "B-G-strong")
    if r.empty:
        return None
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH * 0.7, 2.2))
    r = r[(r.layout == "pp13-ep8") & (r.ac == "full") & (r.precision == "bf16")]
    failed = failed_runs(t["runs"], "B-G-weak", "B-G-strong")
    failed = failed[(failed.layout == "pp13-ep8") & (failed.ac == "full")] if len(failed) else failed
    handles = []
    for exp, col, text, above in (("B-G-weak", SERIES[0], f"weak: {GLM_SEQS_PER_COPY} sequences per copy", False),
                                  ("B-G-strong", SERIES[1], f"strong: {GLM_STRONG_SEQS} sequences in total", True)):
        g = r[r.experiment == exp]
        med = _median(g, ["copies"])
        if med.empty:
            continue
        ax.plot(med.copies, med.value, color=col, lw=1.0)
        for n in sorted(set(failed[failed.experiment == exp].nodes) - set(g.nodes)):
            _failed_x(ax, n / GLM_COPY_NODES, "failed" if exp == "B-G-weak" else None)
        bubble = ((g.pp - 1) / (g.microbatches + g.pp - 1)).groupby(g.copies).median()
        for _, row in med.iterrows():
            _pt(ax, row.copies, row.value, col, row.synthetic, "o" if exp == "B-G-weak" else "s")
            ax.annotate(f"{bubble[row.copies]:.0%}", (row.copies, row.value), xytext=(0, 5 if above else -9),
                        textcoords="offset points", fontsize=6, color=INK2, ha="center")
        eff = 100 * med.value.iloc[-1] / med.value.iloc[0]
        handles.append(_handle(col, f"{text}: {eff:.0f} % of 1 copy at {int(med.copies.iloc[-1])}", med.synthetic.any(),
                               "o" if exp == "B-G-weak" else "s", "-"))
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 0.16), fontsize=6)  # room for a failure's cross
    ticks = sorted({*r.copies, *(failed.nodes / GLM_COPY_NODES if len(failed) else ())})
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{int(c)}\n({int(c) * 13} n.)" for c in ticks], fontsize=6)
    ax.set_xlim(0.7, r.copies.max() + 0.3)
    ax.set_ylim(0, None)
    ax.set_xlabel("model copies (pp13-ep8, 13 nodes each); labels: pipeline bubble (1F1B)")
    ax.set_ylabel("tokens/s per GPU")
    style.finish(ax, trim=False)
    style.synthetic_mark(fig, r.synthetic.any())
    fig.tight_layout()
    return fig


# -- C. Precision ------------------------------------------------------------

def precision_speed(t):
    """Speedup of each recipe over BF16, at small scale and on the whole machine, per workload
    (C-speed). Recipes differ by framework: TorchTitan offers MXFP8 only."""
    r = _runs(t, "C-speed")
    if r.empty:
        return None
    models = [m for m in (QWEN, GLM) if m in set(r.model)]
    fig, axes = plt.subplots(1, len(models), figsize=(TEXT_WIDTH, 1.9), sharex=True, squeeze=False)
    syn = False
    for ax, model in zip(axes[0], models):
        d = _median(r[r.model == model], ["nodes", "precision"])
        base = d[d.precision == "bf16"].set_index("nodes")
        d = d[d.precision != "bf16"].assign(speedup=lambda x: x.value / x.nodes.map(base.value),
                                            synthetic=lambda x: x.synthetic | x.nodes.map(base.synthetic).fillna(False))
        recipes = [p for p in RECIPE_COLOR if p in set(d.precision)]
        small, large = d.nodes.min(), d.nodes.max()
        for i, rec in enumerate(recipes):
            g = d[d.precision == rec].set_index("nodes")
            ax.plot(g.speedup, [i] * len(g), color=LIGHT, lw=1.0, zorder=1)
            for n, row in g.iterrows():
                if n not in (small, large):  # sizes in between: shown, not labelled
                    _pt(ax, row.speedup, i, MUTED, row.synthetic, ms=3)
                    continue
                _pt(ax, row.speedup, i, SCALE_COLOR["small" if n == small else "large"], row.synthetic)
                above = n == small
                ax.annotate(f"×{row.speedup:.2f}", (row.speedup, i), xytext=(0, 5 if above else -6),
                            textcoords="offset points", fontsize=6, color=INK2, ha="center",
                            va="bottom" if above else "top")
        syn |= bool(d.synthetic.any())
        ax.axvline(1, color=INK2, lw=0.6)
        ax.set_yticks(range(len(recipes)))
        ax.set_yticklabels(recipes)
        ax.set_ylim(len(recipes) - 0.4, -0.7)
        ax.set_title(f"{WORKLOAD_LABEL[model]}: {int(small)} vs {int(large)} nodes" if small != large else f"{WORKLOAD_LABEL[model]}: {int(small)} nodes")
        ax.set_xlabel("speedup over BF16 (tokens/s)")
        style.finish(ax, trim=False)
    handles = [_handle(SCALE_COLOR["small"], "smallest", syn), _handle(MUTED, "in between", syn), _handle(SCALE_COLOR["large"], "largest", syn)]
    axes[0][0].legend(handles=handles, loc="lower left")
    style.synthetic_mark(fig, syn)
    fig.tight_layout()
    return fig


GLM_LR_WARMUP = 50  # --lr-warmup-iters of the C-loss entries (benchmarks/megatron/dev.yaml)


def precision_loss(t):
    """Loss difference between runs over 500 steps at 13 nodes (C-loss), by step: each FP8 recipe minus BF16 at the same seed, with the same differences
    between BF16 repeats as the noise. Joined on the loss events' own step, across runs; never with the rate events' batch_id."""
    g = loss_gaps(t["runs"], t["loss"])
    if g.empty:
        return None
    models = [m for m in (QWEN, GLM) if m in set(g.model)]
    curve = {k: d.dropna(subset=["step"]).groupby("step").loss.mean() for k, d in t["loss"][t["loss"].run_id.isin(set(g.run_a) | set(g.run_b))].groupby("run_id")}
    smooth = lambda row: (curve[row.run_a] - curve[row.run_b]).dropna().rolling(25, center=True, min_periods=5).mean()  # noqa: E731
    fig, axes = plt.subplots(1, len(models), figsize=(TEXT_WIDTH, 3.2), sharey=True, squeeze=False)
    seed_color = {}
    for ax, model in zip(axes[0], models):
        d = g[g.model == model]
        for kind, color, alpha, label in (("other seed", LIGHT, 0.45, "other seed"), ("same seed", INK2, 0.3, "same seed")):
            pairs = d[d.kind == kind]
            if len(pairs):
                band = pd.concat([smooth(x) for x in pairs.itertuples()], axis=1)
                ax.fill_between(band.index, band.min(axis=1).clip(upper=0), band.max(axis=1).clip(lower=0), color=color, alpha=alpha, lw=0, zorder=0,
                                label=f"{label}: |gap| up to {pairs.last100.abs().max():.2f}")
        ax.axhline(0, color=INK2, lw=0.6)
        for row in d[d.kind == "recipe"].itertuples():
            color = seed_color.setdefault((row.recipe, row.seed_a), SERIES[len(seed_color) % len(SERIES)])  # GLM-5 has several recipes at one seed
            ax.plot(*zip(*smooth(row).items()), color=color, lw=1.1 if row.same_nodes else 0.9, ls="-" if row.same_nodes else (0, (4, 2)),
                    label=f"{row.recipe}, seed {row.seed_a}: {row.last100:+.2f}" + ("" if row.same_nodes else " (other nodes)"))
        if model == GLM:  # the lr starts at 0: steps 1-2 see the same weights, then an early unstable phase
            ax.axvline(GLM_LR_WARMUP, color=MUTED, lw=0.6, ls=(0, (1, 2)))
            ax.annotate("lr warm-up ends", (GLM_LR_WARMUP, 0), xycoords=("data", "axes fraction"), xytext=(2, 2),
                        textcoords="offset points", fontsize=6, color=INK2, va="bottom")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1 if model == GLM else 2, fontsize=5.5, frameon=False)  # GLM-5's labels are long enough to run into Qwen's
        ax.set_xlim(0, 500)
        ax.set_title(WORKLOAD_LABEL[model] + ("" if (d.kind != "recipe").any() else ": no BF16 repeats yet"))
        ax.set_xlabel("step (legend: mean of steps 401-500)")
        style.finish(ax, trim=False)
    axes[0][0].set_ylim(-0.7, 0.55)  # the settled part; the early unstable phase runs off the axis
    axes[0][0].set_ylabel("loss minus BF16 loss\n(25-step mean)")
    style.synthetic_mark(fig, g.synthetic.any(), "includes synthetic data")
    fig.tight_layout()
    return fig


# -- F. Operations -----------------------------------------------------------

def _thin_ticks(nodes, ratio=1.35):
    """Node-count ticks on a log axis, dropping a tick too close to the next larger one to read (39 next to 50)."""
    ticks = sorted({int(n) for n in nodes})
    keep = []
    for n in reversed(ticks):
        if not keep or keep[-1] / n >= ratio:
            keep.append(n)
    return sorted(keep)


def checkpoint(t):
    """Checkpoint save and load: total time and time training is blocked vs nodes, sync vs async,
    per workload and content; filesystem throughput; peak GPU memory of loads and saves (F1)."""
    c = checkpoints(t["checkpoints"], t["runs"])
    if c.empty:
        return None
    failed = t.get("failed_ops", pd.DataFrame())
    cases = sorted(set(zip(c.model, c.content)), key=lambda mk: (mk[0] != QWEN, mk[0], mk[1] != "weights"))
    case_color = {mk: SERIES[4 + i % 4] for i, mk in enumerate(cases)}
    series = [("save", "sync", "total_s", "save, sync", SERIES[0], "o", "-"),
              ("save", "async", "total_s", "save, async: total", SERIES[1], "s", "-"),
              ("save", "async", "blocked_s", "save, async: blocked", SERIES[1], "s", (0, (3, 2))),
              ("save", "async_pinned", "blocked_s", "save, async pinned: blocked", SERIES[2], "^", (0, (3, 2))),
              ("load", "sync", "total_s", "load", SERIES[3], "D", "-")]
    has_mem = c.peak_allocated_gib.notna().any()
    n_panels = len(cases) + 1 + bool(has_mem)
    n_rows = 1 + (n_panels > 3)
    n_cols = (n_panels + n_rows - 1) // n_rows
    fig, grid = plt.subplots(n_rows, n_cols, figsize=(TEXT_WIDTH, 2.4 * n_rows + 0.4), squeeze=False)
    axes = list(grid.ravel())
    for ax in axes[n_panels:]:
        ax.set_visible(False)
    for ax, (model, content) in zip(axes, cases):
        d = c[(c.model == model) & (c.content == content)]
        nodes_here = set(d.nodes)
        if len(failed):
            here = failed[(failed.model == model) & (failed.content == content)]
            nodes_here |= set(here.nodes)
        for op, mode, col, label, color, marker, ls in series:
            g = d[(d.op == op) & (d["mode"] == mode)]
            if mode == "async_pinned":
                g = pinned_blocked(g)  # the first save also allocates the buffers: an upper bound
            if g.empty:
                continue
            med = g.groupby("nodes")[col].median()
            ax.plot(med.index, med.values, color=color, lw=0.9, ls=ls)
            for _, row in g.iterrows():
                _pt(ax, row.nodes, row[col], color, row.synthetic, marker, ms=3)
        if len(failed):  # no time to draw: a cross where the operation should be
            for (n, op), _ in here.groupby(["nodes", "op"]):
                _failed_x(ax, n, f"{op} failed")
        size = d.bytes.iloc[0] / 1e12
        ax.set_title(f"{WORKLOAD_LABEL[model]}, {content} ({size:.2g} TB)", fontsize=7)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_xticks(_thin_ticks(nodes_here))
        ax.set_xticklabels([str(int(n)) for n in _thin_ticks(nodes_here)])
        ax.minorticks_off()
        ax.set_xlabel("nodes")
        style.finish(ax, trim=False)
    for ax in grid[:, 0]:
        if ax.get_visible():
            ax.set_ylabel("time (s)")
    ax = axes[len(cases)]
    for (model, content), color in case_color.items():
        for op, marker, ls in (("save", "o", "-"), ("load", "D", (0, (3, 2)))):
            g = c[(c.model == model) & (c.content == content) & (c.op == op) & (c["mode"] == "sync")]
            if g.empty:
                continue
            for _, row in g.iterrows():
                _pt(ax, row.nodes, row.gb_s, color, row.synthetic, marker, ms=3)
            med = g.groupby("nodes").gb_s.median()
            ax.plot(med.index, med.values, color=color, lw=0.9, ls=ls)
    ax.set_xscale("log")
    ax.set_xticks(_thin_ticks(c.nodes.unique()))
    ax.set_xticklabels([str(int(n)) for n in _thin_ticks(c.nodes.unique())])
    ax.minorticks_off()
    ax.set_ylim(0, None)
    ax.set_title("filesystem: write (solid), read (dashed)", fontsize=7)
    ax.set_xlabel("nodes")
    ax.set_ylabel("GB/s")
    style.finish(ax, trim=False)
    if has_mem:
        ax = axes[len(cases) + 1]
        ax.axhline(GPU_MEMORY_GIB, color=CRITICAL, lw=0.7, ls=(0, (1, 2)))
        ax.annotate("180 GB", (0, GPU_MEMORY_GIB), xycoords=("axes fraction", "data"), xytext=(2, 2),
                    textcoords="offset points", fontsize=6, color=CRITICAL)
        for (model, content), color in case_color.items():
            d = c[(c.model == model) & (c.content == content)]
            for op, marker in (("load", "D"), ("save", "o")):
                for _, row in d[d.op == op].dropna(subset=["peak_allocated_gib"]).iterrows():
                    _pt(ax, row.nodes, row.peak_allocated_gib, color, row.synthetic, marker, ms=3)
        ax.set_xscale("log")
        ticks = _thin_ticks(c.dropna(subset=["peak_allocated_gib"]).nodes.unique())
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(int(n)) for n in ticks])
        ax.minorticks_off()
        ax.set_ylim(0, None)
        ax.set_title("peak GPU memory: load (diamond), save (circle)", fontsize=7)
        ax.set_xlabel("nodes")
        ax.set_ylabel("GiB per GPU (max over ranks)")
        style.finish(ax, trim=False)
    syn = c.synthetic.any()
    handles = [_handle(color, label, syn, marker, ls, ms=3)
               for op, mode, col, label, color, marker, ls in series if len(c[(c.op == op) & (c["mode"] == mode)])]
    handles += [_handle(color, f"GB/s: {WORKLOAD_LABEL[m]}, {k}", syn, "o", "-", ms=3)
                for (m, k), color in case_color.items()]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=6, bbox_to_anchor=(0.5, 0.0))
    style.synthetic_mark(fig, c.synthetic.any())
    fig.tight_layout(rect=(0, 0.11 if n_rows > 1 else 0.16, 1, 0.97))
    return fig


def _long_run(t, model=QWEN):
    r = _runs(t, "F2")
    r = r[r.model == model] if len(r) else r
    mon = t["gpu_monitor"]
    if r.empty or mon.empty:
        return None, None
    run = r.sort_values("measured_steps").iloc[-1]
    return run, mon[mon.source == run.run_name]


def stability(t, model=QWEN):
    """Throughput per step over the long run, with SM clock and power of every GPU (F2)."""
    run, mon = _long_run(t, model)
    if run is None or mon.empty:
        return None
    s = t["steps"][t["steps"].run_id == run.run_id].sort_values("t_s")
    # Megatron's step times count from its warm-up step, TorchTitan's from the first measured step: from the start of the
    # first measured step for both, as the telemetry below (measure_start)
    s = s.assign(t_s=s.t_s - (s.t_s.iloc[0] - s.step_s.iloc[0])) if len(s) else s
    fig, axes = plt.subplots(3, 1, figsize=(TEXT_WIDTH, 4.0), sharex=True, gridspec_kw={"height_ratios": [1.3, 1, 1]})
    ax = axes[0]
    ax.plot(s.t_s / 60, s.tok_s_gpu, color=LIGHT, lw=0.5)
    roll = 15 if len(s) > 200 else 5  # GLM-5: about one step a minute
    ax.plot(s.t_s / 60, s.tok_s_gpu.rolling(roll, center=True).median(), color=SERIES[0], lw=1.1,
            ls=(0, (4, 2)) if run.synthetic else "-")
    ax.set_ylabel("tokens/s per GPU")
    ax.set_title(f"{run.layout}, {int(run.nodes)} nodes: per step (gray), {roll}-step median")
    ref = _runs(t, "B-Q-weak", "A-G", "B-G-weak")  # the same layout in its short run, same nodes if it ran, else the most, for scale
    ref = ref[(ref.model == run.model) & (ref.layout == run.layout) & (ref.tokens_per_gpu == run.tokens_per_gpu)]
    if len(ref) and not run.synthetic:
        same = ref[ref.nodes == run.nodes]
        r0 = (same if len(same) else ref).sort_values(["nodes", "started"]).iloc[-1]
        ax.axhline(r0.tok_s_gpu, color=INK2, lw=0.7, ls=(0, (3, 2)))
        ax.set_title(f"{run.layout}, {int(run.nodes)} nodes: per step (gray), {roll}-step median; dashed, its short run at {int(r0.nodes)} nodes")
    t0 = run.measure_start if pd.notna(run.get("measure_start")) else mon.time.min()
    t1 = run.measure_end if pd.notna(run.get("measure_end")) else mon.time.max()
    for ax, metric, label in ((axes[1], "sm_clock_mhz", "SM clock (MHz)"), (axes[2], "power_w", "GPU power (W)")):
        m = mon[(mon.metric == metric) & mon.time.between(t0, t1)]
        if m.empty:
            continue
        # 1 s samples of hundreds of GPUs: each GPU's mean over each minute of the measurement
        m = m.assign(minute=((m.time - t0) // 60).astype(int), gid=m.host + " GPU " + m.gpu.astype("Int64").astype(str))
        m = m[m.minute < int((t1 - t0) // 60)]  # whole minutes only
        per = m.groupby(["gid", "minute"]).value.mean().reset_index()
        band = per.groupby("minute").value.agg(["min", "max", "median"])
        ax.fill_between(band.index + 0.5, band["min"], band["max"], color=LIGHT, alpha=0.6, lw=0)
        ax.plot(band.index + 0.5, band["median"], color=INK2, lw=0.8)
        worst = per.groupby("gid").value.mean().idxmin()  # lowest mean over the run
        w = per[per.gid == worst]
        ax.plot(w.minute + 0.5, w.value, color=CRITICAL, lw=0.9)
        ax.annotate(worst, (w.minute.iloc[-1] + 0.5, w.value.iloc[-1]), xytext=(4, 0),
                    textcoords="offset points", fontsize=6, color=CRITICAL, va="center")
        ax.set_ylabel(label)
        style.finish(ax, trim=False)
    fig.legend(handles=[plt.Line2D([], [], color=LIGHT, lw=5, alpha=0.6, label=f"range, all {int(run.gpus)} GPUs"),
                            plt.Line2D([], [], color=INK2, lw=0.8, label="median GPU"),
                            plt.Line2D([], [], color=CRITICAL, lw=0.9, label="GPU with the lowest mean (named)")],
               loc="lower center", ncol=3, fontsize=6, title="SM clock and power, per GPU and minute:", title_fontsize=6)
    style.finish(axes[0], trim=False)
    axes[-1].set_xlabel("minutes since the start of the measurement")
    style.synthetic_mark(fig, bool(run.synthetic) or mon.synthetic.any())
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    return fig


def stragglers(t):
    """Are some nodes slower? Per node, over the Qwen long run (F2), or before it has run the largest Qwen FSDP run, from the node telemetry: GPU activity, CPU samples below 500 MHz, GPU power.
    The slow nodes and the node without clock drops are named."""
    n = t.get("node_telemetry", pd.DataFrame())
    r = _runs(t)
    if n.empty or r.empty:
        return None
    long = r[(r.model == QWEN) & (r.experiment == "F2") & r.run_id.isin(n.run_id)]  # the long run once it has run (DESIGN.md)
    q = long if len(long) else r[(r.model == QWEN) & (r.layout == "fsdp") & r.run_id.isin(n.run_id) & (r.tokens_per_gpu == 24576) & (r.ac == "selective") & (r.grad_reduce == "fp32")]
    if q.empty:
        return None
    run = q.sort_values(["nodes", "started"]).iloc[-1]
    d = n[(n.run_id == run.run_id) & n.host.isin(POS)].copy()
    d["x"] = d.host.map(POS)
    panels = [("gpu_util", "GPU activity\n(mean, %)"), ("cpu_low_share", "CPU samples\nbelow 500 MHz (%)"), ("node_gpu_kw_mean", "GPU power\nper node (kW)")]
    panels = [p for p in panels if p[0] in d and d[p[0]].notna().any()]
    fig, axes = plt.subplots(len(panels), 1, figsize=(TEXT_WIDTH, 1.1 * len(panels) + 0.5), sharex=True, squeeze=False)
    axes = axes[:, 0]
    named = [h for h in (*sorted(SLOW_NODES), "trig0114") if h in set(d.host)]
    for ax, (col, label) in zip(axes, panels):
        v = d[col] * (100 if col in ("cpu_low_share", "gpu_util") else 1)
        _pt(ax, d.x, v, INK2, False, ms=3)
        for k, host in enumerate(named):
            row = d.index[d.host == host][0]
            ax.plot(d.x[row], v[row], "o", color=CRITICAL, ms=3.5, zorder=4)
            ax.annotate(host.replace("trig", ""), (d.x[row], v[row]), xytext=(4, -2 if k % 2 else 2), textcoords="offset points", fontsize=6, color=CRITICAL)
        ax.set_ylabel(label)
        style.finish(ax, trim=False)
    _subnet_axis(axes)
    what = "the long run (F2)" if len(long) else "one run (F2 not run)"
    axes[-1].set_xlabel(f"node, grouped by subnet: {run.layout}, 24k tokens per GPU, {int(run.nodes)} nodes, {what}")
    fig.tight_layout()
    return fig


def startup(t):
    """Time from process start to the first training step vs nodes, both workloads (F3)."""
    r = _runs(t)
    if r.empty:
        return None
    r = r.dropna(subset=["first_step_total_s"])
    r = r[r.experiment.isin(["A-Q", "A-G", "B-Q-weak", "B-G-weak", "C-speed"])]
    if r.empty:
        return None
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH * 0.7, 2.2))
    for model, g in r.groupby("model"):
        col = WORKLOAD_COLOR.get(model, MUTED)
        med = g.groupby("nodes").first_step_total_s.median() / 60
        ax.plot(med.index, med.values, color=col, lw=1.0)
        for syn, part in g.groupby("synthetic"):
            _pt(ax, part.nodes, part.first_step_total_s / 60, col, syn, ms=3)
        ax.annotate(f"{WORKLOAD_LABEL.get(model, model)}: {med.iloc[-1]:.0f} min at {int(med.index[-1])} nodes",
                    (med.index[-1], med.iloc[-1]), xytext=(-4, 6), textcoords="offset points", fontsize=6,
                    color=INK2, ha="right")
    ax.set_xscale("log", base=2)
    ticks = [n for n in (1, 2, 4, 8, 13, 26, r.nodes.max()) if n <= r.nodes.max()]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(int(n)) for n in ticks], fontsize=6)
    ax.minorticks_off()
    ax.set_ylim(0, None)
    ax.set_xlabel("nodes")
    ax.set_ylabel("process start to first step (min)")
    style.finish(ax, trim=False)
    style.synthetic_mark(fig, r.synthetic.any())
    fig.tight_layout()
    return fig


def energy(t):
    """Tokens per kWh of GPU energy (nodemon power over the measurement), by Qwen layout at 13
    nodes, by precision on the whole machine, and by scale (F4)."""
    e = with_energy(_runs(t), t["run_power"])
    if e.empty:
        return None
    base = e[e.run_id.isin(qwen_base(t["runs"]).run_id)]
    speed = e[e.experiment == "C-speed"]
    panels = [
        ("Qwen layouts, 13 nodes, 24k", base[(base.nodes == 13) & (base.tokens_per_gpu == 24576)], "layout"),
        ("precision, largest run", speed[speed.nodes == speed.groupby("model").nodes.transform("max")], "prec"),
        ("Qwen scale (B-Q-weak)", e[(e.experiment == "B-Q-weak") & (e.slow_nodes.fillna("") == "")], "scale"),
    ]
    panels = [x for x in panels if not x[1].empty]
    if not panels:
        return None
    fig, axes = plt.subplots(1, len(panels), figsize=(TEXT_WIDTH, 2.2), squeeze=False)
    for ax, (title, d, kind) in zip(axes[0], panels):
        if kind == "scale":
            for layout, g in d.groupby("layout"):
                col = SERIES[1] if layout == "fsdp" else SERIES[0]
                med = g.groupby("nodes").mtok_kwh.median()
                ax.plot(med.index, med.values, color=col, lw=0.9)
                for _, row in g.iterrows():
                    _pt(ax, row.nodes, row.mtok_kwh, col, row.synthetic, ms=3)
                ax.plot([], [], color=col, lw=0.9, label=layout)
            ax.legend(loc="lower right", fontsize=6)
            ax.set_xscale("log", base=2)
            ticks = [n for n in sorted(d.nodes.unique()) if n in (1, 4, 13, d.nodes.max())]
            ax.set_xticks(ticks)
            ax.set_xticklabels([str(int(n)) for n in ticks], fontsize=6)
            ax.minorticks_off()
            ax.set_xlabel("nodes")
        else:
            key = d.layout if kind == "layout" else d.model.map(WORKLOAD_LABEL) + ", " + d.precision
            d = d.assign(key=key)
            order = ([l for l in QWEN_LAYOUTS if l in set(d.key)] if kind == "layout"
                     else sorted(d.key.unique()))
            for i, k in enumerate(order):
                g = d[d.key == k]
                col = WORKLOAD_COLOR.get(g.model.iloc[0], MUTED)
                for _, row in g.iterrows():
                    _pt(ax, row.mtok_kwh, i, col, row.synthetic, ms=3.5)
                    if kind == "prec":
                        ax.annotate(f"{row.mtok_kwh:.1f}", (row.mtok_kwh, i), xytext=(4, 0), textcoords="offset points",
                                    fontsize=6, color=INK2, va="center")
            ax.set_yticks(range(len(order)))
            ax.set_yticklabels(order, fontsize=6)
            ax.invert_yaxis()
            ax.set_xlabel("M tokens per kWh")
        ax.set_title(title, fontsize=7)
        if kind == "scale":
            ax.set_ylabel("M tokens per kWh")
            ax.set_ylim(0, 1.25 * d.mtok_kwh.max())
        elif kind == "prec":
            ax.set_xlim(0, 1.2 * d.mtok_kwh.max())  # the two models are ten times apart
        else:
            ax.set_xlim(0.9 * d.mtok_kwh.min(), 1.05 * d.mtok_kwh.max())
        style.finish(ax, trim=False)
    fig.text(0.01, 0.005, "GPU power only (no CPU, network, cooling)", fontsize=6, color=INK2)
    style.synthetic_mark(fig, e.synthetic.any())
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def kimi_inference(t):
    """Kimi K3 inference on B200 (vLLM), one series per server (colour and marker): output tokens/s per GPU and time per output token
    vs concurrency (1k in, 256 out), each server's sequence cap as a line; then time to first token and time per output token vs input
    length at 1 and 16 concurrent requests (128 out). Measured passes only."""
    p, runs = t.get("kimi_passes", pd.DataFrame()), t.get("kimi_runs", pd.DataFrame())
    if p.empty:
        return None
    p = p[~p.throwaway.astype(bool)]
    if not p["pass"].str.fullmatch(r"conc\d+").any():
        return None
    fig, axes = plt.subplots(2, 2, figsize=(TEXT_WIDTH, 4.3))
    (a, b), (c, d) = axes
    servers = []
    for i, (job, g) in enumerate(sorted(p.groupby(p.job.astype(str)), key=lambda x: (x[1].nodes.iloc[0], x[0]))):
        n = int(g.nodes.iloc[0])
        run = runs[runs.job.astype(str) == job].iloc[0] if len(runs) and (runs.job.astype(str) == job).any() else None
        servers.append((g, SERIES[i], ("o", "s", "D")[i % 3], f"{n} nodes ({n * 8} GPUs)", run))
    conc_ticks = set()
    for g, col, mk, label, run in servers:
        conc = g[g["pass"].str.fullmatch(r"conc\d+")].sort_values("concurrency")
        conc_ticks |= set(conc.concurrency)
        a.plot(conc.concurrency, conc.output_throughput_gpu, color=col, lw=1.0)
        _pt(a, conc.concurrency, conc.output_throughput_gpu, col, False, marker=mk)
        if run is not None and pd.notna(run.max_num_seqs):
            cap = int(run.max_num_seqs)
            a.axvline(cap, color=col, lw=0.6, ls=":", alpha=0.7, zorder=1)
            a.annotate(f"cap {cap:,} sequences", (cap, 0.04), xycoords=a.get_xaxis_transform(), xytext=(-2, 0), textcoords="offset points",
                       fontsize=5.5, color=col, ha="right", va="bottom", rotation=90)
        for q, ls, alpha in (("p50", "-", 1.0), ("p99", "--", 0.55)):
            y = conc[f"tpot_ms_{q}"]
            b.plot(conc.concurrency, y, color=col, lw=1.0 if q == "p50" else 0.7, ls=ls, alpha=alpha)
            _pt(b, conc.concurrency, y, col, False, marker=mk, alpha=alpha, ms=3.5 if q == "p99" else 4.5)
        ctx = g[g["pass"].str.fullmatch(r"ctx\d+k-c\d+")].sort_values("input_len")
        for n, h in ctx.groupby("concurrency"):
            k = h.input_len / 1024
            for ax, ycol, scale in ((c, "ttft_ms_p50", 1000), (d, "tpot_ms_p50", 1)):
                y = h[ycol] / scale
                ax.plot(k, y, color=col, lw=1.0, ls="-" if n == 1 else "--")
                _pt(ax, k, y, col, False, marker=mk)
    runs_legend = [_handle(col, label, False, marker=mk) for _, col, mk, label, _ in servers]
    a.set_ylim(0, None)
    a.set_ylabel("output tokens/s per GPU")
    a.legend(handles=runs_legend, loc="upper left", fontsize=6)
    b.set_yscale("log")
    b.set_ylabel("time per output token (ms)")
    b.legend(handles=runs_legend + [_handle(INK2, "median", False, marker="", ls="-"), _handle(INK2, "p99", False, marker="", ls="--", alpha=0.55)],
             loc="upper left", fontsize=6, ncol=2)
    ticks = sorted(x for x in conc_ticks if np.log2(x) % 2 == 0)  # 1, 4, 16, ...: 128, 256, 512 side by side do not read
    for ax in (a, b):
        ax.set_xscale("log", base=2)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(int(x)) for x in ticks], fontsize=6)
        ax.minorticks_off()
        ax.set_xlabel("concurrent requests (1k in, 256 out)")
    ctx_all = p[p["pass"].str.fullmatch(r"ctx\d+k-c\d+")]
    conc_legend = [_handle(INK2, "1 concurrent", False, marker="", ls="-"), _handle(INK2, "16 concurrent", False, marker="", ls="--")]
    for ax, label in ((c, "time to first token, median (s)"), (d, "time per output token, median (ms)")):
        if ctx_all.empty:
            ax.set_visible(False)
            continue
        ks = sorted(ctx_all.input_len.unique() // 1024)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(ks)
        ax.set_xticklabels([f"{int(x)}k" for x in ks], fontsize=6)
        ax.minorticks_off()
        ax.set_xlabel("input tokens (128 out)")
        ax.set_ylabel(label)
    if not ctx_all.empty:  # one legend for the bottom row: on the left it would cross the lines
        d.legend(handles=runs_legend + conc_legend, loc="upper left", fontsize=6)
    for ax in axes.flat:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        style.finish(ax, trim=False)
    fig.tight_layout()
    return fig


def kimi_training(t):
    """Kimi K3 pretraining at 48 nodes: step time per measured step (milabench's timer), peak memory per pipeline stage (max over its
    8 ranks) against the GPU's capacity, and the loss per step. Completed runs only: the full-size one, and fewer layers beside it."""
    r, st, sg = (t.get(k, pd.DataFrame()) for k in ("kimi_train_runs", "kimi_train_steps", "kimi_train_stages"))
    if r.empty or st.empty or "step_median" not in r:
        return None
    done = r[(r.job_state == "COMPLETED") & r.step_median.notna()].sort_values("layers", ascending=False)
    if done.empty:
        return None
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 2.35), gridspec_kw={"width_ratios": [1, 1.5, 1]})
    handles = []
    for (_, x), col, marker in zip(done.iterrows(), (SERIES[0], SERIES[1]), ("o", "s")):
        s = st[st.job == x.job].sort_values("step")
        m = s[s.measured.astype(bool)]
        a.plot(m.step, m.step_s, color=col, lw=1.0)
        _pt(a, m.step, m.step_s, col, False, marker=marker, ms=3.5)
        c.plot(s.step, s.loss, color=col, lw=1.0)
        _pt(c, s.step, s.loss, col, False, marker=marker, ms=3.5)
        g = sg[sg.job == x.job].sort_values("stage")
        full = col == SERIES[0]
        b.plot(g.stage, g.reserved_gib, color=col, lw=1.0 if full else 0.8)
        _pt(b, g.stage, g.reserved_gib, col, False, marker=marker, ms=3 if full else 2.5)
        if full:  # allocated for the full-size run only: two lines per run do not read
            b.plot(g.stage, g.allocated_gib, color=col, lw=0.8, ls=(0, (2, 1.5)), alpha=0.8)
        handles.append(_handle(col, f"{int(x.layers)} layers, {str(x.grad_reduce).upper()} reduction", False, marker=marker, ls="-"))
    cap = done.capacity_gib.max()
    if pd.notna(cap):
        b.axhline(cap, color=CRITICAL, lw=0.7, ls=(0, (1, 2)))
        b.annotate(f"GPU capacity, {cap:.1f} GiB", (0.5, cap), xycoords=("axes fraction", "data"), xytext=(0, -2), textcoords="offset points",
                   fontsize=6, color=CRITICAL, ha="center", va="top")
    b.legend(handles=[_handle(SERIES[0], "reserved", False, ls="-"), _handle(SERIES[0], "allocated", False, marker="", ls=(0, (2, 1.5)))],
             loc="lower left", fontsize=6)
    a.set_xlabel("step (measured)")
    a.set_ylabel("step time (s)")
    a.set_ylim(0, None)
    b.set_xlabel("pipeline stage (one node each)")
    b.set_ylabel("peak memory per GPU (GiB)")
    b.set_ylim(0, None)
    c.set_xlabel("step")
    c.set_ylabel("training loss")
    for ax in (a, b, c):
        style.finish(ax, trim=False)
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), fontsize=6, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


FIGURES = {
    "machine_status": machine_status,
    "health_gpus": health_gpus,
    "health_links": health_links,
    "pair_matrix": pair_matrix,
    "collectives_scale": collectives_scale,
    "interference": interference,
    "reference_drift": reference_drift,
    "qwen_layouts": qwen_layouts,
    "qwen_levers": qwen_levers,
    "step_decomposition": step_decomposition,
    "glm_layouts": glm_layouts,
    "qwen_scaling": qwen_scaling,
    "glm_scaling": glm_scaling,
    "precision_speed": precision_speed,
    "precision_loss": precision_loss,
    "checkpoint": checkpoint,
    "stability": stability,
    "stability_glm": lambda t: stability(t, GLM),
    "stragglers": stragglers,
    "startup": startup,
    "energy": energy,
    "kimi_inference": kimi_inference,
    "kimi_training": kimi_training,
}


def render_all(tables, out_dir, only=None) -> dict[str, bool]:
    """Every figure to <out_dir>/<id>.pdf, and the synthetic flags (\\syntheticnote{<id>})."""
    style.apply()
    done, flags = {}, []
    for name, fn in FIGURES.items():
        if only and name not in only:
            continue
        style.reset_synthetic()
        fig = fn(tables)
        flags.append(rf"\expandafter\def\csname syn@{name}\endcsname{{{int(style.used_synthetic() and fig is not None)}}}")
        done[name] = fig is not None
        if fig is not None:
            style.save(fig, out_dir / f"{name}.pdf")
        else:
            (out_dir / f"{name}.pdf").unlink(missing_ok=True)  # never leave a stale figure
    if not only:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "synthetic.tex").write_text("\n".join(flags) + "\n")
    return done
