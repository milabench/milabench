"""Figures of the Trillium report, one function per figure ID of DESIGN.md (tables in, figure out).

Every function takes the tables dict and returns a matplotlib figure, or None when its data is
missing. Conventions (style.py): measured points filled, synthetic points hollow, and a
SYNTHETIC watermark in every figure drawn wholly or partly from synthetic data.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, LogLocator
from tt_analysis.figures import _activity, _cpu_drop

from . import style
from .machine import GLM_SEQS_PER_COPY, GLM_STRONG_SEQS, GPUS_PER_NODE, NODE_NET_GBS, NODES, SUBNET_OF
from .style import (BLUE, COLLECTIVE_COLOR, COLLECTIVE_MARKER, CRITICAL, GLM, INK2, LIGHT, MUTED, QWEN,
                    RECIPE_COLOR, SCALE_COLOR, SEQUENTIAL, SERIES, TEXT_WIDTH, TOKENS_COLOR, WORKLOAD_COLOR,
                    WORKLOAD_LABEL, marker_kw)
from .select import completed, drift, qwen_base, with_energy

QWEN_LAYOUTS = ["fsdp", "hsdp8", "hsdp16", "tp2-fsdp", "tp4-fsdp", "pp2-fsdp", "pp4-fsdp"]
GLM_LAYOUTS = ["pp13-ep8", "pp13-ep16", "pp13-ep32", "pp26-ep8", "pp6-ep16"]
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

def health_gpus(t):
    """Per-GPU matmul BF16 and FP8 by node, with the median and a -5 % line (D1)."""
    h = t["node_health"]
    if h.empty:
        return None
    h = h[h.metric.isin(["matmul_bf16_tflops", "matmul_fp8_tflops"]) & h.host.isin(POS)]
    if h.empty:
        return None
    fig, axes = plt.subplots(2, 1, figsize=(TEXT_WIDTH, 3.0), sharex=True)
    for ax, (metric, label) in zip(axes, (("matmul_bf16_tflops", "BF16 matmul\n(TFLOP/s per GPU)"),
                                          ("matmul_fp8_tflops", "FP8 matmul\n(TFLOP/s per GPU)"))):
        d = h[h.metric == metric].copy()
        d["x"] = d.host.map(POS) + (d.gpu.astype(float) - 3.5) * 0.06
        for syn, part in d.groupby("synthetic"):
            _pt(ax, part.x, part.value, INK2, syn, ms=2.2, mew=0.5)
        med = d.value.median()
        ax.axhline(med, color=MUTED, lw=0.7, zorder=1)
        ax.axhline(0.95 * med, color=CRITICAL, lw=0.7, ls=(0, (1, 2)), zorder=1)
        ax.text(-0.5, med, f"median {med:,.0f}", fontsize=6, color=INK2, ha="left", va="bottom",
                bbox={"fc": "white", "ec": "none", "pad": 0.3}, zorder=5)
        ax.text(-0.5, 0.95 * med, "−5 %", fontsize=6, color=CRITICAL, ha="left", va="bottom")
        _label_outliers(ax, d, "x", "value", 0.95 * med, lambda r: f"{r.host} GPU{int(r.gpu)}")
        ax.set_ylim(0.9 * med, 1.03 * med)
        ax.set_ylabel(label)
        style.finish(ax, trim=False)
    _subnet_axis(axes)
    axes[-1].set_xlabel(f"node ({h.host.nunique()} nodes, 8 GPUs each), grouped by subnet")
    style.synthetic_mark(fig, h.synthetic.any())
    fig.tight_layout()
    return fig


def health_links(t):
    """Per-node NVLink all-reduce bus bandwidth, and per-HCA loopback bandwidth (D1)."""
    h = t["node_health"]
    if h.empty:
        return None
    h = h[h.host.isin(POS)]
    nv, hca = h[h.metric == "intra_busbw_gbs"].copy(), h[h.metric == "hca_loopback_gbs"].copy()
    if nv.empty and hca.empty:
        return None
    fig, axes = plt.subplots(2, 1, figsize=(TEXT_WIDTH, 3.0), sharex=True)
    for ax, d, label, frac, name in (
        (axes[0], nv, "NVLink all-reduce\nbus BW (GB/s, per node)", 0.9, lambda r: r.host),
        (axes[1], hca, "loopback BW per HCA\n(GB/s)", 0.8, lambda r: f"{r.host} {r.hca}"),
    ):
        if d.empty:
            continue
        offsets = {k: i for i, k in enumerate(sorted(d.hca.dropna().unique()))} if "hca" in d else {}
        d["x"] = d.host.map(POS) + (d.hca.map(offsets).fillna(1.5) - 1.5) * 0.15 if offsets else d.host.map(POS)
        for syn, part in d.groupby("synthetic"):
            _pt(ax, part.x, part.value, INK2, syn, ms=2.5, mew=0.6)
        med = d.value.median()
        ax.axhline(med, color=MUTED, lw=0.7, zorder=1)
        ax.axhline(frac * med, color=CRITICAL, lw=0.7, ls=(0, (1, 2)), zorder=1)
        ax.text(-0.5, med, f"median {med:.0f}", fontsize=6, color=INK2, ha="left", va="bottom",
                bbox={"fc": "white", "ec": "none", "pad": 0.3}, zorder=5)
        ax.text(-0.5, frac * med, f"{frac - 1:+.0%}".replace("-", "−"), fontsize=6, color=CRITICAL,
                ha="left", va="bottom")
        _label_outliers(ax, d, "x", "value", frac * med, name)
        ax.set_ylim(0, 1.15 * d.value.max())
        ax.set_ylabel(label)
        style.finish(ax, trim=False)
    n_hca = hca.groupby("host").hca.nunique().median() if not hca.empty else 0
    _subnet_axis(axes)
    axes[-1].set_xlabel(f"node, grouped by subnet ({n_hca:.0f} HCAs per node)")
    style.synthetic_mark(fig, h[h.metric.isin(["intra_busbw_gbs", "hca_loopback_gbs"])].synthetic.any())
    fig.tight_layout()
    return fig


def pair_matrix(t):
    """Bandwidth and latency of every node pair, nodes ordered by subnet (D2)."""
    p = t["pairs"]
    if p.empty:
        return None
    p = p[(p.status == "ok") & p.a.isin(POS) & p.b.isin(POS)]
    n = len(NODES)
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.75))
    groups, edges, centers, starts = _subnets()
    for ax, col, label in ((axes[0], "bw_gbs", "bandwidth (GB/s, one direction)"),
                           (axes[1], "lat_us", "latency (µs, 8 B)")):
        m = np.full((n, n), np.nan)
        agg = p.groupby(["a", "b"])[col].median()
        for (a, b), v in agg.items():
            m[POS[a], POS[b]] = m[POS[b], POS[a]] = v
        im = ax.imshow(m, cmap=SEQUENTIAL, interpolation="nearest")  # darker = more
        for e in edges:
            ax.axhline(e - 0.5, color="white", lw=0.6)
            ax.axvline(e - 0.5, color="white", lw=0.6)
        ax.set_xticks(centers)
        ax.set_xticklabels([groups.iloc[a] for a in starts], fontsize=6)
        ax.set_yticks(centers)
        ax.set_yticklabels([groups.iloc[a] for a in starts], fontsize=6)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label(label, fontsize=7)
        cb.ax.tick_params(labelsize=6)
        cb.outline.set_visible(False)
        ax.set_title(f"{'bandwidth' if col == 'bw_gbs' else 'latency'}: median {np.nanmedian(m):.3g}, "
                     f"range {np.nanmin(m):.3g}–{np.nanmax(m):.3g}", fontsize=7)
    style.synthetic_mark(fig, p.synthetic.any(), "includes synthetic data")
    fig.tight_layout()
    return fig


def collectives_scale(t):
    """NCCL bus bandwidth vs nodes (1 GiB) per collective, bisection cuts; bus BW vs size (D3, D4)."""
    c = t["collectives"]
    if c.empty:
        return None
    c = c[c.status == "ok"]
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
    for y, text in ((NODE_NET_GBS, "800 Gb/s per node"), (NODE_NET_GBS / GPUS_PER_NODE, "its share per GPU")):
        ax.axhline(y, color=MUTED, lw=0.7, ls=(0, (4, 3)), zorder=1)
        ax.text(1, y, text, fontsize=6, color=INK2, va="bottom")
    handles = [_handle(COLLECTIVE_COLOR[k], k.replace("_", "-"), d3[d3.collective == k].synthetic.any(),
                       COLLECTIVE_MARKER[k], "-", ms=3.5) for k in COLLECTIVE_COLOR if k in set(d3.collective)]
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
    ax.set_xticklabels([str(int(x)) for x in ticks])
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
    ax.set_ylabel("bus bandwidth (GB/s)")
    style.finish(ax, trim=False)
    style.synthetic_mark(fig, c.synthetic.any())
    fig.tight_layout()
    return fig


# -- E, R. Jobs side by side, background load -------------------------------

def interference(t):
    """Throughput of each 13-node set alone and with the 3 others running (E1)."""
    r = _runs(t, "E1")
    if r.empty:
        return None
    m = _median(r, ["node_set", "condition"])
    sets = sorted(m.node_set.unique())
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH * 0.75, 2.2))
    color = {"alone": SERIES[0], "together": SERIES[1]}
    changes = []
    for i, s in enumerate(sets):
        g = m[m.node_set == s].set_index("condition")
        if {"alone", "together"} <= set(g.index):
            ax.plot([i, i], [g.value["alone"], g.value["together"]], color=LIGHT, lw=1.2, zorder=1)
            changes.append(100 * (g.value["together"] / g.value["alone"] - 1))
        for cond, row in g.iterrows():
            _pt(ax, i, row.value, color[cond], row.synthetic, "o" if cond == "alone" else "s")
    ax.set_xticks(range(len(sets)))
    ax.set_xticklabels([s.replace("d", "draw ").replace("-s", "\nset ") for s in sets], fontsize=6)
    handles = [_handle(color[c], c + (" (4 jobs at once)" if c == "together" else ""),
                       m[m.condition == c].synthetic.any(), "o" if c == "alone" else "s") for c in color]
    ax.legend(handles=handles, loc="lower left", ncol=2)
    if changes:
        ax.set_title(f"median change together vs alone: {np.median(changes):+.1f} %".replace("-", "−"))
    lo = m.value.min()
    ax.set_ylim(lo * 0.97, m.value.max() * 1.01)
    ax.set_ylabel("tokens/s per GPU")
    style.finish(ax, trim=False)
    style.synthetic_mark(fig, m.synthetic.any())
    fig.tight_layout()
    return fig


def reference_drift(t):
    """The same reference run (Qwen, 4 nodes, FSDP, 12k tokens per GPU) every ~3 h (R)."""
    r = _runs(t, "R")
    if r.empty:
        return None
    hours, trend = drift(r)
    r = r.assign(h=hours)
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH * 0.75, 2.0))
    med = r.tok_s_gpu.median()
    ax.axhspan(0.99 * med, 1.01 * med, color=LIGHT, alpha=0.35, lw=0, zorder=0)
    ax.text(r.h.max(), 1.01 * med, "median ±1 %", fontsize=6, color=INK2, ha="right", va="bottom")
    ax.plot(r.h, r.tok_s_gpu, color=SERIES[0], lw=0.7, zorder=2)
    for _, row in r.iterrows():
        _pt(ax, row.h, row.tok_s_gpu, SERIES[0], row.synthetic)
    if not np.isnan(trend):
        ax.set_title(f"linear trend over the window: {trend:+.1f} %".replace("-", "−"))
    ax.set_ylim(0.96 * r.tok_s_gpu.min(), 1.03 * r.tok_s_gpu.max())
    ax.set_xlabel("hours since the first reference run")
    ax.set_ylabel("tokens/s per GPU")
    style.finish(ax, trim=False)
    style.synthetic_mark(fig, r.synthetic.any())
    fig.tight_layout()
    return fig


# -- A-Q. Qwen layouts ---------------------------------------------------------

def qwen_layouts(t):
    """Throughput per GPU by layout and tokens per GPU, at 2, 13 and all nodes (A-Q)."""
    r = qwen_base(t["runs"])
    if r.empty:
        return None
    scales = [n for n in sorted(r.nodes.unique()) if n > 1]
    m = _median(r, ["nodes", "layout", "tokens_per_gpu"])
    layouts = [l for l in QWEN_LAYOUTS if l in set(m.layout)]
    fig, axes = plt.subplots(1, len(scales), figsize=(TEXT_WIDTH, 0.6 + 0.24 * len(layouts)), sharey=True,
                             sharex=True, squeeze=False)
    peak, flops = r.peak_flops_per_gpu.iloc[0], r.flops_per_token.iloc[0]
    for ax, n in zip(axes[0], scales):
        d = m[m.nodes == n]
        for i, layout in enumerate(layouts):
            g = d[d.layout == layout]
            if len(g):
                ax.plot([g.value.min(), g.value.max()], [i, i], color=LIGHT, lw=0.8, zorder=1)
            for _, row in g.iterrows():
                _pt(ax, row.value, i, TOKENS_COLOR.get(int(row.tokens_per_gpu), MUTED), row.synthetic, ms=4)
        if len(d):
            best = d.loc[d.value.idxmax()]
            ax.annotate(f"{100 * flops * best.value / peak:.0f} % MFU", (best.value, layouts.index(best.layout)),
                        xytext=(0, 5), textcoords="offset points", fontsize=6, color=INK2, ha="center")
        ax.set_title(f"{n} nodes ({n * GPUS_PER_NODE} GPUs)")
        ax.set_xlabel("tokens/s per GPU")
        style.finish(ax, trim=False)
    ax = axes[0][0]
    ax.set_yticks(range(len(layouts)))
    ax.set_yticklabels(layouts)
    ax.invert_yaxis()
    handles = [_handle(c, f"{k // 1024}k", m[m.tokens_per_gpu == k].synthetic.any())
               for k, c in TOKENS_COLOR.items() if k in set(m.tokens_per_gpu)]
    fig.legend(handles=handles, title="tokens per GPU per step", loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.0 + 0.02), fontsize=6, title_fontsize=6)
    style.synthetic_mark(fig, m.synthetic.any())
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


def _key(row) -> tuple:
    return tuple(row[k] for k in REF)


def qwen_levers(t):
    """Throughput change of one lever at a time, at 13 nodes, vs the fsdp / 12k / FP32 reduction /
    selective recomputation reference; BF16 reduction vs FP32 on the same layout (A-Q)."""
    r = _runs(t, "A-Q")
    if r.empty:
        return None
    r = r[r.nodes == 13]
    m = r.groupby(list(REF), dropna=False).agg(value=("tok_s_gpu", "median"), synthetic=("synthetic", "any"))
    ref_key = tuple(REF.values())
    if ref_key not in m.index:
        return None
    ref = m.loc[ref_key]
    rows = []
    names = {"layout": "{}", "tokens_per_gpu": "{} tokens per GPU", "ac": "recompute: {}",
             "grad_reduce": "BF16 reduction"}
    for key, row in m.iterrows():
        diff = [k for k, a, b in zip(REF, key, ref_key) if a != b]
        if len(diff) == 1:
            v = key[list(REF).index(diff[0])]
            v = f"{int(v) // 1024}k" if diff[0] == "tokens_per_gpu" else v
            label = "BF16 reduction (fsdp, vs FP32)" if diff[0] == "grad_reduce" else names[diff[0]].format(v)
            rows.append({"group": diff[0], "label": label,
                         "pct": 100 * (row.value / ref.value - 1), "synthetic": row.synthetic or ref.synthetic})
        if key[list(REF).index("grad_reduce")] == "bf16" and key[0] != "fsdp":  # vs FP32, same layout
            base = tuple("fp32" if k == "grad_reduce" else v for k, v in zip(REF, key))
            if base in m.index and diff == ["layout", "grad_reduce"]:
                rows.append({"group": "grad_reduce", "label": f"BF16 reduction ({key[0]}, vs FP32)",
                             "pct": 100 * (row.value / m.loc[base].value - 1),
                             "synthetic": row.synthetic or m.loc[base].synthetic})
    best_key = m.value.idxmax()
    rows.append({"group": "best", "label": "best: " + ", ".join(
        f"{int(v) // 1024}k" if k == "tokens_per_gpu" else str(v) for k, v in zip(REF, best_key)
        if v != REF[k]), "pct": 100 * (m.value.max() / ref.value - 1), "synthetic": m.loc[best_key].synthetic})
    b = pd.DataFrame(rows)
    order = ["tokens_per_gpu", "layout", "grad_reduce", "ac", "best"]
    b["o"] = b.group.map(order.index)
    b = b.sort_values(["o", "pct"], ascending=[True, False]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH * 0.85, 0.5 + 0.17 * len(b)))
    y = np.arange(len(b))
    for yi, row in zip(y, b.itertuples()):
        col = BLUE[650] if row.group == "best" else SERIES[0]
        ax.barh(yi, row.pct, height=0.62, color="white" if row.synthetic else col, edgecolor=col, lw=0.8)
        ax.text(row.pct, yi, f" {row.pct:+.0f} %".replace("-", "−") if row.pct >= 0 else
                f"{row.pct:+.0f} % ".replace("-", "−"), va="center", ha="left" if row.pct >= 0 else "right",
                fontsize=6, color=INK2)
    ax.axvline(0, color=INK2, lw=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(b.label)
    ax.invert_yaxis()
    lim = max(abs(b.pct).max() * 1.25, 5)
    ax.set_xlim(min(b.pct.min() * 1.4, -2), lim)
    ax.set_xlabel(f"throughput change vs fsdp, 12k tokens, FP32 reduction, selective recompute "
                  f"({ref.value:,.0f} tokens/s per GPU), 13 nodes (%)")
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
    # 12-13 nodes, 26, and the machine (48-52, or 39 at 47 nodes): a layout comparison holds at a node count.
    r["scale"] = np.select([r.nodes >= 0.75 * r.nodes.max(), r.nodes < 20], ["large", "small"], "mid")
    m = _median(r, ["label", "layout", "nodes", "copies", "scale"])
    ep = t["ep_groups"]
    a2a = (ep.merge(r[["run_id", "label", "nodes"]], on="run_id").groupby(["label", "nodes"]).a2a_s.median()
           if not ep.empty else pd.Series(dtype=float))
    order = {l: i for i, l in enumerate(GLM_LAYOUTS)}
    labels = sorted(m.label.unique(), key=lambda s: (order.get(s.split(" ")[0], len(order)), s))
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 0.7 + 0.22 * len(labels)), sharey=True)
    for i, lab in enumerate(labels):
        for _, row in m[m.label == lab].iterrows():
            col = SCALE_COLOR[row.scale]
            _pt(axes[0], row.value, i, col, row.synthetic)
            if (lab, row.nodes) in a2a.index:
                _pt(axes[1], a2a[(lab, row.nodes)], i, col, row.synthetic)
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
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=6)
    style.synthetic_mark(fig, m.synthetic.any())
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return fig


# -- B. Scaling --------------------------------------------------------------

def _gpu_axis(ax, values) -> None:
    ax.set_xscale("log", base=2)
    ticks = sorted(set(int(v) for v in values))
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(v) for v in ticks], fontsize=6)
    ax.minorticks_off()


def qwen_scaling(t):
    """Weak scaling (24k tokens per GPU) of the best layout and fsdp, and strong scaling (fixed
    global batch) of the best layout (B-Q)."""
    w, s = _runs(t, "B-Q-weak"), _runs(t, "B-Q-strong")
    if w.empty and s.empty:
        return None
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
        _gpu_axis(ax, s.gpus)
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
    handles = []
    for exp, col, text, above in (("B-G-weak", SERIES[0], f"weak: {GLM_SEQS_PER_COPY} sequences per copy", False),
                                  ("B-G-strong", SERIES[1], f"strong: {GLM_STRONG_SEQS} sequences in total", True)):
        g = r[r.experiment == exp]
        med = _median(g, ["copies"])
        if med.empty:
            continue
        ax.plot(med.copies, med.value, color=col, lw=1.0)
        bubble = g.groupby("copies").apply(lambda d: (d.pp - 1) / (d.microbatches + d.pp - 1), include_groups=False
                                           ).groupby("copies").median()
        for _, row in med.iterrows():
            _pt(ax, row.copies, row.value, col, row.synthetic, "o" if exp == "B-G-weak" else "s")
            ax.annotate(f"{bubble[row.copies]:.0%}", (row.copies, row.value), xytext=(0, 5 if above else -9),
                        textcoords="offset points", fontsize=6, color=INK2, ha="center")
        eff = 100 * med.value.iloc[-1] / med.value.iloc[0]
        handles.append(_handle(col, f"{text}: {eff:.0f} % of 1 copy at {int(med.copies.iloc[-1])}", med.synthetic.any(),
                               "o" if exp == "B-G-weak" else "s", "-"))
    ax.legend(handles=handles, loc="lower left", fontsize=6)
    ax.set_xticks(sorted(r.copies.unique()))
    ax.set_xticklabels([f"{int(c)}\n({int(c) * 13} n.)" for c in sorted(r.copies.unique())], fontsize=6)
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
        ax.set_title(f"{WORKLOAD_LABEL[model]}: {int(small)} vs {int(large)} nodes")
        ax.set_xlabel("speedup over BF16 (tokens/s)")
        style.finish(ax, trim=False)
    handles = [_handle(SCALE_COLOR["small"], "small scale", syn), _handle(SCALE_COLOR["large"], "whole machine", syn)]
    axes[0][0].legend(handles=handles, loc="lower left")
    style.synthetic_mark(fig, syn)
    fig.tight_layout()
    return fig


def precision_loss(t):
    """Loss difference to BF16 over 500 steps from the same seed and data, 13 nodes (C-loss)."""
    r = _runs(t, "C-loss")
    loss = t["loss"]
    if r.empty or loss.empty:
        return None
    d = loss.merge(r[["run_id", "model", "precision"]], on="run_id")
    models = [m for m in (QWEN, GLM) if m in set(d.model)]
    fig, axes = plt.subplots(1, len(models), figsize=(TEXT_WIDTH, 2.0), sharey=True, squeeze=False)
    for ax, model in zip(axes[0], models):
        g = d[d.model == model].pivot_table(index="step", columns="precision", values="loss")
        if "bf16" not in g:
            continue
        ax.axhline(0, color=INK2, lw=0.6)
        for rec in [p for p in RECIPE_COLOR if p in g]:
            diff = g[rec] - g["bf16"]
            ax.plot(diff.index, diff, color=RECIPE_COLOR[rec], lw=0.3, alpha=0.15)
            smooth = diff.rolling(25, center=True, min_periods=5).mean()
            syn = d[(d.model == model) & (d.precision == rec)].synthetic.any()
            ax.plot(smooth.index, smooth, color=RECIPE_COLOR[rec], lw=1.2, ls=(0, (4, 2)) if syn else "-",
                    label=f"{rec}: {diff.iloc[-100:].mean():+.3f}")
        ax.legend(loc="upper right", fontsize=6)
        ax.set_xlim(0, g.index.max())
        ax.set_title(WORKLOAD_LABEL[model])
        ax.set_xlabel("step (legend: mean of the last 100 steps)")
        style.finish(ax, trim=False)
    axes[0][0].set_ylabel("loss − BF16 loss\n(25-step mean; thin: per step)")
    style.synthetic_mark(fig, d.synthetic.any(), "includes synthetic data (dashed)")
    fig.tight_layout()
    return fig


# -- F. Operations -----------------------------------------------------------

def checkpoint(t):
    """Checkpoint save and load: total time and time training is blocked vs nodes, sync vs async,
    per workload and content; filesystem throughput (F1)."""
    c = t["checkpoints"]
    if c.empty:
        return None
    c = c.merge(t["runs"][["run_id", "model", "nodes"]], on="run_id")
    cases = sorted(set(zip(c.model, c.content)), key=lambda mk: (mk[0] != QWEN, mk[0], mk[1] != "weights"))
    case_color = {mk: SERIES[4 + i % 3] for i, mk in enumerate(cases)}
    series = [("save", "sync", "total_s", "save, sync", SERIES[0], "o", "-"),
              ("save", "async", "total_s", "save, async: total", SERIES[1], "s", "-"),
              ("save", "async", "blocked_s", "save, async: blocked", SERIES[1], "s", (0, (3, 2))),
              ("save", "async_pinned", "blocked_s", "save, async pinned: blocked", SERIES[2], "^", (0, (3, 2))),
              ("load", "sync", "total_s", "load", SERIES[3], "D", "-")]
    fig, axes = plt.subplots(1, len(cases) + 1, figsize=(TEXT_WIDTH, 2.6))
    for ax, (model, content) in zip(axes, cases):
        d = c[(c.model == model) & (c.content == content)]
        for op, mode, col, label, color, marker, ls in series:
            g = d[(d.op == op) & (d["mode"] == mode)]
            if g.empty:
                continue
            med = g.groupby("nodes")[col].median()
            ax.plot(med.index, med.values, color=color, lw=0.9, ls=ls)
            for _, row in g.iterrows():
                _pt(ax, row.nodes, row[col], color, row.synthetic, marker, ms=3)
        size = d.bytes.iloc[0] / 1e12
        ax.set_title(f"{WORKLOAD_LABEL[model]}, {content} ({size:.2g} TB)", fontsize=7)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_xticks(sorted(d.nodes.unique()))
        ax.set_xticklabels([str(int(n)) for n in sorted(d.nodes.unique())])
        ax.minorticks_off()
        ax.set_xlabel("nodes")
        style.finish(ax, trim=False)
    axes[0].set_ylabel("time (s)")
    ax = axes[-1]
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
    ax.set_xticks(sorted(c.nodes.unique()))
    ax.set_xticklabels([str(int(n)) for n in sorted(c.nodes.unique())])
    ax.minorticks_off()
    ax.set_ylim(0, None)
    ax.set_title("filesystem: write (solid), read (dashed)", fontsize=7)
    ax.set_xlabel("nodes")
    ax.set_ylabel("GB/s")
    style.finish(ax, trim=False)
    syn = c.synthetic.any()
    handles = [_handle(color, label, syn, marker, ls, ms=3)
               for op, mode, col, label, color, marker, ls in series if len(c[(c.op == op) & (c["mode"] == mode)])]
    handles += [_handle(color, f"GB/s: {WORKLOAD_LABEL[m]}, {k}", syn, "o", "-", ms=3)
                for (m, k), color in case_color.items()]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=6, bbox_to_anchor=(0.5, 0.0))
    style.synthetic_mark(fig, c.synthetic.any())
    fig.tight_layout(rect=(0, 0.16, 1, 0.97))
    return fig


def _long_run(t):
    r = _runs(t, "F2")
    mon = t["gpu_monitor"]
    if r.empty or mon.empty:
        return None, None
    run = r.sort_values("measured_steps").iloc[-1]
    return run, mon[mon.source == run.run_name]


def stability(t):
    """Throughput per step over the long run, with SM clock and power of every GPU (F2)."""
    run, mon = _long_run(t)
    if run is None or mon.empty:
        return None
    s = t["steps"][t["steps"].run_id == run.run_id].sort_values("t_s")
    fig, axes = plt.subplots(3, 1, figsize=(TEXT_WIDTH, 4.0), sharex=True, gridspec_kw={"height_ratios": [1.3, 1, 1]})
    ax = axes[0]
    ax.plot(s.t_s / 60, s.tok_s_gpu, color=LIGHT, lw=0.5)
    ax.plot(s.t_s / 60, s.tok_s_gpu.rolling(15, center=True).median(), color=SERIES[0], lw=1.1,
            ls=(0, (4, 2)) if run.synthetic else "-")
    ax.set_ylabel("tokens/s per GPU")
    ax.set_title(f"{run.layout}, {int(run.nodes)} nodes: per step (gray), 15-step median")
    t0 = run.measure_start if pd.notna(run.get("measure_start")) else mon.time.min()
    for ax, metric, label in ((axes[1], "sm_clock_mhz", "SM clock (MHz)"), (axes[2], "power_w", "GPU power (W)")):
        m = mon[mon.metric == metric].assign(minute=lambda d: (d.time - t0) / 60, gid=lambda d: d.host + " GPU"
                                             + d.gpu.astype(str))
        if m.empty:
            continue
        band = m.groupby("minute").value.agg(["min", "max", "median"])
        ax.fill_between(band.index, band["min"], band["max"], color=LIGHT, alpha=0.5, lw=0)
        ax.plot(band.index, band["median"], color=INK2, lw=0.8)
        worst = m.groupby("gid").value.min().idxmin()
        w = m[m.gid == worst]
        ax.plot(w.minute, w.value, color=CRITICAL, lw=0.9)
        ax.annotate(worst, (w.minute.iloc[w.value.argmin()], w.value.min()), xytext=(4, 0),
                    textcoords="offset points", fontsize=6, color=CRITICAL, va="center")
        ax.set_ylabel(label)
        style.finish(ax, trim=False)
    fig.legend(handles=[plt.Line2D([], [], color=LIGHT, lw=5, alpha=0.6, label=f"range, all {int(run.gpus)} GPUs"),
                            plt.Line2D([], [], color=INK2, lw=0.8, label="median GPU"),
                            plt.Line2D([], [], color=CRITICAL, lw=0.9, label="lowest GPU (named)")],
               loc="lower center", ncol=3, fontsize=6, title="SM clock and power:", title_fontsize=6)
    style.finish(axes[0], trim=False)
    axes[-1].set_xlabel("minutes since the start of the measurement")
    style.synthetic_mark(fig, bool(run.synthetic) or mon.synthetic.any())
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    return fig


def stragglers(t):
    """Per-GPU activity (busiest 10-minute window) and per-node CPU busy-core clock (lowest minute
    vs the first) over the long run (F2, node telemetry)."""
    run, mon = _long_run(t)
    if run is None or mon.empty:
        return None
    util, activity = _activity(mon)  # sharpest signal available (Rorqual helper)
    if util is None:
        return None
    util = util.assign(win=((util.time - util.time.min()) // 600).astype(int))
    peak = util.groupby(["host", "gpu", "win"]).value.median().groupby(["host", "gpu"]).max().reset_index()
    peak["x"] = peak.host.map(POS) + (peak.gpu.astype(float) - 3.5) * 0.06
    cpu = mon[mon.metric == "cpu_mhz_busy_mean"]
    fig, axes = plt.subplots(2 if len(cpu) else 1, 1, figsize=(TEXT_WIDTH, 3.0 if len(cpu) else 1.7),
                             sharex=True, squeeze=False)
    ax = axes[0][0]
    syn = bool(util.synthetic.any())
    _pt(ax, peak.x, peak.value, INK2, syn, ms=2.2, mew=0.5)
    out = peak.loc[peak.value.idxmax()]
    ax.plot(out.x, out.value, "o", color=CRITICAL, ms=3.5)
    ax.annotate(f"{out.host} GPU{int(out.gpu)}", (out.x, out.value), xytext=(5, -2), textcoords="offset points",
                fontsize=6, color=CRITICAL)
    ax.set_ylim(None, peak.value.min() + 1.3 * (peak.value.max() - peak.value.min()))  # room under the tag
    ax.set_ylabel(f"{activity.splitlines()[0]}\n(busiest 10 min)")
    style.finish(ax, trim=False)
    if len(cpu):
        ax = axes[1][0]
        drop = _cpu_drop(cpu).rename(columns={"cpu_drop_pct": "value"})  # Rorqual helper
        drop["x"] = drop.host.map(POS)
        _pt(ax, drop.x, drop.value, INK2, bool(cpu.synthetic.any()), ms=3)
        worst = drop.loc[drop.value.idxmin()]
        ax.plot(worst.x, worst.value, "o", color=CRITICAL, ms=3.5)
        ax.annotate(worst.host, (worst.x, worst.value), xytext=(5, -2), textcoords="offset points", fontsize=6,
                    color=CRITICAL)
        ax.set_ylabel("CPU busy-core clock,\nlowest minute vs first (%)")
        style.finish(ax, trim=False)
    _subnet_axis(axes)
    axes[-1][0].set_xlabel("node, grouped by subnet (the slow GPU waits least: highest activity)")
    style.synthetic_mark(fig, syn)
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
    panels = [
        ("Qwen layouts, 13 nodes, 24k", base[(base.nodes == 13) & (base.tokens_per_gpu == 24576)], "layout"),
        ("precision, whole machine", e[(e.experiment == "C-speed") & (e.nodes >= 0.9 * e.nodes.max())], "prec"),
        ("Qwen scale (B-Q-weak)", e[e.experiment == "B-Q-weak"], "scale"),
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
                top = med.values[-1] >= d.groupby("layout").mtok_kwh.median().max() - 1e-9
                ax.annotate(layout, (med.index[-1], med.values[-1]), xytext=(-2, 5 if top else -6),
                            textcoords="offset points", fontsize=6, color=INK2, ha="right",
                            va="bottom" if top else "top")
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
            ax.set_yticks(range(len(order)))
            ax.set_yticklabels(order, fontsize=6)
            ax.invert_yaxis()
            ax.set_xlabel("M tokens per kWh")
        ax.set_title(title, fontsize=7)
        if kind == "scale":
            ax.set_ylabel("M tokens per kWh")
            ax.set_ylim(0, 1.25 * d.mtok_kwh.max())
        else:
            ax.set_xlim(0.9 * d.mtok_kwh.min(), 1.05 * d.mtok_kwh.max())
        style.finish(ax, trim=False)
    fig.text(0.01, 0.005, "GPU power only (no CPU, network, cooling)", fontsize=6, color=INK2)
    style.synthetic_mark(fig, e.synthetic.any())
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


FIGURES = {
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
    "stragglers": stragglers,
    "startup": startup,
    "energy": energy,
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
