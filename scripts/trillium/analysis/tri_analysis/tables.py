"""LaTeX tables and macros for the report (``report/generated/tables/``).

Every number taken from synthetic rows is wrapped in ``\\TriSyn{...}`` (red, "syn" superscript),
so a placeholder can never read as a measurement in the text.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
from tt_analysis.tables import ledger

from .machine import GLM_SEQS_PER_COPY, GLM_TOP_COPIES, GLM_TOP_NODES, N_NODES
from .select import completed as _completed
from .select import QWEN_BASE, checkpoints, drift, qwen_base, with_energy
from .style import GLM, QWEN

# Key numbers for the text (meaning in key_numbers); every one is defined, "??" without data.
MACROS = ["TriMatmulBF", "TriMatmulFP", "TriNodesChecked", "TriNodesFlagged", "TriPairBw", "TriPairBwMin",
          "TriPairLat", "TriAllReduceFull", "TriAlltoallFull", "TriBisection", "TriInterferencePct",
          "TriDriftPct", "TriQwenBestLayout", "TriQwenBestTokens", "TriQwenBestTokS", "TriQwenBestMFU",
          "TriQwenFsdpCommPct", "TriQwenWeakEff", "TriQwenWeakEffFromTwo", "TriGlmBestLayout", "TriGlmBestTokS", "TriGlmBestMFU",
          "TriGlmCopiesEff", "TriGlmCopiesEffTwo", "TriGlmCopiesEffFromTwo", "TriQwenFpBestRecipe", "TriQwenFpSpeedupSmall", "TriQwenFpSpeedupFull",
          "TriGlmFpBestRecipe", "TriGlmFpSpeedupSmall", "TriGlmFpSpeedupFull", "TriCkptQwenSave",
          "TriCkptGlmFullSave", "TriCkptGlmFullBlocked", "TriCkptGlmLoadPeak", "TriCkptWriteGBs", "TriStartupQwen", "TriStartupGlm",
          "TriEnergyBest"]
SYN_DEF = r"\providecommand{\TriSyn}[1]{{\color{red!70!black}#1\textsuperscript{syn}}}"


def _value(v, fmt, synthetic) -> str:
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return "??"  # never "nan" in the text
    text = format(v, fmt) if not isinstance(v, str) else v
    return rf"\TriSyn{{{text}}}" if synthetic else text


def key_numbers(t) -> dict[str, tuple]:
    """Macro name -> (value, format, synthetic); a name missing here is written as "??"."""
    out = {}
    runs = t["runs"]
    h = t["node_health"]
    if not h.empty:
        for metric, name in (("matmul_bf16_tflops", "TriMatmulBF"), ("matmul_fp8_tflops", "TriMatmulFP")):
            d = h[h.metric == metric]
            if len(d):
                out[name] = (d.value.median(), ",.0f", d.synthetic.any())
        v = h[h.metric == "verdict"]
        if len(v):
            out["TriNodesChecked"] = (v.host.nunique(), "d", v.synthetic.any())
            out["TriNodesFlagged"] = (v[v.status != "pass"].host.nunique(), "d", v.synthetic.any())
    p = t["pairs"]
    if not p.empty:
        out["TriPairBw"] = (p.bw_gbs.median(), ".0f", p.synthetic.any())
        out["TriPairBwMin"] = (p.bw_gbs.min(), ".0f", p.synthetic.any())
        out["TriPairLat"] = (p.lat_us.median(), ".1f", p.synthetic.any())
    c = t["collectives"]
    if not c.empty:
        d3 = c[(c.experiment == "D3") & (c.size_bytes == 2**30)]
        for coll, name in (("all_reduce", "TriAllReduceFull"), ("alltoall", "TriAlltoallFull")):
            d = d3[(d3.collective == coll) & (d3.nodes == d3.nodes.max())]
            if len(d):
                out[name] = (d.busbw_gbs.median(), ".0f", d.synthetic.any())
        d4 = c[c.experiment == "D4"]
        if len(d4):
            out["TriBisection"] = (d4.busbw_gbs.median(), ".0f", d4.synthetic.any())
    e1 = _completed(runs, "E1")
    if len(e1):
        m = e1.groupby(["node_set", "condition"]).tok_s_gpu.median().unstack()
        if {"alone", "together"} <= set(m.columns):
            out["TriInterferencePct"] = ((100 * (m.together / m.alone - 1)).median(), "+.1f", e1.synthetic.any())
    rr = _completed(runs, "R")
    if len(rr) > 1:
        out["TriDriftPct"] = (drift(rr)[1], "+.1f", rr.synthetic.any())  # linear trend over the window
    aq = _completed(runs, "A-Q")
    base = qwen_base(runs)
    base = base[base.nodes == 13] if len(base) else base
    if len(base):
        m = base.groupby(["layout", "tokens_per_gpu"]).agg(v=("tok_s_gpu", "median"), mfu=("mfu_pct", "median"),
                                                           syn=("synthetic", "any"))
        (layout, tokens), best = m.v.idxmax(), m.loc[m.v.idxmax()]
        out["TriQwenBestLayout"] = (layout, "", best.syn)
        out["TriQwenBestTokens"] = (f"{int(tokens) // 1024}k", "", best.syn)
        out["TriQwenBestTokS"] = (best.v, ",.0f", best.syn)
        out["TriQwenBestMFU"] = (best.mfu, ".0f", best.syn)
        one = aq[(aq.nodes == 1) & (aq.layout == "fsdp") & (aq.tokens_per_gpu == 24576)
                 & (aq.ac == QWEN_BASE["ac"]) & (aq.grad_reduce == QWEN_BASE["grad_reduce"])]
        f13 = base[(base.layout == "fsdp") & (base.tokens_per_gpu == 24576)]
        if len(one) and len(f13):
            share = 100 * (1 - one.step_s.median() / f13.step_s.median())
            out["TriQwenFsdpCommPct"] = (share, ".0f", one.synthetic.any() or f13.synthetic.any())
    w = _completed(runs, "B-Q-weak")
    if len(w):
        g = w[w.layout != "fsdp"] if (w.layout != "fsdp").any() else w
        med = g.groupby("nodes").tok_s_gpu.median()
        out["TriQwenWeakEff"] = (100 * med.iloc[-1] / med.iloc[0], ".0f", g.synthetic.any())
        if 2 in med.index:  # from the first run with cross-node traffic
            out["TriQwenWeakEffFromTwo"] = (100 * med.iloc[-1] / med[2], ".0f", g.synthetic.any())
    ag = _completed(runs, "A-G")
    if len(ag):
        per_copy = ag.global_batch_tokens / ag.copies / ag.seq_len
        big = ag[(ag.nodes >= 0.9 * ag.nodes.max()) & (ag.ac == "full") & (per_copy >= GLM_SEQS_PER_COPY / 2)]
        m = big.groupby("layout").agg(v=("tok_s_gpu", "median"), mfu=("mfu_pct", "median"), syn=("synthetic", "any"))
        if len(m):
            best = m.loc[m.v.idxmax()]
            out["TriGlmBestLayout"] = (m.v.idxmax(), "", best.syn)
            out["TriGlmBestTokS"] = (best.v, ",.0f", best.syn)
            out["TriGlmBestMFU"] = (best.mfu, ".0f", best.syn)
    bg = _completed(runs, "B-G-weak")
    if len(bg) > 1:
        bg = bg[(bg.layout == "pp13-ep8") & (bg.ac == "full") & (bg.precision == "bf16")]
        med = bg.groupby("copies").tok_s_gpu.median()
        if 1 in med.index and GLM_TOP_COPIES in med.index:  # never a ratio of other copy counts
            out["TriGlmCopiesEff"] = (100 * med[GLM_TOP_COPIES] / med[1], ".0f", bg.synthetic.any())
        if 1 in med.index and 2 in med.index:
            out["TriGlmCopiesEffTwo"] = (100 * med[2] / med[1], ".0f", bg.synthetic.any())
        if 2 in med.index and GLM_TOP_COPIES in med.index:
            out["TriGlmCopiesEffFromTwo"] = (100 * med[GLM_TOP_COPIES] / med[2], ".0f", bg.synthetic.any())
    cs = _completed(runs, "C-speed")
    for model, prefix in ((QWEN, "TriQwen"), (GLM, "TriGlm")):
        d = cs[cs.model == model] if len(cs) else cs
        if d.empty:
            continue
        m = d.groupby(["nodes", "precision"]).tok_s_gpu.median().unstack()
        if "bf16" not in m:
            continue
        speed = m.drop(columns="bf16").div(m.bf16, axis=0)
        if speed.empty:
            continue
        recipe = speed.iloc[-1].idxmax()
        out[f"{prefix}FpBestRecipe"] = (recipe, "", d.synthetic.any())
        out[f"{prefix}FpSpeedupSmall"] = (speed[recipe].iloc[0], ".2f", d.synthetic.any())
        out[f"{prefix}FpSpeedupFull"] = (speed[recipe].iloc[-1], ".2f", d.synthetic.any())
    ck = checkpoints(t["checkpoints"], runs)
    if not ck.empty:
        p = ck[(ck.model == GLM) & (ck.content == "full") & (ck.op == "load")].dropna(subset=["peak_allocated_gib"])
        if len(p):
            p = p[p.nodes == p.nodes.min()]
            out["TriCkptGlmLoadPeak"] = (p.peak_allocated_gib.median(), ".0f", p.synthetic.any())
        for model, content, col, mode, name in (
                (QWEN, "full", "total_s", "sync", "TriCkptQwenSave"),
                (GLM, "full", "total_s", "sync", "TriCkptGlmFullSave"),
                (GLM, "full", "blocked_s", "async", "TriCkptGlmFullBlocked")):
            d = ck[(ck.model == model) & (ck.content == content) & (ck.op == "save") & (ck["mode"] == mode)]
            d = d[d.nodes == d.nodes.max()]
            if len(d):
                out[name] = (d[col].median(), ".0f", d.synthetic.any())
        s = ck[(ck.op == "save") & (ck["mode"] == "sync")]
        if len(s):
            out["TriCkptWriteGBs"] = (s.groupby(["model", "content", "nodes"]).gb_s.median().max(), ".0f",
                                      s.synthetic.any())
    for model, name in ((QWEN, "TriStartupQwen"), (GLM, "TriStartupGlm")):
        d = _completed(runs)
        d = d[(d.model == model)].dropna(subset=["first_step_total_s"]) if len(d) else d
        if len(d):
            d = d[d.nodes == d.nodes.max()]
            out[name] = (d.first_step_total_s.median() / 60, ".0f", d.synthetic.any())
    pw = t["run_power"]
    if not pw.empty and len(base):
        e = with_energy(base, pw)
        if len(e):
            out["TriEnergyBest"] = (e.mtok_kwh.max(), ".0f", e.synthetic.any())
    return out


def macros(t) -> str:
    runs = t["runs"]
    real = runs[~runs.synthetic] if len(runs) else runs
    n_syn = int(runs.synthetic.sum()) if len(runs) else 0
    lines = [
        SYN_DEF,
        rf"\newcommand{{\DataGenerated}}{{{datetime.now():%Y-%m-%d %H:%M}}}",
        rf"\newcommand{{\NRunsReal}}{{{len(real)}}}",
        rf"\newcommand{{\NRunsFailed}}{{{int((real.status != 'completed').sum()) if len(real) else 0}}}",
        rf"\newcommand{{\NRunsSynthetic}}{{{n_syn}}}",
        rf"\newcommand{{\HasSynthetic}}{{{'1' if n_syn else '0'}}}",
        rf"\newcommand{{\TriNNodes}}{{{N_NODES}}}",
        rf"\newcommand{{\TriNGpus}}{{{N_NODES * 8}}}",
        rf"\newcommand{{\TriGlmTopNodes}}{{{GLM_TOP_NODES}}}",
        rf"\newcommand{{\TriGlmTopCopies}}{{{GLM_TOP_COPIES}}}",
        rf"\newcommand{{\TriPairsPerRound}}{{{N_NODES // 2}}}",
        rf"\newcommand{{\TriPairRounds}}{{{N_NODES - 1 + N_NODES % 2}}}",
        rf"\newcommand{{\TriPairsAll}}{{{N_NODES * (N_NODES - 1) // 2:,}}}",
    ]
    numbers = key_numbers(t)
    for name in MACROS:  # always defined: "??" until the data exists
        v, fmt, syn = numbers.get(name, ("??", "", False))
        lines.append(rf"\newcommand{{\{name}}}{{{_value(v, fmt, bool(syn))}}}")
    return "\n".join(lines) + "\n"


def qwen_layouts_table(runs: pd.DataFrame) -> str:
    """Qwen at 13 nodes, selective recompute: tokens/s per GPU (MFU %) by layout x tokens per GPU; synthetic in \\TriSyn."""
    r = qwen_base(runs)
    if r.empty:
        return ""
    r = r[r.nodes == 13]
    m = r.groupby(["layout", "tokens_per_gpu"]).agg(v=("tok_s_gpu", "median"), mfu=("mfu_pct", "median"),
                                                    syn=("synthetic", "any")).reset_index()
    cols = sorted(m.tokens_per_gpu.unique())
    rows = []
    for layout, g in m.groupby("layout", sort=False):
        cells = g.set_index("tokens_per_gpu")
        rows.append([layout] + [_value(f"{cells.v[c]:,.0f} ({cells.mfu[c]:.0f})", "", cells.syn[c])
                                if c in cells.index else "" for c in cols])
    df = pd.DataFrame(rows, columns=["layout"] + [f"{int(c) // 1024}k" for c in cols])
    # Not tt_analysis.tables.tabular: it escapes backslashes, and the cells hold \TriSyn{}.
    lines = [rf"\begin{{tabular}}{{l{'r' * len(cols)}}}", r"\toprule",
             " & ".join(df.columns) + r" \\", r"\midrule"]
    lines += [" & ".join(str(v) for v in row) + r" \\" for row in df.itertuples(index=False)]
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def write_all(tables: dict[str, pd.DataFrame], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    runs = tables["runs"]
    (out / "macros.tex").write_text(macros(tables))
    (out / "qwen_layouts.tex").write_text(qwen_layouts_table(runs))
    if len(runs):
        (out / "ledger.tex").write_text(ledger(runs.assign(excluded_nodes=runs.excluded_nodes.astype(object).fillna(""))))
