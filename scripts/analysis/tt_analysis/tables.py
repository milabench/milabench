"""LaTeX tables and macros for the report (``report/generated/``)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd


def tex(value) -> str:
    s = "" if value is None or (isinstance(value, float) and pd.isna(value)) else str(value)
    for a, b in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#")):
        s = s.replace(a, b)
    return s


def tabular(df: pd.DataFrame, align: str) -> str:
    lines = [rf"\begin{{tabular}}{{{align}}}", r"\toprule",
             " & ".join(tex(c) for c in df.columns) + r" \\", r"\midrule"]
    lines += [" & ".join(tex(v) for v in row) + r" \\" for row in df.itertuples(index=False)]
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def _fmt(v, spec):
    return "" if pd.isna(v) else format(v, spec)


def results(runs: pd.DataFrame) -> str:
    r = runs[(runs.status == "completed") & ~runs.synthetic].sort_values(["cluster", "started"])
    df = pd.DataFrame({
        "cluster": r.cluster,
        "bench": r.bench.str.replace("tt-qwen38-", "").str.replace("tt-", ""),
        "job": r.job_id.astype(str).str.replace(r"\.0$", "", regex=True),
        "tag": r.variant.where(r.family.isin(["H", "S", "P", "ALL"]), ""),
        "GPUs": r.gpus.astype(int),
        "L": r.n_layers.astype(int),
        "tok/GPU": r.mb_seqs.astype(int).astype(str) + "×" + r.seq_len.astype(int).astype(str)
                   + r.grad_accum.map(lambda a: f"×{int(a)}" if a > 1 else ""),
        "leaves": r.nodes_per_leaf,
        "mom.": r.adam_moments,
        "tok/s/GPU": r.tok_s_gpu.map(lambda v: _fmt(v, ".0f")),
        "MFU": r.mfu_pct.map(lambda v: _fmt(v, ".1f")),
        "step s": r.step_s.map(lambda v: _fmt(v, ".2f")),
        "p90/p10": (r.step_s_p90 / r.step_s_p10).map(lambda v: _fmt(v, ".3f")),
        "GiB": r.mem_alloc_gib.map(lambda v: _fmt(v, ".0f")) + "/" + r.mem_reserved_gib.map(lambda v: _fmt(v, ".0f")),
        "init": r.init_s.map(lambda v: _fmt(v, ".0f")),
        "1st": r.first_step_s.map(lambda v: _fmt(v, ".0f")),
    })
    return tabular(df, "lllllrlllrrrrrrr")


def ledger(runs: pd.DataFrame) -> str:
    r = runs[~runs.synthetic].sort_values("started")
    df = pd.DataFrame({
        "started": r.started.str.replace("T", " ").str[:16],
        "cluster": r.cluster,
        "run": r.run_name.str.split(".").str[0],
        "bench": r.bench,
        "job": r.job_id.astype(str).str.replace(r"\.0$", "", regex=True).replace("nan", ""),
        "status": r.status,
        "rc": r.return_code.map(lambda v: _fmt(v, ".0f")),
        "excluded nodes": r.excluded_nodes.fillna(""),
    })
    return tabular(df, "lllllllp{3cm}")


def macros(tables: dict[str, pd.DataFrame]) -> str:
    runs = tables["runs"]
    real = runs[~runs.synthetic]
    n_syn = int(runs.synthetic.sum())
    lines = [
        rf"\newcommand{{\DataGenerated}}{{{datetime.now():%Y-%m-%d %H:%M}}}",
        rf"\newcommand{{\NRunsReal}}{{{len(real)}}}",
        rf"\newcommand{{\NRunsFailed}}{{{int((real.status != 'completed').sum())}}}",
        rf"\newcommand{{\NRunsSynthetic}}{{{n_syn}}}",
        rf"\newcommand{{\HasSynthetic}}{{{'1' if n_syn else '0'}}}",
    ]
    return "\n".join(lines) + "\n"


def write_all(tables: dict[str, pd.DataFrame], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.tex").write_text(results(tables["runs"]))
    (out / "ledger.tex").write_text(ledger(tables["runs"]))
    (out / "macros.tex").write_text(macros(tables))
