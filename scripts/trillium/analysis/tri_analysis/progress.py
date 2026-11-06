"""What ran against what is planned, per figure: ``analyze.py progress`` and data/tables/progress.csv.

A point is a configuration at a node count; repetitions don't count. Done: a completed run in the report. Answered: it
failed, and the failure is the result (out of memory, too slow). The plan is DESIGN.md, the launchers (glm/10_ag.sh,
20_bg.sh, 30_c.sh, 40_f1.sh), the Qwen agent's list for 26, 39 and 43 nodes ($SCRATCH/trillium/qwen/large/dry.sh) and the journal: since 00:28 on Nov 6
GLM-5's full checkpoint goes to 39 nodes, the pp13-ep16 26-node fallback only if that load fails. Network goes up to 52
as the design has it. Points at 50 or more nodes count apart. User, Nov 6: a compromise is a test at a smaller scale than initially planned, or
one holding the slow nodes when a run of that size could avoid them; full experiments at 52 nodes also counted until those
nodes were fixed, the same day, so runs after that are clean (select.SLOW_FIXED). Compromises go last of all and count
apart too. A quick 52-node run to check that it works is a verification, not the full-scale point: it goes in select.SET_ASIDE.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .select import checkpoints, completed
from .style import GLM, QWEN

TOP_NOW = 51  # all 52 passed at 16:35 on Nov 6, then trig0114 stopped responding at 17:03; a point above this waits for it
NODE_OUT = "trig0114"
FIFTY = 50
# Node counts of a group that are compromises: below the design's scale (DESIGN.md, 10_ag.sh) or holding the slow nodes.
COMPROMISE = {"Network (D2-D4)": {25}, "Qwen layouts": {39}, "Qwen weak scaling": {25, 39, 43}, "Qwen strong scaling": {39},
              "GLM-5 layouts": {36}, "Precision speed": {25, 26, 39}, "Checkpoints": {25, 39}}
FIGURES = {"Network (D2-D4)": "pair_matrix, collectives_scale", "Qwen layouts": "qwen_layouts, qwen_levers, step_decomposition", "Qwen weak scaling": "qwen_scaling",
           "Qwen strong scaling": "qwen_scaling", "GLM-5 layouts": "glm_layouts", "GLM-5 scaling": "glm_scaling",
           "Precision speed": "precision_speed", "Precision loss": "precision_loss", "Checkpoints": "checkpoint",
           "Long run": "stability, stragglers"}
# Replaces the computed "what's left" of a group.
LEFT_NOTE = {"Network (D2-D4)": f"D2, D3, D4 at the end: at 52 with {NODE_OUT} back, otherwise at the largest scale available",
             "Long run": f"Qwen and GLM-5, at least 1 h each, at the end: at 52 with {NODE_OUT} back; under 50 only as a compromise"}
# Job names in sacct -> group. First match wins.
JOB_GROUPS = [("tri-net", "Network (D2-D4)"), ("tri-aq-fsdp-24k-rbf16", "Qwen layouts"), ("tri-aq-", "Qwen weak scaling"), ("tri-bqs", "Qwen strong scaling"),
              ("tri-c-", "Precision speed"), ("tri-closs", "Precision loss"), ("tri-ckpt", "Checkpoints"), ("tri-ref", "Reference"),
              ("glm5-f1", "Checkpoints"), ("glm5-ag", "GLM-5 layouts"), ("glm5-bg", "GLM-5 scaling"), ("glm5-c-speed", "Precision speed"),
              ("glm5-c-loss", "Precision loss")]


@dataclass
class Point:
    block: str
    label: str
    nodes: int
    want: dict  # columns of runs.csv -> value
    experiment: str = ""
    op: tuple | None = None  # F1: (model, content, op, mode)
    net: tuple | None = None  # D2-D4: (experiment, nodes)
    compromise: bool = False
    job_names: tuple = ()  # sacct JobName of a job that would run it (GLM's jobs hold several entries)


def _qwen(block, experiment, nodes, layout, tokens, ac="selective", reduce="fp32", precision="bf16", label=None, **extra):
    want = dict(model=QWEN, layout=layout, tokens_per_gpu=tokens, ac=ac, grad_reduce=reduce, precision=precision, nodes=nodes, **extra)
    text = label or f"{layout} {tokens // 1024}k" + ("" if ac == "selective" else f" {ac}") + ("" if reduce == "fp32" else " bf16 red.")
    return Point(block, f"{text} n{nodes}", nodes, want, experiment)


def _glm(block, experiment, layout, nodes, seqs, ac="full", precision="bf16", label=None):
    want = dict(model=GLM, layout=layout, nodes=nodes, global_batch_tokens=seqs * 8192, ac=ac, precision=precision)
    return Point(block, label or f"{layout} n{nodes} batch {seqs}", nodes, want, experiment)


def plan() -> list[Point]:
    p = []
    # Network, as the design has it (D3's 26 ran at 25, the job limit then).
    net = "Network (D2-D4)"
    for n in (1, 2, 4, 8, 13, 25, 26, 52):
        p.append(Point(net, f"D3 collectives n{n}", n, {}, net=("D3", n)))
    p += [Point(net, "D2 pairs, two jobs of 25", 25, {}, net=("D2", 25)), Point(net, "D2 all pairs n52", 52, {}, net=("D2", 52)),
          Point(net, "D4 bisection 26 + 26", 52, {}, net=("D4", 52))]
    # Qwen layouts: the 13-node cross as it was run (48k selective was skipped, projected out of memory), then small and top scale.
    a = "Qwen layouts"
    for layout, n in (("fsdp", 13), ("hsdp8", 13), ("hsdp16", 12), ("tp2-fsdp", 13), ("tp4-fsdp", 13), ("pp2-fsdp", 13), ("pp4-fsdp", 13)):
        for tokens in (12288, 24576):
            p.append(_qwen(a, "A-Q", n, layout, tokens))
    p += [_qwen(a, "A-Q", 13, "pp2-fsdp", 49152), _qwen(a, "A-Q", 13, "pp4-fsdp", 49152),
          _qwen(a, "A-Q", 13, "fsdp", 12288, ac="none"), _qwen(a, "A-Q", 13, "fsdp", 24576, ac="none"),
          _qwen(a, "A-Q", 13, "fsdp", 24576, ac="full"), _qwen(a, "A-Q", 13, "hsdp8", 24576, ac="full"),
          _qwen(a, "A-Q", 13, "fsdp", 49152, ac="full"), _qwen(a, "A-Q", 13, "hsdp8", 49152, ac="full"),
          _qwen(a, "A-Q", 13, "fsdp", 24576, reduce="bf16"), _qwen(a, "A-Q", 13, "hsdp8", 24576, reduce="bf16")]
    # The design's top-scale runs of the 3 best, as at 39 nodes: 50 nodes, as weak scaling; fsdp and hsdp8 are the same runs as B-Q-weak's.
    p += [_qwen(a, "A-Q", 50, "fsdp", 24576), _qwen(a, "A-Q", 50, "hsdp8", 24576), _qwen(a, "A-Q", 50, "fsdp", 24576, reduce="bf16")]
    p += [_qwen(a, "A-Q", n, "fsdp", 24576, ac="full") for n in (1, 2)]
    p += [_qwen(a, "A-Q", n, "fsdp", 49152, ac="full") for n in (1, 2)]
    p += [_qwen(a, "A-Q", 2, "fsdp", 24576, reduce="bf16"), _qwen(a, "A-Q", 1, "tp2-fsdp", 12288), _qwen(a, "A-Q", 2, "tp2-fsdp", 12288),
          _qwen(a, "A-Q", 1, "tp4-fsdp", 12288), _qwen(a, "A-Q", 39, "fsdp", 24576, reduce="bf16")]
    # Qwen scaling
    w = "Qwen weak scaling"
    for n in (1, 2, 4, 8, 13, 25, 26, 39, 50):
        p += [_qwen(w, "B-Q-weak", n, layout, 24576) for layout in ("fsdp", "hsdp8")]
    p.append(_qwen(w, "B-Q-weak", 43, "fsdp", 24576, label="fsdp 24k, 3 slow nodes"))
    s = "Qwen strong scaling"
    for layout, nodes in (("fsdp", (13, 26, 39, 52)), ("hsdp8", (13, 26, 52))):
        p += [Point(s, f"{layout} n{n}", n, dict(model=QWEN, layout=layout, nodes=n, experiment="B-Q-strong"), "B-Q-strong") for n in nodes]
    # GLM-5 layouts and scaling
    g = "GLM-5 layouts"
    for layout, n, seqs in (("pp13-ep8", 13, 256), ("pp13-ep8", 13, 64), ("pp6-ep16", 12, 256), ("pp10-ep8", 10, 200), ("pp13-ep16", 26, 512),
                            ("pp26-ep8", 26, 512), ("pp25-ep8", 25, 512), ("pp10-ep8", 20, 400), ("pp10-ep8", 30, 600), ("pp10-ep8", 40, 800),
                            ("pp6-ep16", 36, 768), ("pp10-ep8", 50, 1000), ("pp25-ep8", 50, 1024), ("pp6-ep16", 48, 1024), ("pp13-ep8", 52, 1024),
                            ("pp13-ep16", 52, 1024), ("pp13-ep32", 52, 1024), ("pp26-ep8", 52, 1024)):
        p.append(_glm(g, "A-G", layout, n, seqs))
    p += [_glm(g, "A-G", "pp13-ep8", 13, 256, ac="selective", label="pp13-ep8 selective n13"),
          _glm(g, "A-G", "pp13-ep8", 52, 1024, ac="selective", label="pp13-ep8 selective n52")]
    b = "GLM-5 scaling"
    for n, seqs, exp in ((13, 256, "weak"), (26, 512, "weak"), (39, 768, "weak"), (52, 1024, "weak"), (13, 768, "strong"), (26, 768, "strong"), (39, 768, "strong"), (52, 768, "strong")):
        p.append(_glm(b, f"B-G-{exp}", "pp13-ep8", n, seqs, label=f"{exp} n{n}"))
    # Precision
    c = "Precision speed"
    for n in (1, 25, 26, 39, 50):
        p += [Point(c, f"Qwen {rec} n{n}", n, dict(model=QWEN, precision=rec, nodes=n, experiment="C-speed"), "C-speed") for rec in ("bf16", "mxfp8")]
    for n in (13, 39, 52):
        p += [Point(c, f"GLM-5 {rec} n{n}", n, dict(model=GLM, precision=rec, nodes=n, experiment="C-speed"), "C-speed")
              for rec in ("bf16", "fp8-tensorwise", "fp8-blockwise", "mxfp8")]
    l = "Precision loss"
    for seed in (1234, 4321, 5678):
        p += [Point(l, f"Qwen {rec} seed {seed}", 13, dict(model=QWEN, precision=rec, seed=seed, experiment="C-loss"), "C-loss") for rec in ("bf16", "mxfp8")]
    p += [Point(l, "Qwen bf16 seed 1234, repeat", 13, dict(model=QWEN, precision="bf16", seed=1234, repetition=2, experiment="C-loss"), "C-loss"),
          Point(l, "Qwen bf16 seed 1234, crossed lane", 13, dict(model=QWEN, precision="bf16", seed=1234, repetition=3, experiment="C-loss"), "C-loss"),
          Point(l, "Qwen mxfp8 seed 1234, crossed lane", 13, dict(model=QWEN, precision="mxfp8", seed=1234, repetition=2, experiment="C-loss"), "C-loss")]
    for rec, label, want in (("bf16", "bf16", {}), ("mxfp8", "mxfp8", {}), ("bf16", "bf16 same seed", {"repetition": 2}),
                             ("bf16", "bf16 other seed", {"seed": 4321}), ("fp8-tensorwise", "tensorwise", {}), ("fp8-blockwise", "blockwise", {})):
        p.append(Point(l, f"GLM-5 {label}", 13, dict(model=GLM, precision=rec, experiment="C-loss", **want), "C-loss"))
    # Checkpoints: one operation at one node count
    k = "Checkpoints"
    for n in (1, 13, 25, 39, 50):
        for content, op, mode in (("weights", "save", "sync"), ("weights", "load", "sync"), ("full", "save", "sync"), ("full", "save", "async"),
                                  ("full", "save", "async_pinned"), ("full", "load", "sync")):
            p.append(Point(k, f"Qwen {content} {op} {mode} n{n}", n, {}, "F1", op=(QWEN, content, op, mode, n)))
    for n in (13, 39, 52):
        for content, op, mode in (("weights", "save", "sync"), ("weights", "save", "async"), ("weights", "load", "sync"),
                                  ("full", "save", "sync"), ("full", "save", "async"), ("full", "load", "sync")):
            p.append(Point(k, f"GLM-5 {content} {op} {mode} n{n}", n, {}, "F1", op=(GLM, content, op, mode, n)))
    # User, Nov 6: last of all; at 39 nodes only if nothing else is left and 50+ nodes are still out of reach.
    p.append(Point("Long run", "best Qwen layout, at least 1 h", 52, dict(model=QWEN, experiment="F2"), "F2"))
    p.append(Point("Long run", "GLM-5 pp13-ep8, at least 1 h", 52, dict(model=GLM, experiment="F2"), "F2"))
    for x in p:
        x.compromise = x.nodes in COMPROMISE.get(x.block, ())
    return p


def _matches(r: pd.DataFrame, want: dict) -> pd.DataFrame:
    for k, v in want.items():
        if k in r:
            # A first run without TT_REPETITION is repetition 1 (figures.precision_loss).
            r = r[r[k].fillna(1) == v] if k == "repetition" else r[r[k] == v]
    return r


def table(t: dict[str, pd.DataFrame], sacct: Path | None = None) -> pd.DataFrame:
    """One row per planned point: block, label, nodes, state (done, answered, active, not run, needs nodes)."""
    runs = t["runs"]
    report = runs[runs.in_report.astype(bool)] if "in_report" in runs else runs
    done = completed(runs, *{"A-Q", "A-G", "B-Q-weak", "B-Q-strong", "B-G-weak", "B-G-strong", "C-speed", "C-loss", "F2"})
    failed = report[report.status == "failed"]
    active = report[report.status == "running"]
    ck = checkpoints(t["checkpoints"], runs)
    ops, ops_slow = set(), set()
    if len(ck):
        held = ck.run_id.map(runs.drop_duplicates("run_id").set_index("run_id").slow_nodes).fillna("") != ""
        key = lambda d: set(zip(d.model, d.content, d.op, d["mode"], d.nodes.astype(int)))
        ops, ops_slow = key(ck[~held]), key(ck[held])
    fo = t["failed_ops"]
    failed_ops = set()
    if len(fo) and "model" in fo:
        failed_ops = set(zip(fo.model, fo.content, fo.op, fo["mode"], fo.nodes.astype(int)))
    coll, pairs = t["collectives"], t["pairs"]
    net = set()
    if len(coll):
        net |= set(zip(coll.experiment, coll.nodes.astype(int)))
    if len(pairs):
        net.add(("D2", 25))  # the only sweep so far; a larger one needs its size in the table
    rows = []
    for x in plan():
        if x.net:
            state = "done" if x.net in net else None
        elif x.op:
            state = "done" if x.op in ops else "with slow nodes" if x.op in ops_slow else "answered" if x.op in failed_ops else None
        else:
            exp = {"experiment": x.experiment} if x.experiment else {}
            want = {**exp, **x.want}
            hits = _matches(done, want)
            if len(hits) and (x.compromise or (hits.slow_nodes.fillna("") == "").any()):
                state = "done"
            elif len(hits):
                state = "with slow nodes"  # a compromise, not the full-scale point
            elif len(_matches(active, want)):
                state = "active"
            elif len(_matches(failed, want)):
                state = "answered"
            else:
                state = None
        rows.append({"block": x.block, "point": x.label, "nodes": x.nodes, "compromise": x.compromise,
                     "state": state or ("not run" if x.nodes <= TOP_NOW else "needs nodes")})
    return pd.DataFrame(rows)


def jobs_now(sacct: Path) -> dict[str, list[str]]:
    """Group -> jobs running or queued, read from the collector's sacct snapshot."""
    out = {}
    if not sacct.exists():
        return out
    j = pd.read_csv(sacct, sep="|", dtype=str)
    for row in j[j.State.str.startswith(("RUNNING", "PENDING"))].itertuples():
        group = next((g for prefix, g in JOB_GROUPS if row.JobName.startswith(prefix)), "Other")
        out.setdefault(group, []).append(f"{row.JobName} n{row.NNodes} ({row.State.lower()})")
    return out


def _left(g: pd.DataFrame) -> str:
    note = LEFT_NOTE.get(g.block.iloc[0])
    if note:
        return note
    left, wait = g[(g.state == "not run") & ~g.compromise], g[(g.state == "needs nodes") & ~g.compromise]
    parts = []
    if len(left) > 4:
        parts.append(f"{len(left)} points at {', '.join(str(n) for n in sorted(set(left.nodes)))} nodes")
    elif len(left):
        parts.append("; ".join(left.point))
    if len(wait):
        parts.append(f"{len(wait)} at {wait.nodes.max()} with {NODE_OUT} back")
    if (g.state == "active").any():
        parts.append("the rest is running")
    return "; ".join(parts) or "nothing"


def _compromise(g: pd.DataFrame) -> str:
    c = g[g.compromise]
    done, active = int(c.state.isin(["done", "answered"]).sum()) + int((g.state == "with slow nodes").sum()), int((c.state == "active").sum())
    last = int((c.state == "not run").sum())
    parts = [f"{done} done"] * bool(done) + [f"{active} running"] * bool(active) + [f"{last} last"] * bool(last)
    return ", ".join(parts) or "none"


def summary(p: pd.DataFrame, t: dict[str, pd.DataFrame], sacct: Path) -> pd.DataFrame:
    jobs = jobs_now(sacct)
    rows = []
    runs = t["runs"]
    h, pairs, coll = t["node_health"], t["pairs"], t["collectives"]
    if len(h):
        rows.append({"Figures": "health_gpus, health_links", "Group": "Health (D1)", "Done / planned": f"{h.host.nunique()} / 52 nodes",
                     "Running or queued": "", "What's left": "no matmul or per-HCA checks in the tool", "50+ nodes": "none", "Compromise": "none"})
    done = runs[(runs.status == "completed") & runs.in_report.astype(bool)]
    e1, ref = (done.experiment == "E1").sum(), (done.experiment == "R").sum()
    # The design has 4 sets of 13 nodes, so 52; the 3 sets that ran are a compromise.
    rows.append({"Figures": "interference", "Group": "Side by side (E1)", "Done / planned": "none", "Running or queued": "",
                 "What's left": f"4 sets of 13, at 52 with {NODE_OUT} back", "50+ nodes": "0 / 16", "Compromise": f"{e1} done"})
    rows.append({"Figures": "reference_drift", "Group": "Reference (R)", "Done / planned": f"{ref} runs, no fixed count",
                 "Running or queued": "; ".join(jobs.get("Reference", [])), "What's left": "more as the window goes on", "50+ nodes": "none", "Compromise": "none"})
    for block, g in p.groupby("block", sort=False):
        full = g[~g.compromise]
        now, big = full[full.nodes < FIFTY], full[full.nodes >= FIFTY]
        n_done = int(now.state.isin(["done", "answered"]).sum())
        failed = int((now.state == "answered").sum())
        share = f"{n_done} / {len(now)} ({100 * n_done / len(now):.0f}%)" if len(now) else "none"
        rows.append({"Figures": FIGURES[block], "Group": block, "Done / planned": share + (f", {failed} failed" if failed else ""),
                     "Running or queued": "; ".join(jobs.get(block, [])), "What's left": _left(g),
                     "50+ nodes": f"{int(big.state.isin(['done', 'answered']).sum())} / {len(big)}" if len(big) else "none",
                     "Compromise": _compromise(g)})
    rows.sort(key=lambda r: {"Health (D1)": 0, "Network (D2-D4)": 1}.get(r["Group"], 2))  # stable: the design's order, D first
    return pd.DataFrame(rows)[["Group", "Done / planned", "50+ nodes", "Compromise", "Running or queued", "What's left", "Figures"]]


def markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(out)
