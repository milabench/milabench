#!/usr/bin/env python
"""Regenerate the tables, figures and report from raw results.

    analyze.py all                    # collect + synth + build + plot + report
    analyze.py all --no-collect       # from the local copy in data/raw
    analyze.py all --no-synthetic     # measured data only
    analyze.py quick                  # build + results table + scaling / placement figures (window checks)
    analyze.py collect [--clusters rorqual fir]
    analyze.py build | plot | report | synth

Raw data is copied from each cluster's login node (rsync over ssh, read-only)
into data/raw/<cluster>/; synthetic data (FORMAT.md formats) goes to
data/synthetic/. Run with scripts/analysis/.venv/bin/python.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from tt_analysis import build as build_mod  # noqa: E402
from tt_analysis import figures, synthetic, tables  # noqa: E402

RAW = HERE / "data" / "raw"
SYN = HERE / "data" / "synthetic"
TABLES = HERE / "data" / "tables"
REPORT = HERE / "report"
GENERATED = REPORT / "generated"
CLUSTERS = ["rorqual", "fir", "tamia"]
# Per cluster: directories under $SCRATCH copied into data/raw/<cluster>/.
REMOTE = ["tt38/output/runs", "tt38/health", "tt38/logs", "nodemon"]
QUICK_FIGURES = {"weak_scaling", "placement_pairs"}


def ssh(cluster: str, command: str) -> str:
    return subprocess.run(["ssh", "-o", "BatchMode=yes", cluster, command], capture_output=True,
                          text=True, timeout=60, check=True).stdout


def collect(clusters: list[str], srygaard: list[str]) -> None:
    for cluster in clusters:
        dest = RAW / cluster
        dest.mkdir(parents=True, exist_ok=True)
        try:
            scratch = ssh(cluster, "echo $SCRATCH").strip()
        except (subprocess.SubprocessError, OSError) as exc:
            print(f"[collect] {cluster}: ssh failed ({exc}); keeping the local copy", file=sys.stderr)
            continue
        existing = ssh(cluster, " ".join(f"test -e {scratch}/{d} && echo {d};" for d in REMOTE)).split()
        sources = [f"{cluster}:{scratch}/{d}" for d in existing]
        if sources:
            subprocess.run(["rsync", "-a", "--exclude", "install.*", "-e", "ssh -o BatchMode=yes",
                            *sources, f"{dest}/"], check=True)
        (dest / "topology.txt").write_text(ssh(cluster, "scontrol show topology"))
        # GPU configuration per node (full GPUs vs MIG slices), to select usable nodes.
        (dest / "node_gres.txt").write_text(ssh(cluster, "scontrol -o show node | sed -nE "
                                                "'s/^NodeName=([^ ]+) .* Gres=([^ ]+).*/\\1 \\2/p'"))
        # Jobs that ran in our reservations (tracker kept in journals/), for traffic isolation.
        tracker = HERE.parent.parent / "journals" / f"{cluster}_reservation_usage.csv"
        if tracker.exists():
            (dest / "reservations.csv").write_text(tracker.read_text())
        print(f"[collect] {cluster}: {', '.join(existing) or 'nothing'} + topology")
    for spec in srygaard:  # cluster:/path/to/slurm-healthcheck/results/<job>
        cluster, _, path = spec.partition(":")
        dest = RAW / cluster / "srygaard"
        dest.mkdir(parents=True, exist_ok=True)
        subprocess.run(["rsync", "-a", "-e", "ssh -o BatchMode=yes", f"{cluster}:{path}", f"{dest}/"], check=True)
        print(f"[collect] {cluster}: srygaard {path}")


def build(use_synthetic: bool):
    if use_synthetic and not SYN.exists():
        synthetic.generate(RAW, SYN)
    t = build_mod.build(RAW, SYN if use_synthetic else None, TABLES)
    runs = t["runs"]
    print(f"[build] {len(runs)} runs ({int(runs.synthetic.sum()) if len(runs) else 0} synthetic) -> {TABLES}")
    return t


def plot(t, only=None) -> None:
    done = figures.render_all(t, GENERATED / "figures", only)
    missing = [k for k, ok in done.items() if not ok]
    print(f"[plot] {sum(done.values())} figures -> {GENERATED / 'figures'}"
          + (f" (no data: {', '.join(missing)})" if missing else ""))
    tables.write_all(t, GENERATED / "tables")


def report() -> None:
    for _ in range(2):  # second pass resolves references
        proc = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
                              cwd=REPORT, capture_output=True, text=True)
        if proc.returncode:
            print(proc.stdout[-3000:], file=sys.stderr)
            raise SystemExit("[report] pdflatex failed")
    print(f"[report] {REPORT / 'main.pdf'}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["all", "quick", "collect", "synth", "build", "plot", "report"])
    parser.add_argument("--clusters", nargs="+", default=CLUSTERS)
    parser.add_argument("--srygaard", nargs="*", default=[], metavar="CLUSTER:PATH",
                        help="srygaard/slurm-healthcheck result directories to copy")
    parser.add_argument("--no-collect", action="store_true")
    parser.add_argument("--no-synthetic", action="store_true")
    args = parser.parse_args(argv)
    use_syn = not args.no_synthetic

    if args.command == "collect" or (args.command in ("all", "quick") and not args.no_collect):
        collect(args.clusters, args.srygaard)
    if args.command in ("synth", "all") and use_syn:
        synthetic.generate(RAW, SYN)
        print(f"[synth] -> {SYN}")
    if args.command in ("build", "all", "quick"):
        t = build(use_syn)
    elif args.command in ("plot", "report"):
        t = build_mod.load(TABLES)
    if args.command in ("plot", "all", "report"):
        plot(t)
    if args.command == "quick":
        plot(t, QUICK_FIGURES)
        r = t["runs"]
        r = r[~r.synthetic].sort_values("started").tail(20)
        cols = ["started", "cluster", "bench", "variant", "status", "gpus", "nodes_per_leaf", "tok_s_gpu", "mfu_pct", "step_s"]
        print(r[cols].to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    if args.command in ("report", "all"):
        report()


if __name__ == "__main__":
    main()
