#!/usr/bin/env python
"""Regenerate the Trillium tables, figures and report from raw results.

    analyze.py all                    # collect + synth + build + plot + report
    analyze.py all --no-collect       # from the local copy in data/raw
    analyze.py all --no-synthetic     # measured data only
    analyze.py collect | synth | build | plot | report

Raw data is copied read-only from trillium-gpu:$SCRATCH/trillium/ (rsync over ssh) into
data/raw/trillium/; synthetic data (../FORMAT.md formats) goes to data/synthetic/trillium/.
Run with scripts/analysis/.venv/bin/python (same requirements as the Rorqual pipeline).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from tri_analysis import build as build_mod  # noqa: E402
from tri_analysis import figures, synthetic, tables  # noqa: E402
from tri_analysis.machine import CLUSTER  # noqa: E402
from tri_analysis.select import failed_runs, series_gaps  # noqa: E402

RAW = HERE / "data" / "raw"
SYN = HERE / "data" / "synthetic"
TABLES = HERE / "data" / "tables"
REPORT = HERE.parent / "report"
GENERATED = REPORT / "generated"
HOST = "trillium-gpu"
# Directories under $SCRATCH/trillium copied into data/raw/trillium/. Checkpoints: only their
# JSON metadata, never the checkpoint files themselves (up to ~9 TB).
REMOTE = ["runs", "health", "nccl", "nodemon"]
CHECKPOINT_FILTER = ["--include", "*/", "--include", "*.json", "--include", "*.jsonl", "--exclude", "*",
                     "--prune-empty-dirs"]


def ssh(command: str) -> str:
    return subprocess.run(["ssh", "-o", "BatchMode=yes", HOST, command], capture_output=True, text=True,
                          timeout=60, check=True).stdout


def collect() -> None:
    dest = RAW / CLUSTER
    dest.mkdir(parents=True, exist_ok=True)
    try:
        root = ssh("echo $SCRATCH").strip() + "/trillium"
        existing = ssh(" ".join(f"test -e {root}/{d} && echo {d};" for d in [*REMOTE, "checkpoints"])).split()
    except (subprocess.SubprocessError, OSError) as exc:
        print(f"[collect] {HOST}: ssh failed ({exc}); keeping the local copy", file=sys.stderr)
        return
    rsync = ["rsync", "-a", "-e", "ssh -o BatchMode=yes"]
    try:
        sources = [f"{HOST}:{root}/{d}" for d in existing if d != "checkpoints"]
        if sources:
            subprocess.run([*rsync, *sources, f"{dest}/"], check=True, timeout=3600)
        if "checkpoints" in existing:
            subprocess.run([*rsync, *CHECKPOINT_FILTER, f"{HOST}:{root}/checkpoints", f"{dest}/"], check=True,
                           timeout=600)
    except (subprocess.SubprocessError, OSError) as exc:
        print(f"[collect] {HOST}: rsync failed ({exc}); the local copy may be partial", file=sys.stderr)
        return
    print(f"[collect] {HOST}: {', '.join(existing) or 'nothing'}")


def build(use_synthetic: bool):
    if use_synthetic and not SYN.exists():
        synthetic.generate(SYN)
    t = build_mod.build(RAW, SYN if use_synthetic else None, TABLES)
    runs = t["runs"]
    n_syn = int(runs.synthetic.sum()) if len(runs) else 0
    print(f"[build] {len(runs)} runs ({n_syn} synthetic) -> {TABLES}")
    return t


def check_series(t) -> None:
    """Each GLM-5 series holds all its node counts: always true with the synthetic data filling the gaps."""
    if t["runs"].empty:
        return
    gaps = series_gaps(t["runs"])
    failed = failed_runs(t["runs"], *gaps)  # a failure leaves its node count empty on purpose: the figure marks it
    known = {(e, n) for e, n in zip(failed.experiment, failed.nodes)} if len(failed) else set()
    unexplained = {e: [n for n in nodes if (e, n) not in known] for e, nodes in gaps.items()}
    if any(unexplained.values()) and t["runs"].synthetic.any():
        raise SystemExit(f"[plot] GLM-5 series without some of their node counts: {unexplained}")
    if gaps:
        print(f"[plot] not run yet, GLM-5 series without these node counts: {gaps}")


def plot(t) -> None:
    check_series(t)
    done = figures.render_all(t, GENERATED / "figures")
    missing = [k for k, ok in done.items() if not ok]
    print(f"[plot] {sum(done.values())} figures -> {GENERATED / 'figures'}"
          + (f" (no data: {', '.join(missing)})" if missing else ""))
    tables.write_all(t, GENERATED / "tables")


def report() -> None:
    if not (REPORT / "main.tex").exists():
        print(f"[report] no {REPORT / 'main.tex'}; skipped")
        return
    for _ in range(2):  # second pass resolves references
        proc = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
                              cwd=REPORT, capture_output=True, text=True)
        if proc.returncode:
            print(proc.stdout[-3000:], file=sys.stderr)
            raise SystemExit("[report] pdflatex failed")
    print(f"[report] {REPORT / 'main.pdf'}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["all", "collect", "synth", "build", "plot", "report"])
    parser.add_argument("--no-collect", action="store_true")
    parser.add_argument("--no-synthetic", action="store_true")
    args = parser.parse_args(argv)
    use_syn = not args.no_synthetic

    if args.command == "collect" or (args.command == "all" and not args.no_collect):
        collect()
    if args.command in ("synth", "all") and use_syn:
        synthetic.generate(SYN)
        print(f"[synth] -> {SYN}")
    if args.command in ("build", "all"):
        t = build(use_syn)
    elif args.command in ("plot", "report"):
        t = build_mod.load(TABLES)
    if args.command in ("plot", "all", "report"):
        plot(t)
    if args.command in ("report", "all"):
        report()


if __name__ == "__main__":
    main()
