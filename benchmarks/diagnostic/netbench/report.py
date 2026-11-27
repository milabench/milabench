#!/usr/bin/env python
"""Slow-GPU report for a milabench run folder.

Collects the rows of every netbench entry (net-host, net-p2p, net-inter,
net-net, gpu-hbm, ...) from ``<run>/*.data`` and prints the GPU matrices and a
per node GPU x test table with the slow GPUs flagged.

    python report.py $MILABENCH_BASE/runs/<run name> [--threshold 0.1]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gpureport import print_matrices, print_slow_gpus


def load_rows(folder: Path):
    rows = []
    # workers' copies are tagged nolog and only rank 0 logs rows anyway
    for path in sorted(folder.glob("*.data")):
        if ".nolog." in path.name:
            continue
        with open(path) as f:
            for text in f:
                try:
                    entry = json.loads(text)
                except json.JSONDecodeError:
                    continue
                data = entry.get("data")
                if entry.get("event") == "data" and isinstance(data, dict) and "test" in data and "rate" in data:
                    rows.append(data)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path, help="milabench run folder")
    parser.add_argument("--threshold", type=float, default=0.10, help="Flag GPUs this fraction below the median")
    args = parser.parse_args()

    rows = load_rows(args.folder)
    if not rows:
        print(f"No netbench rows in {args.folder}/*.data")
        return 1

    def line(s=""):
        print(s)

    line(f"{args.folder}: {len(rows)} measurements")
    print_matrices(rows, line)
    print_slow_gpus(rows, line, args.threshold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
