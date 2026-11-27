#!/usr/bin/env python
"""The benchmarks a homelab run will execute, in order, for the report's progress view.

    <milabench venv>/bin/python scripts/homelab/plan.py <config.yaml> --out <run dir>/report/plan.json

One entry per enabled benchmark (milabench's config order, which is its run order): its name, the client
passes that each produce a result (<bench>.<pass>), and whether it is skipped (quarantined on this machine).
Needs milabench (its config resolution); the report scripts read the file with the standard library.
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--out", required=True)
    parser.add_argument("--exclude", default="", help="comma-separated benchmarks this run skips")
    args = parser.parse_args(argv)

    from milabench.config import build_config

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import quarantine

    quarantined = quarantine.benches(quarantine.load())
    excluded = {b for b in args.exclude.split(",") if b}
    plan = []
    for name, cfg in build_config(args.config).items():
        if name.startswith("_") or not cfg.get("enabled", False):
            continue
        client = cfg.get("client") or {}
        passes = [p["name"] for p in client.get("passes") or []]
        entry = {"bench": name, "passes": passes, "definition": os.path.basename(cfg.get("definition", "")),
                 "tags": sorted(set(cfg.get("tags", [])))}
        if name in quarantined:
            entry["skipped"] = f"quarantined ({quarantined[name].get('kind')}): {quarantined[name].get('reason')}"
        elif name in excluded:
            entry["skipped"] = "excluded from this run"
        plan.append(entry)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(plan, f, indent=1)
    print(f"{len(plan)} benchmarks, {sum(max(1, len(p['passes'])) for p in plan)} results -> {args.out}")


if __name__ == "__main__":
    main()
