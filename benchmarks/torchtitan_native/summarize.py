#!/usr/bin/env python
"""Tabulate torchtitan_native results records (Markdown).

    python summarize.py <results dir or *.json files>...
"""

import json
import sys
from pathlib import Path


def _records(paths):
    for arg in paths:
        path = Path(arg)
        files = sorted(path.rglob("*.json")) if path.is_dir() else [path]
        for file in files:
            try:
                record = json.loads(file.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if record.get("schema", "").startswith("milabench.torchtitan_native"):
                yield file, record


def main(argv):
    columns = [
        "run", "gpus", "layers", "compile", "ac", "par", "tok/s/gpu", "MFU %",
        "step med (s)", "p90/med", "alloc GiB", "rsvd GiB", "data %", "init s", "1st step s",
    ]
    print("| " + " | ".join(columns) + " |")
    print("|" + "---|" * len(columns))
    for _, r in _records(argv or ["."]):
        c, t, m = r["config"], r["throughput"], r["memory_gib"]
        st, s = t["step_time_s"], r["startup"]
        p = c["parallelism"]
        par = f"r{p['dp_replicate']}s{p['dp_shard']}t{p['tp']}"
        row = [
            r["run_name"],
            c["gpus"],
            c.get("n_layers"),
            c.get("compile"),
            c.get("activation_checkpointing"),
            par,
            f"{t['tokens_per_s_per_gpu']:.0f}",
            f"{r['mfu']['mfu_percent']:.1f}" if r["mfu"]["mfu_percent"] else "-",
            f"{st['median']:.3f}",
            f"{st['p90'] / st['median']:.3f}",
            f"{m['peak_allocated']['max']:.1f}",
            f"{m['peak_reserved']['max']:.1f}",
            (f"{100 * r['data_loading']['fraction_of_median_step']:.2f}"
             if "data_loading" in r else "-"),
            f"{s['model_initialized']['max_s']:.0f}",
            f"{s['first_step_done']['max_s'] - s['model_initialized']['max_s']:.0f}",
        ]
        print("| " + " | ".join(str(x) for x in row) + " |")


if __name__ == "__main__":
    main(sys.argv[1:])
