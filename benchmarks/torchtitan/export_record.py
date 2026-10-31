#!/usr/bin/env python
"""Build the JSON results record from milabench run data.

    python export_record.py <milabench runs dir or run dir>... [-o OUTPUT_DIR]

Milabench's run data (``<run>/<bench>.<n>.data``, JSONL) is the single source
of truth: the per-step ``rate`` events (global tokens/s, pushed by benchmate)
and the final ``summary`` event pushed by tt_native/instrument.py. The record
adds throughput statistics computed from those same rates. The official
throughput is ``throughput.global_tokens_per_s``: the median of the rates,
i.e. milabench's ``train_rate`` median.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def percentile(values, q):
    ordered = sorted(values)
    k = (len(ordered) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def _data_files(paths):
    for arg in paths:
        path = Path(arg)
        if path.is_file():
            yield path
        else:
            yield from sorted(p for p in path.rglob("*.data") if ".nolog." not in p.name)


def build_record(data_file: Path) -> dict | None:
    rates, summary, run_name = [], None, None
    for line in data_file.read_text().splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("event") == "config":
            run_name = entry["data"].get("run_name")
        if entry.get("event") != "data":
            continue
        data = entry["data"]
        if "rate" in data and data.get("task") == "train":
            rates.append(float(data["rate"]))
        if "summary" in data:
            summary = data["summary"]
    if summary is None or not rates:
        return None

    if "workload" in summary:  # milabench.pretrain.v1
        gpus = summary["parallelism"]["gpus"]
        tokens = summary["workload"]["global_batch_tokens"]
    else:  # milabench.torchtitan_native.v2
        gpus = summary["config"]["gpus"]
        tokens = summary["config"]["global_batch_tokens"]
    median = statistics.median(rates)
    step_times = [tokens / r for r in rates]
    mfu_in = summary["mfu_inputs"]
    peak = mfu_in.get("peak_flops_per_gpu")
    record = {
        **summary,
        "milabench_run": run_name,
        "bench": data_file.name.split(".")[0],
        "throughput": {
            "global_tokens_per_s": median,
            "tokens_per_s_per_gpu": median / gpus,
            "global_tokens_per_s_mean": statistics.fmean(rates),
            "rate_p10": percentile(rates, 0.10),
            "rate_p90": percentile(rates, 0.90),
            "observations": len(rates),
            "step_time_s": {
                "median": tokens / median,
                "mean": statistics.fmean(step_times),
                "stdev": statistics.pstdev(step_times),
                "min": min(step_times),
                "p10": percentile(step_times, 0.10),
                "p90": percentile(step_times, 0.90),
                "max": max(step_times),
            },
        },
        "mfu": {
            **mfu_in,
            "mfu_percent": (
                100.0 * mfu_in["flops_per_token"] * median / (peak * gpus) if peak else None
            ),
        },
    }
    record["data_loading"]["fraction_of_median_step"] = (
        record["data_loading"]["median_s_per_step"] / (tokens / median)
    )
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+")
    parser.add_argument("-o", "--output", help="directory for <run>.<bench>.json records")
    args = parser.parse_args(argv)
    for data_file in _data_files(args.paths):
        record = build_record(data_file)
        if record is None:
            continue
        if args.output:
            out = Path(args.output)
            out.mkdir(parents=True, exist_ok=True)
            target = out / f"{record['milabench_run']}.{record['bench']}.json"
            target.write_text(json.dumps(record, indent=2))
            print(target)
        else:
            print(json.dumps(record))


if __name__ == "__main__":
    main()
