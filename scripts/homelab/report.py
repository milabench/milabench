#!/usr/bin/env python
"""Homelab run report: one row per benchmark (and per client pass) of a run, for the weekly / monthly report.

    python scripts/homelab/report.py $MILABENCH_BASE/runs/<run> [--out DIR]

Writes ``<out>/report.json`` (every row, every field), ``report.csv`` and ``report.md`` (the summary tables).
Per row:

  model, engine, disk_gb       models it serves / loads (server argv, client --model, ComfyUI --model) and
                               their size in the Hugging Face cache (HF_HUB_CACHE)
  status                       ok / failed (return code) / killed (memguard: out of memory)
  throughput                   serving passes: output / total tok/s, request/s, TTFT / TPOT / ITL p50 and p99
                               (pass_summary); others: their rate metric (img/s, items/s)
  accuracy                     QualityBench / lm-eval tasks (eval_summary), SWE-bench agent (agent_summary)
  memory                       peak used memory over the bench (milabench's GPU samples; on unified memory
                               this is the machine's memory), vLLM's weights / KV cache / max context from the
                               server log (torch allocator), the node's peak (nodemon)
  power                        mean / peak GPU power, peak temperature, energy (NVML energy counter from
                               nodemon when present, else power integrated over the samples), tokens per joule
Stdlib only.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import statistics
from collections import defaultdict

# --- reading ----------------------------------------------------------------------------------------


def events(path):
    with open(path, errors="replace") as f:
        for line in f:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def model_ids(cfg: dict) -> list[str]:
    """HF repos a bench loads: positional server model, --model / --model-path, drafters, ComfyUI files."""
    ids = []

    def add(value):
        if isinstance(value, str) and re.fullmatch(r"[\w.-]+/[\w.-]+", value) and value not in ids:
            ids.append(value)

    for section in ("server", "client"):
        argv = (cfg.get(section) or {}).get("argv") or {}
        for key, value in argv.items():
            if value is True and "/" in key and not key.startswith("-"):
                add(key)
            elif key in ("--model", "--model-path", "--tokenizer", "--repo", "--speculative-draft-model-path"):
                add(value)
            elif key == "--speculative-config":
                try:
                    add(json.loads(value).get("model"))
                except (json.JSONDecodeError, AttributeError):
                    pass
    # single-pack benches: --model is an HF id, or ComfyUI's dest:repo:file (one or a list)
    model = (cfg.get("argv") or {}).get("--model")
    for spec in [model] if isinstance(model, str) else model or []:
        add(spec.split(":")[1] if isinstance(spec, str) and spec.count(":") >= 2 else spec)
    return ids


def hf_cache() -> str:
    return os.environ.get("HF_HUB_CACHE") or os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")


_disk_cache = {}


def disk_gb(repo: str) -> float | None:
    if repo in _disk_cache:
        return _disk_cache[repo]
    folder = os.path.join(hf_cache(), "models--" + repo.replace("/", "--"), "blobs")
    size = None
    if os.path.isdir(folder):
        size = sum(os.path.getsize(os.path.join(folder, f)) for f in os.listdir(folder)) / 1e9
    _disk_cache[repo] = size
    return size


# vLLM's own accounting of the memory it reserved (torch allocator, per engine core)
VLLM_PATTERNS = {
    "weights_gib": re.compile(r"Model loading took ([\d.]+) Gi?B"),
    "kv_cache_gib": re.compile(r"Available KV cache memory: ([\d.]+) Gi?B"),
    "kv_cache_tokens": re.compile(r"GPU KV cache size: ([\d,]+) tokens"),
    "max_concurrency_x": re.compile(r"Maximum concurrency for [\d,]+ tokens per request: ([\d.]+)x"),
    "cuda_graph_gib": re.compile(r"Graph capturing finished in .* took ([\d.]+) Gi?B"),
}


def server_memory(path: str) -> dict:
    out = {}
    for e in events(path):
        if e.get("event") != "line" or not isinstance(e.get("data"), str):
            continue
        for key, pattern in VLLM_PATTERNS.items():
            if key not in out and (m := pattern.search(e["data"])):
                out[key] = float(m.group(1).replace(",", ""))
    return out


def nodemon_rows(run_dir: str, kind: str) -> list[dict]:
    rows = []
    for path in glob.glob(os.path.join(run_dir, "nodemon", "*", "nodes", f"*.{kind}.tsv")):
        with open(path) as f:
            for row in csv.DictReader(f, delimiter="\t"):
                rows.append(row)
    rows.sort(key=lambda r: int(r["time_ms"]))
    return rows


def window(rows, start, end):
    lo, hi = start * 1000, end * 1000
    return [r for r in rows if lo <= int(r["time_ms"]) <= hi]


def num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# --- one bench / pass ---------------------------------------------------------------------------------


def read_bench(path: str, gpu_rows, mem_rows) -> dict | None:
    cfg = meta = None
    start = end = None
    rc = None
    rates, samples = [], []
    passes, evals, agents = [], [], []
    for e in events(path):
        kind, data = e.get("event"), e.get("data")
        if kind == "config":
            cfg = data
        elif kind == "meta":
            meta = data
        elif kind == "start":
            start = data.get("time")
        elif kind == "end":
            end, rc = data.get("time"), data.get("return_code")
        elif kind == "data" and isinstance(data, dict):
            if "gpudata" in data:
                for gpu in data["gpudata"].values():
                    samples.append((data.get("time"), gpu))
            if "pass_summary" in data:
                passes.append(data["pass_summary"])
            if "eval_summary" in data:
                evals.append(data["eval_summary"])
            if "agent_summary" in data:
                agents.append(data["agent_summary"])
            for key in ("rate", "img_per_s"):
                if key in data and isinstance(data[key], (int, float)) and data.get("task", "train") == "train":
                    rates.append((key, data[key], data.get("units")))
    if cfg is None:
        return None
    tags = cfg.get("tag") or []
    if "server" in tags:
        return None  # the server's own file: read for its log only

    row = {
        "bench": cfg.get("name"), "file": os.path.basename(path), "tags": sorted(set(cfg.get("tags", []))),
        "group": cfg.get("group"), "engine": (cfg.get("server") or {}).get("backend") or cfg.get("group"),
        "models": model_ids(cfg),
        "gpu": next(iter(((meta or {}).get("accelerators") or {}).get("gpus", {}).values()), {}).get("product"),
        "start": start, "end": end, "duration_s": (end - start) if start and end else None,
        "return_code": rc, "status": "ok" if rc == 0 else ("missing-end" if rc is None else "failed"),
    }
    row["disk_gb"] = round(sum(d for m in row["models"] if (d := disk_gb(m))), 2) or None

    # throughput
    if passes:
        p = passes[-1]
        row.update({k: p.get(k) for k in ("output_throughput", "total_throughput", "request_throughput",
                                          "ttft_ms_p50", "ttft_ms_p99", "tpot_ms_p50", "tpot_ms_p99", "itl_ms_p50",
                                          "e2el_ms_p50", "completed", "failed", "mean_input_len", "mean_output_len")})
        row["window"] = (p.get("start_time"), p.get("end_time"))
    elif rates:
        values = [v for _, v, _ in rates]
        row["rate_metric"] = rates[0][0] if rates[0][0] != "rate" else (rates[0][2] or "rate")
        row["rate_mean"] = statistics.fmean(values)
        row["rate_median"] = statistics.median(values)
    # accuracy
    for ev in evals:
        task = ev.get("task")
        key = f"acc.{ev.get('suite', 'lmeval')}.{task}" + (f".think-{ev['thinking']}" if ev.get("thinking") else "")
        scores = ev.get("scores") or {}
        score = scores.get("acc") if "acc" in scores else next((v for k, v in scores.items() if "acc" in k or "exact" in k), None)
        if score is not None:
            row[key] = score
            if ev.get("ci95"):
                row[key + ".ci95"] = ev["ci95"]
    for ag in agents:
        row.update({f"swebench.{k}": ag.get(k) for k in ("samples", "solved", "submitted", "step_limit", "looped",
                                                        "errors", "agent_hours", "solved_per_hour", "mean_steps")})
        row["swebench.resolved"] = (ag.get("scores") or {}).get("resolved")
        row["swebench.ci95"] = ag.get("ci95")

    # memory and power over the bench (or the pass' request window)
    lo, hi = row.get("window") or (start, end)
    if lo and hi:
        in_window = [g for t, g in samples if t and lo <= t <= hi] or [g for _, g in samples]
        mem = [g["memory"][0] for g in in_window if isinstance(g.get("memory"), list) and g["memory"][0] is not None]
        power = [g["power"] for g in in_window if isinstance(g.get("power"), (int, float))]
        temp = [g["temperature"] for g in in_window if isinstance(g.get("temperature"), (int, float))]
        if mem:
            row["mem_peak_gib"] = round(max(mem) / 1024, 2)
            row["mem_total_gib"] = round(in_window[0]["memory"][1] / 1024, 1) if in_window[0]["memory"][1] else None
        if power:
            row["power_mean_w"] = round(statistics.fmean(power), 1)
            row["power_peak_w"] = round(max(power), 1)
        if temp:
            row["temp_peak_c"] = max(temp)
        gpu = window(gpu_rows, lo, hi)
        energy = [num(r.get("energy_j")) for r in gpu if num(r.get("energy_j")) is not None]
        if len(energy) >= 2:
            row["energy_j"] = round(energy[-1] - energy[0], 1)
            row["energy_source"] = "nvml-counter"
        elif power:
            row["energy_j"] = round(statistics.fmean(power) * (hi - lo), 1)
            row["energy_source"] = "power-integral"
        node_mem = [num(r.get("mem_used_gib")) for r in window(mem_rows, lo, hi) if num(r.get("mem_used_gib")) is not None]
        if node_mem:
            row["node_mem_peak_gib"] = round(max(node_mem), 2)
        if row.get("energy_j") and row.get("output_throughput") and passes:
            p = passes[-1]
            tokens = p.get("output_throughput", 0) * p.get("duration_s", 0)
            row["output_tokens_per_joule"] = round(tokens / row["energy_j"], 3) if row["energy_j"] else None
    return row


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--out", default=None, help="default: <run_dir>/report")
    args = parser.parse_args(argv)
    out = args.out or os.path.join(args.run_dir, "report")
    os.makedirs(out, exist_ok=True)

    gpu_rows, mem_rows = nodemon_rows(args.run_dir, "gpu"), nodemon_rows(args.run_dir, "mem")
    files = sorted(glob.glob(os.path.join(args.run_dir, "*.data")))

    # server logs: vLLM's memory accounting, attached to every row of its benchmark
    server_mem = {}
    for path in files:
        first = next(events(path), {})
        cfg = first.get("data") if first.get("event") == "config" else None
        if cfg and "server" in (cfg.get("tag") or []):
            server_mem[cfg["tag"][0]] = server_memory(path)

    kills = [json.loads(line) for line in open(os.path.join(args.run_dir, "memguard.log"))] \
        if os.path.exists(os.path.join(args.run_dir, "memguard.log")) else []
    kills = [k for k in kills if k.get("event") == "kill"]

    rows = []
    for path in files:
        row = read_bench(path, gpu_rows, mem_rows)
        if row is None:
            continue
        bench = row["bench"] or ""
        for server, mem in server_mem.items():
            if bench == server or bench.startswith(server + "."):
                row.update({f"vllm.{k}": v for k, v in mem.items()})
        if row["start"] and row["end"] and any(row["start"] <= k["time"] <= row["end"] + 5 for k in kills):
            row["status"] = "killed-oom"
        rows.append(row)

    json.dump(rows, open(os.path.join(out, "report.json"), "w"), indent=1)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("bench", "status"), k))
    with open(os.path.join(out, "report.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in r.items()})
    write_markdown(rows, kills, os.path.join(out, "report.md"), args.run_dir)
    print(f"{len(rows)} rows -> {out}/report.{{json,csv,md}}")


def fmt(v, digits=1):
    if v is None or v == "":
        return ""
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def write_markdown(rows, kills, path, run_dir):
    lines = [f"# Homelab run `{os.path.basename(os.path.normpath(run_dir))}`", ""]
    status = defaultdict(int)
    for r in rows:
        status[r["status"]] += 1
    lines += [f"{len(rows)} results: " + ", ".join(f"{k} {v}" for k, v in sorted(status.items())), ""]

    serving = [r for r in rows if r.get("output_throughput") is not None]
    if serving:
        lines += ["## Serving (per pass)", "",
                  "| bench | disk GB | out tok/s | TTFT p50 ms | TPOT p50 ms | mem peak GiB | weights GiB | KV GiB | power W | energy kJ | tok/J |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in serving:
            lines.append("| " + " | ".join([r["bench"], fmt(r.get("disk_gb")), fmt(r.get("output_throughput")),
                                              fmt(r.get("ttft_ms_p50"), 0), fmt(r.get("tpot_ms_p50")),
                                              fmt(r.get("mem_peak_gib")), fmt(r.get("vllm.weights_gib")),
                                              fmt(r.get("vllm.kv_cache_gib")), fmt(r.get("power_mean_w")),
                                              fmt((r.get("energy_j") or 0) / 1000 or None), fmt(r.get("output_tokens_per_joule"), 2)]) + " |")
        lines.append("")

    accuracy = [r for r in rows if any(k.startswith(("acc.", "swebench.resolved")) for k in r)]
    if accuracy:
        tasks = sorted({k for r in accuracy for k in r if k.startswith("acc.") and not k.endswith(".ci95")})
        lines += ["## Accuracy", "", "| bench | " + " | ".join(t.removeprefix("acc.") for t in tasks) + " | SWE-bench solved | looped | step limit | solved/h |",
                  "|---|" + "---|" * (len(tasks) + 4)]
        for r in accuracy:
            sw = f"{r['swebench.solved']}/{r['swebench.samples']}" if r.get("swebench.samples") else ""
            lines.append("| " + " | ".join([r["bench"], *[fmt(100 * r[t]) if r.get(t) is not None else "" for t in tasks],
                                              sw, fmt(r.get("swebench.looped")), fmt(r.get("swebench.step_limit")),
                                              fmt(r.get("swebench.solved_per_hour"), 2)]) + " |")
        lines.append("")

    other = [r for r in rows if r.get("rate_mean") is not None]
    if other:
        lines += ["## Other benchmarks", "", "| bench | metric | mean | disk GB | mem peak GiB | power W | temp °C | energy kJ |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in other:
            lines.append("| " + " | ".join([r["bench"], str(r.get("rate_metric")), fmt(r.get("rate_mean"), 3), fmt(r.get("disk_gb")),
                                              fmt(r.get("mem_peak_gib")), fmt(r.get("power_mean_w")), fmt(r.get("temp_peak_c"), 0),
                                              fmt((r.get("energy_j") or 0) / 1000 or None)]) + " |")
        lines.append("")

    failed = [r for r in rows if r["status"] != "ok"]
    if failed or kills:
        lines += ["## Failures", ""]
        for r in failed:
            lines.append(f"- `{r['bench']}`: {r['status']} (return code {r.get('return_code')})")
        for k in kills:
            lines.append(f"- memguard: {k.get('reason')} — killed {', '.join(v['cmd'][:60] for v in k.get('victims', [])[:2])}")
        lines.append("")
    open(path, "w").write("\n".join(lines))


if __name__ == "__main__":
    main()
