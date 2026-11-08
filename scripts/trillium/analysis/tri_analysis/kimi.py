"""Kimi K3 inference (vLLM on B200): one row per client pass, per eval task and per run, from the run directories
under ``data/raw/trillium/k3-inf/runs/`` (``analyze.py collect``: ``$SCRATCH/trillium/new/k3-inf/runs/``).

The event parsing (``pass_summary`` and ``eval_summary`` events, the main node's GPU power over each pass's request
window) is copied from ``scripts/analysis/kimi_study.py`` on branch ``trillium-new`` (``read_events``, ``pass_energy``),
cut to what the report needs. The pass settings (concurrency, lengths) come from the run's own config event; the server
facts (vLLM version, MoE backend, KV cache size and use) from the server's log lines in its ``.data`` file.

Power: voir samples the GPUs of the main node only, so a pass's power is that node's mean times the node count, an
estimate (``power_nodes_measured`` = 1), as in kimi_study.py.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

# The study's entries on B200 (config/kimi_study.yaml). The -bringup, -notune and -dummy entries are tests: not read.
STUDY_ENTRIES = ("kimi-vllm-2n-tp8pp2",)
THROWAWAY = {"warmup"}  # run before the measured passes, so that kernels compile outside them

_VLLM_VERSION = re.compile(r"\bversion (\d+\.\d+\.\d+\S*)")
_MOE_BACKEND = re.compile(r"Using '(\w+)' Mxfp4 MoE backend")
_KV_TOKENS = re.compile(r"GPU KV cache size: ([\d,]+) tokens")
_KV_USE = re.compile(r"Running: (\d+) reqs, Waiting: (\d+) reqs, GPU KV cache usage: ([\d.]+)%")


def _events(path: Path):
    with open(path) as f:
        for line in f:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _pass_power(summary: dict, power: list, nodes: int) -> dict:
    """GPU power over [start_time, end_time] of the pass (kimi_study.pass_energy): the main node's mean x nodes."""
    start, end, dur = summary.get("start_time"), summary.get("end_time"), summary.get("duration_s")
    samples = [w for t, w in power if start is not None and start <= t <= end]
    if not samples or not dur:
        return {}
    watts = sum(samples) / len(samples) * nodes
    out_tokens = summary["output_throughput"] * dur
    return {"power_w_est": watts, "power_samples": len(samples), "power_nodes_measured": 1,
            "j_per_output_tok_est": watts * dur / out_tokens if out_tokens else float("nan")}


def _server_facts(run_dir: Path, entry: str, max_seqs) -> dict:
    """vLLM version, MoE backend, KV cache capacity, and the highest KV use with the server full (max_seqs running)."""
    facts = {}
    full = []
    for path in sorted(run_dir.glob(f"{entry}.*server*.data")):
        for e in _events(path):
            if e.get("event") != "line" or not isinstance(e.get("data"), str):
                continue
            text = e["data"]
            if "vllm_version" not in facts and "version" in text and (m := _VLLM_VERSION.search(text)) and "api_utils" in text:
                facts["vllm_version"] = m[1]
            if "moe_backend" not in facts and (m := _MOE_BACKEND.search(text)):
                facts["moe_backend"] = m[1]
            if "kv_cache_tokens" not in facts and (m := _KV_TOKENS.search(text)):
                facts["kv_cache_tokens"] = int(m[1].replace(",", ""))
            if (m := _KV_USE.search(text)) and max_seqs and int(m[1]) == int(max_seqs):
                full.append(float(m[3]))
    if full:
        facts["kv_use_pct_full"] = max(full)
    return facts


def read_run(run_dir: Path) -> tuple[list[dict], list[dict], dict]:
    """(passes, evals, run) of one run directory <entry>.<job>."""
    entry, _, job = run_dir.name.rpartition(".")
    passes, evals, run = [], [], {"entry": entry, "job": job, "run": run_dir.name}
    for path in sorted(run_dir.glob(f"{entry}.*.data")):
        if ".server." in path.name:
            continue
        power, config, found = [], {}, []
        for e in _events(path):
            ev, data = e.get("event"), e.get("data")
            if ev == "config":
                config = data or {}
            elif ev == "data" and isinstance(data, dict):
                if (gpus := data.get("gpudata")) and data.get("time"):
                    power.append((data["time"], sum(g.get("power", 0) for g in gpus.values())))
                elif "pass_summary" in data:
                    found.append(dict(data["pass_summary"]))
                elif (s := data.get("eval_summary")):
                    for metric, value in s["scores"].items():
                        evals.append({"entry": entry, "job": job, "task": s["task"].removeprefix("milabench_"), "metric": metric,
                                      "value": value, "stderr": s["stderr"].get(metric.replace(",", "_stderr,", 1)),
                                      "n": s.get("samples")})
        if not config:
            continue
        nodes = int(config.get("num_machines") or 1)
        gpus_per_node = int((config.get("system") or {}).get("gpu", {}).get("count") or 8)
        server = (config.get("server") or {}).get("argv") or {}
        hosts = [n["name"] for n in (config.get("system") or {}).get("nodes", [])]
        run.update(nodes=nodes, gpus=nodes * gpus_per_node, hosts=";".join(hosts), tp=gpus_per_node, pp=nodes,
                   max_num_seqs=server.get("--max-num-seqs"), max_model_len=server.get("--max-model-len"),
                   max_batched_tokens=server.get("--max-num-batched-tokens"))
        settings = {p["name"]: p.get("argv", {}) for p in (config.get("client") or {}).get("passes", []) if "name" in p}
        for s in found:
            s.pop("itl_by_context", None)
            argv = settings.get(s["name"], {})
            row = {"entry": entry, "job": job, "order": list(settings).index(s["name"]) if s["name"] in settings else None,
                   "pass": s.pop("name"), "concurrency": argv.get("--max-concurrency"),
                   "prompts": argv.get("--num-prompts"), "input_len": argv.get("--random-input-len"),
                   "output_len": argv.get("--random-output-len"), "nodes": nodes, "gpus": nodes * gpus_per_node,
                   **{k: v for k, v in s.items()}, **_pass_power(s, power, nodes)}
            row["throwaway"] = row["pass"] in THROWAWAY
            for col in ("output_throughput", "total_throughput"):
                row[f"{col}_gpu"] = row[col] / row["gpus"]
                row[f"{col}_node"] = row[col] / nodes
            passes.append(row)
    run.update(_server_facts(run_dir, entry, run.get("max_num_seqs")))
    return passes, evals, run


def build(runs_dir: Path, sacct: Path, unfinished: tuple) -> dict[str, pd.DataFrame]:
    """kimi_passes, kimi_accuracy, kimi_runs. Only jobs that have ended (the collector's sacct snapshot)."""
    states = {}
    if sacct.exists():
        s = pd.read_csv(sacct, sep="|", dtype=str)
        states = dict(zip(s.JobID, zip(s.State, s.Start, s.End)))
    passes, evals, runs = [], [], []
    for d in sorted(runs_dir.glob("*.*")) if runs_dir.is_dir() else []:
        entry, _, job = d.name.rpartition(".")
        if entry not in STUDY_ENTRIES or not job.isdigit():
            continue
        state, start, end = states.get(job, ("RUNNING", None, None))
        if state.startswith(unfinished):
            continue  # a job still running is not a failed run: left out until it ends
        p, e, r = read_run(d)
        r.update(job_state=state, start=start, end=end)
        passes += p
        evals += e
        runs.append(r)
    passes = pd.DataFrame(passes)
    if len(passes):
        passes = passes.sort_values(["job", "order"]).reset_index(drop=True)
    return {"kimi_passes": passes, "kimi_accuracy": pd.DataFrame(evals), "kimi_runs": pd.DataFrame(runs)}
