"""Kimi K3 inference (vLLM on B200): one row per client pass, per eval task and per run, from the run directories
under ``data/raw/trillium/k3-inf/runs/`` (``analyze.py collect``: ``$SCRATCH/trillium/new/k3-inf/runs/``), and, in tables of
their own (``kimi_xn_*``), the experts over 2 nodes (K3-XN-01, EP16 = DP2 x TP8 with expert parallelism, one job per all-to-all
backend) from ``data/raw/trillium/k3-xn/runs/``.

The event parsing (``pass_summary`` and ``eval_summary`` events, the main node's GPU power over each pass's request
window) is copied from ``scripts/analysis/kimi_study.py`` on branch ``trillium-new`` (``read_events``, ``pass_energy``),
cut to what the report needs. The pass settings (concurrency, lengths) come from the run's own config event; the server
facts (vLLM version, MoE backend, KV cache size and use) from the server's log lines in its ``.data`` file.

Power: voir samples the GPUs of the main node only, so a pass's power is that node's mean times the node count, an
estimate (``power_nodes_measured`` = 1), as in kimi_study.py.

Zero windows: vLLM logs its prompt and generation throughput every 10 s; a pass's ``zero_windows`` counts the logged
intervals overlapping it with requests running and no token at all, prompt or output. The log's clock is Trillium's
local time (America/Toronto), the passes' Unix time.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

# The study's entries on B200 (config/kimi_study.yaml). The -bringup, -notune and -dummy entries are tests: not read.
# The 4-node bring-up is kimi-vllm-4n-tp8pp4-bringup (1264956): not read; the study on 4 nodes is one job, 1265338.
STUDY_ENTRIES = ("kimi-vllm-2n-tp8pp2", "kimi-vllm-4n-tp8pp4")
# K3-XN-01: EP16 on 2 nodes, all-gather / reduce-scatter (1271952) and DeepEP V2 (1271953) all-to-all, keyed on job; the bring-ups,
# EP32 and P/D entries beside them in k3-xn/runs are not read.
XN_ENTRIES = ("kimi-vllm-2n-dp2tp8-ep16", "kimi-vllm-2n-dp2tp8-ep16-deepep")
XN_JOBS = ("1271952", "1271953")
# run before the measured passes, so that kernels compile outside them; EP16 has a second one (64 requests of 1k in, 256 out)
THROWAWAY = {"warmup", "warmup-mixed"}

_VLLM_VERSION = re.compile(r"\bversion (\d+\.\d+\.\d+\S*)")
_MOE_BACKEND = re.compile(r"Using '(\w+)' Mxfp4 MoE backend")
_KV_TOKENS = re.compile(r"GPU KV cache size: ([\d,]+) tokens")
_KV_USE = re.compile(r"Running: (\d+) reqs, Waiting: (\d+) reqs, GPU KV cache usage: ([\d.]+)%")
_WEIGHTS = re.compile(r"Model loading took ([\d.]+) GiB")
_KV_GIB = re.compile(r"Available KV cache memory: ([\d.]+) GiB")
_ALL2ALL = re.compile(r"Using (\w+) all2all manager")
_ENGINE_UP = re.compile(r"INFO (\d\d-\d\d \d\d:\d\d:\d\d) \[core\.py:\d+\] init engine \(.*\) took ([\d.]+) s")
_THROUGHPUT = re.compile(r"INFO (\d\d-\d\d \d\d:\d\d:\d\d) \[loggers\.py:\d+\] Engine \d+: Avg prompt throughput: ([\d.]+) tokens/s, "
                         r"Avg generation throughput: ([\d.]+) tokens/s, Running: (\d+) reqs")
LOG_TZ = ZoneInfo("America/Toronto")
LOG_INTERVAL_S = 10


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


def _server_facts(run_dir: Path, entry: str, max_seqs, year: int) -> tuple[dict, list]:
    """vLLM version, MoE backend, KV cache capacity (tokens per engine, GiB per GPU), weights per GPU, the all-to-all manager, the
    time the last engine was up (Unix, end of its "init engine" line), and the highest KV use with the server full (max_seqs
    running); and the zero windows: Unix end time of each logged 10 s interval with requests running and no prompt or output
    token."""
    facts = {}
    full, zero = [], set()
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
            if "weights_gib" not in facts and (m := _WEIGHTS.search(text)):
                facts["weights_gib"] = float(m[1])
            if "kv_gib" not in facts and (m := _KV_GIB.search(text)):
                facts["kv_gib"] = float(m[1])
            if "all2all_manager" not in facts and (m := _ALL2ALL.search(text)):
                facts["all2all_manager"] = m[1]
            if m := _ENGINE_UP.search(text):
                up = datetime.strptime(f"{year}-{m[1]}", "%Y-%m-%d %H:%M:%S").replace(tzinfo=LOG_TZ).timestamp()
                facts["engine_ready"] = max(facts.get("engine_ready", up), up)
            if (m := _KV_USE.search(text)) and max_seqs and int(m[1]) == int(max_seqs):
                full.append(float(m[3]))
            if (m := _THROUGHPUT.search(text)) and int(m[4]) > 0 and float(m[2]) == 0 and float(m[3]) == 0:
                zero.add(datetime.strptime(f"{year}-{m[1]}", "%Y-%m-%d %H:%M:%S").replace(tzinfo=LOG_TZ).timestamp())
    if full:
        facts["kv_use_pct_full"] = max(full)
    return facts, sorted(zero)


def _questions(run_dir: Path, entry: str, job: str) -> list[dict]:
    """One row per eval question (lm-eval samples files): right or not, and whether an answer was parsed at all."""
    rows = []
    for path in sorted(run_dir.glob(f"{entry}.eval.lm_eval/*/samples_milabench_*.jsonl")):
        task = re.sub(r"_\d{4}-\d\d-\d\dT.*$", "", path.stem.removeprefix("samples_milabench_"))
        for e in _events(path):
            rows.append({"entry": entry, "job": job, "task": task, "doc_hash": e["doc_hash"], "correct": float(e[e["metrics"][0]]),
                         "parsed": e["filtered_resps"][0] != "[invalid]"})
    return rows


def read_run(run_dir: Path) -> tuple[list[dict], list[dict], dict, list[dict]]:
    """(passes, evals, run, questions) of one run directory <entry>.<job>."""
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
        server_config = config.get("server") or {}
        server = server_config.get("argv") or {}
        hosts = [n["name"] for n in (config.get("system") or {}).get("nodes", [])]
        if server_config.get("multinode") == "dp":  # one engine per node (TP over its GPUs); the experts spread over all of them
            dp, pp = nodes, 1
            layout = f"dp{nodes}tp{gpus_per_node}" + (f"ep{nodes * gpus_per_node}" if server.get("--enable-expert-parallel") else "")
        else:  # one engine, a pipeline stage per node
            dp, pp = 1, nodes
            layout = f"tp{gpus_per_node}pp{nodes}"
        run.update(nodes=nodes, gpus=nodes * gpus_per_node, hosts=";".join(hosts), tp=gpus_per_node, pp=pp, dp=dp, layout=layout,
                   all2all=server.get("--all2all-backend"), max_num_seqs=server.get("--max-num-seqs"), max_model_len=server.get("--max-model-len"),
                   max_batched_tokens=server.get("--max-num-batched-tokens"))
        settings = {p["name"]: p.get("argv", {}) for p in (config.get("client") or {}).get("passes", []) if "name" in p}
        for s in found:
            s.pop("itl_by_context", None)
            argv = settings.get(s["name"], {})
            row = {"entry": entry, "job": job, "order": list(settings).index(s["name"]) if s["name"] in settings else None,
                   "pass": s.pop("name"), "concurrency": argv.get("--max-concurrency"),
                   "prompts": argv.get("--num-prompts"), "input_len": argv.get("--random-input-len"),
                   "output_len": argv.get("--random-output-len"), "nodes": nodes, "layout": layout, "gpus": nodes * gpus_per_node,
                   **{k: v for k, v in s.items()}, **_pass_power(s, power, nodes)}
            row["throwaway"] = row["pass"] in THROWAWAY
            for col in ("output_throughput", "total_throughput"):
                row[f"{col}_gpu"] = row[col] / row["gpus"]
                row[f"{col}_node"] = row[col] / nodes
            passes.append(row)
    year = datetime.fromtimestamp(passes[0]["start_time"], LOG_TZ).year if passes else datetime.now().year
    facts, zero = _server_facts(run_dir, entry, run.get("max_num_seqs"), year)
    run.update(facts)
    for row in passes:  # logged intervals (t - 10 s, t] overlapping the pass
        row["zero_windows"] = sum(row["start_time"] < t and t - LOG_INTERVAL_S < row["end_time"] for t in zero)
    return passes, evals, run, _questions(run_dir, entry, job)


def build(runs_dir: Path, sacct: Path, unfinished: tuple, entries=STUDY_ENTRIES, jobs=None, prefix="kimi") -> dict[str, pd.DataFrame]:
    """<prefix>_passes, _accuracy, _runs, _questions: the runs of these entries (and, if given, only these jobs). Only jobs that have
    ended (the collector's sacct snapshot). engine_up_min: from the job's start to the last engine up, measured."""
    states = {}
    if sacct.exists():
        s = pd.read_csv(sacct, sep="|", dtype=str)
        states = dict(zip(s.JobID, zip(s.State, s.Start, s.End)))
    passes, evals, runs, questions = [], [], [], []
    for d in sorted(runs_dir.glob("*.*")) if runs_dir.is_dir() else []:
        entry, _, job = d.name.rpartition(".")
        if entry not in entries or not job.isdigit() or (jobs is not None and job not in jobs):
            continue
        state, start, end = states.get(job, ("RUNNING", None, None))
        if state.startswith(unfinished):
            continue  # a job still running is not a failed run: left out until it ends
        p, e, r, q = read_run(d)
        r.update(job_state=state, start=start, end=end)
        if r.get("engine_ready") and start:
            r["engine_up_min"] = (r["engine_ready"] - datetime.fromisoformat(start).replace(tzinfo=LOG_TZ).timestamp()) / 60
        passes += p
        evals += e
        runs.append(r)
        questions += q
    passes = pd.DataFrame(passes)
    if len(passes):
        passes = passes.sort_values(["job", "order"]).reset_index(drop=True)
    return {f"{prefix}_passes": passes, f"{prefix}_accuracy": pd.DataFrame(evals), f"{prefix}_runs": pd.DataFrame(runs),
            f"{prefix}_questions": pd.DataFrame(questions)}
