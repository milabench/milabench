#!/usr/bin/env python
"""Homelab run report as one self-contained HTML page (no external assets), from report.py's rows.

    python scripts/homelab/report.py <run dir> --out <run dir>/report
    python scripts/homelab/report_html.py <run dir> --out report.html

Rerun as the run goes: the page shows what has finished, what runs now, and what failed.
"""

from __future__ import annotations
import re

import argparse
import datetime
import glob
import html
import json
import os
import subprocess
from collections import defaultdict

PASS_ORDER = ["single", "conc1", "fill4k", "conc4", "conc8", "conc16", "conc32", "conc64", "conc256", "ctx32k-1", "ctx32k-4", "ctx60k-4"]
TASKS = [("gsm8k", "GSM8K"), ("ifeval", "IFEval"), ("mmlu_pro", "MMLU-Pro"), ("humaneval", "HumanEval+"),
         ("mbpp", "MBPP+"), ("lcb", "LiveCode"), ("lcb_full", "LiveCode full"), ("bfcl", "BFCL")]


def esc(v) -> str:
    return html.escape(str(v))


def fmt(v, digits=1, scale=1.0, suffix="") -> str:
    if v is None or v == "":
        return '<span class="na">–</span>'
    if isinstance(v, (int, float)):
        return f"{v * scale:,.{digits}f}{suffix}"
    return esc(v)


def split(bench: str) -> tuple[str, str | None]:
    """bench -> (model bench, pass) for serving passes."""
    if "." in bench:
        head, tail = bench.rsplit(".", 1)
        return head, tail
    return bench, None


def per_user(r: dict | None):
    """Output tokens per second one user sees: 1000 / TPOT p50 (decode speed of one stream at the pass's concurrency)."""
    tpot = (r or {}).get("tpot_ms_p50")
    return 1000 / tpot if tpot else None


# what each serving pass is for; prompt / output lengths and request counts are read from the results
PASS_INFO = {
    "single": (1, "One user, short prompt. The speed one person sees chatting or running a single agent: "
                  "TTFT is the wait before text starts, tok/s/user the reading speed after that."),
    "conc1": (1, "One user, 1k tokens out (the 32 GB and llama.cpp benches). Same question as single, with longer answers."),
    "fill4k": (1, "One user, 4× longer prompt. Compared with single: what a longer context (a file pasted in, "
                  "a long conversation) costs in prefill (TTFT) and in decoding speed."),
    "conc8": (8, "8 requests always in flight: a small team or an agent running tools in parallel. "
                 "Total tok/s shows how much batching gains; tok/s/user how much each user slows down."),
    "conc16": (16, "16 requests in flight."),
    "conc32": (32, "32 requests in flight: the most most servers here are configured for (max-num-seqs 32). "
                   "Roughly the machine's peak output throughput for this model."),
    "conc64": (64, "64 requests in flight."),
    "conc256": (256, "256 requests in flight (server allows 256): peak throughput for batch work."),
    "ctx32k-1": (1, "Agent-shaped: one session with a 32k-token shared context, continued by short new turns; "
                    "with prefix caching only the new turn is prefilled."),
    "ctx32k-4": (4, "Agent-shaped: 4 sessions, each its own 32k-token context, turns interleaved."),
    "ctx60k-4": (4, "Agent-shaped: 4 sessions, each its own 60k-token context, turns interleaved."),
}


def _span(values, digits=0) -> str:
    values = sorted({round(v) for v in values if v})
    if not values:
        return "–"
    lo, hi = values[0], values[-1]
    return f"{lo:,}" if lo == hi else f"{lo:,}–{hi:,}"


def pass_definitions(models: list[dict], passes: list[str]) -> str:
    h = ["<h2>What each pass measures</h2><div class=wrap><table><tr><th>pass</th><th>requests in flight</th>"
         "<th>prompt tokens</th><th>output tokens</th><th>requests measured</th><th>what it tells you</th></tr>"]
    for p in passes:
        rs = [m["passes"][p] for m in models if p in m["passes"]]
        conc, text = PASS_INFO.get(p, (None, ""))
        h.append(f"<tr><td>{esc(p)}</td><td>{esc(conc or '–')}</td><td>{_span(r.get('mean_input_len') for r in rs)}</td>"
                 f"<td>{_span(r.get('mean_output_len') for r in rs)}</td><td>{_span(r.get('completed') for r in rs)}</td>"
                 f"<td class=left>{esc(text)}</td></tr>")
    h.append("</table></div><div class=note>Every pass: prompts cut from real conversations (OpenHermes), not random tokens; "
             "each request generates the output length shown (end-of-sequence ignored; llama.cpp can still stop a little early), so passes compare like for like. "
             "Warm-up requests run first and are not counted. llama.cpp runs one benchmark per concurrency (its server's slots); "
             "they share their model's row, the 1-slot one as single. Fixed-length outputs flatter speculative decoding (MTP, DFlash): "
             "the draft keeps guessing on text the model would have ended.</div>")
    return "\n".join(h)


# thinking-on runs left out of the report, with why (bench prefix -> reason)
THINK_INVALID: dict[str, str] = {}

# thinking-on runs shown with a caveat (bench prefix -> what to know)
THINK_CAVEAT = {
    "homelab-vllm-gemma4-26b": "Gemma 4 26B thinking on loops under greedy decoding (temperature 0, which QualityBench uses for "
                           "every model): on checkable constraints (\"at least 30 sentences\") it recounts its draft until "
                           "the token cap, with no answer. The same prompt finishes in ~5k tokens with Google's recommended "
                           "sampling (temperature 1.0, top_p 0.95). Its thinking-on scores measure that loop, which a "
                           "coding agent at temperature 0 would hit too. Gemma 4 E4B does not loop (no reply at the cap)",
    "homelab-vllm-exaone45-33b": "Too slow with thinking on: EXAONE 4.5 33B reasons for ~8.5k tokens per reply at ~11 tokens/s "
                                 "per request, so IFEval alone ran past 75 min (estimate: 12) and hour-long replies were "
                                 "retried from scratch (2026-11-16). It is rerun with the 1 h time-to-solution window: a "
                                 "reply not back within the hour scores as wrong (the timed-out count)",
    "homelab-vllm-qwen36-27b": "Too slow with thinking on: Qwen3.6-27B (dense) took 209 min on MBPP+ and 53 on HumanEval+, "
                               "42 replies ran past an hour and were retried from scratch, and its LiveCodeBench task "
                               "failed. That run predates the 1 h time-to-solution window (2026-11-16); its scores are "
                               "from the replies that came back",
    "homelab-vllm-inkling2-mini-dspark": "Inkling 2 Mini always reasons: its \"off\" is the same model at reasoning effort "
                                         "none, its \"on\" effort 0.9, both with a 32k reply cap (thinking.yaml). The "
                                         "8k-cap run of text.yaml is superseded by these",
    "homelab-vllm-nemotron3-super-120b": "Thinking on, 8 of its 18 LiveCodeBench answers ran past the 1 h time-to-solution "
                                         "window (about 10 tokens/s per request with 16 requests in flight) and score as "
                                         "wrong: the window's rule, applied from 2026-11-16. Its thinking-on coding score "
                                         "measures that, not what one user waiting longer would get",
}


LLAMACPP_BENCH = re.compile(r"^(?P<model>homelab-llamacpp-.+?)(?:-(?:native|cuda|hip|vulkan|sycl))?-(?:conc(?P<conc>\d+)|quality)$")


def model_key(bench: str) -> tuple[str, str | None]:
    """bench -> (model row, pass). llama.cpp runs one bench per concurrency (and one for accuracy): they share
    the model's row, their 1-stream bench is the `single` pass (same workload since the unified pass set)."""
    head, part = split(bench)
    if head.endswith("-effort-none-lcbfull"):  # an always-reasoning model at effort none: its "off" (model_rows)
        return head.removesuffix("-effort-none-lcbfull"), "lcbfull-off"
    if head.endswith("-lcbfull"):  # lcb-full.yaml: the full LiveCodeBench of a model in text.yaml, its row
        return head.removesuffix("-lcbfull").removesuffix("-think"), part  # -think-lcbfull: the same, thinking on
    if head.endswith("-think"):  # thinking.yaml: the same model's accuracy, thinking on
        return head.removesuffix("-think"), "quality-think" if part == "quality" else part
    if head.endswith("-effort-none"):  # thinking.yaml: an always-reasoning model at reasoning effort none, its "off"
        return head.removesuffix("-effort-none"), "quality-off" if part == "quality" else part
    if part is None and (m := LLAMACPP_BENCH.match(head)):
        conc = m.group("conc")
        return m.group("model"), (None if conc is None else "single" if conc == "1" else f"conc{conc}")
    return head, part


def model_rows(rows: list[dict], exclude: set[str] = frozenset()) -> list[dict]:
    """exclude: benches that are not this machine's results (quarantined as unsupported: e.g. the Vulkan twin of a
    llama.cpp bench, which shares its model's row)."""
    models = defaultdict(lambda: {"passes": {}, "status": set(), "benches": set()})
    for r in rows:
        if split(r["bench"])[0] in exclude:
            continue
        if any(split(r["bench"])[0].startswith(p) and split(r["bench"])[0].endswith("-think") for p in THINK_INVALID):
            continue  # thinking.yaml runs whose scores measure a harness problem, not the model
        head, part = model_key(r["bench"])
        scored = any(k.startswith(("acc.", "ppl.", "swebench.")) for k in r)
        if part is None and not r.get("output_throughput") and not scored:
            continue
        m = models[head]
        m["bench"] = head
        m["status"].add(r["status"])
        m["benches"].add(split(r["bench"])[0])
        for key in ("models", "what", "disk_gb", "vllm.weights_gib", "vllm.kv_cache_gib", "vllm.kv_cache_tokens"):
            if r.get(key) is not None and (key != "what" or "what" not in m or r.get("output_throughput") is not None):
                m[key] = r[key]
        if r.get("output_throughput") is not None:
            m["passes"][part or "default"] = r
        # thinking.yaml's runs (32k cap) over a text.yaml run of an always-reasoning model (8k), whatever the order
        rank = 2 if part in ("quality-think", "quality-off", "lcbfull-off") else 1
        for key, value in r.items():
            if key.startswith(("acc.", "ppl.", "swebench.")):
                if part in ("quality-off", "lcbfull-off"):
                    key = key.replace(".think-on", ".think-off")
                if m.setdefault("_rank", {}).get(key, 0) > rank:
                    continue
                m["_rank"][key] = rank
                m[key] = value
        for key in ("mem_peak_gib", "temp_peak_c", "power_peak_w"):
            if r.get(key) is not None:
                m[key] = max(m.get(key) or 0, r[key])
        # a run in progress has no cost yet; nor has a rerun appended to a finished run's file (its last end is the
        # earlier run's, before the new start: a negative duration)
        if part in ("quality", "quality-think", "quality-off") and (r.get("duration_s") or 0) > 0 and r.get("status") == "ok":
            mode = "off" if part == "quality-off" else "on" if part == "quality-think" or thinking_only(r) else "off"
            if m.setdefault("_rank", {}).get(f"cost.{mode}", 0) <= rank:
                m["_rank"][f"cost.{mode}"] = rank
                m[f"cost.{mode}"] = (r["duration_s"], r.get("energy_j") if (r.get("energy_j") or 0) > 0 else None)
        if r.get("energy_j") and part not in ("quality-think", "quality-off"):  # thinking.yaml's extra run: in its own section
            m["energy_j"] = (m.get("energy_j") or 0) + r["energy_j"]
    for m in models.values():
        m.pop("_rank", None)
    return [m for _, m in sorted(models.items(), key=lambda kv: (not kv[0].startswith("homelab-vllm"), kv[0]))]


def _client_side(error: str) -> bool:
    return any(k in error for k in ("Endpoint is not ready", "Cannot connect to host", "Never received a valid chunk"))


def server_error(run_dir: str, rows: list[dict]) -> str | None:
    """When the clients only saw a dead endpoint: the last error line of the model's server log."""
    if not all(_client_side(r.get("error") or "Endpoint is not ready") for r in rows):
        return None
    for r in rows:
        path = os.path.join(run_dir, f"{split(r['bench'])[0]}.0.server.nolog.data")
        if not os.path.exists(path):
            continue
        lines = []
        for line in open(path, errors="replace"):
            try:
                data = json.loads(line.replace("\0", "")).get("data")
            except (ValueError, AttributeError):
                continue
            if isinstance(data, str):
                lines.append(data.strip())
        for pattern in ("Segfault encountered", "AssertionError", "ImportError", "RuntimeError", "Error:", "error:"):
            hits = [l for l in lines if pattern in l]
            if hits:
                i = lines.index(hits[-1])
                detail = " ".join(lines[i:i + 2]) if pattern in ("ImportError",) else hits[-1]
                if pattern == "Segfault encountered":
                    frames = [l for l in lines[i:i + 8] if l.startswith("File") and "unknown" in l]
                    detail = "server segfault" + (" in " + frames[1].split(" in ")[-1] if len(frames) > 1 else "")
                return re.sub(r"^\((EngineCore|APIServer) pid=\d+\) ERROR [\d-]+ [\d:]+ \[[^]]+\] ", "", detail)
    return None


def describe_html(m: dict) -> str:
    what = m.get("what") or {}
    model = ", ".join(m.get("models") or [])
    parts = [p for p in (model, what.get("engine"), what.get("precision"), what.get("workload")) if p]
    return f"<span class=desc>{esc(' · '.join(parts))}</span>" if parts else ""


def same_weights_accuracy(row: dict, models: list[dict]) -> str:
    """Accuracy measured on the same weights by another benchmark (accuracy is the weights', not the engine's)."""
    mine = set(row.get("models") or [])
    for m in models:
        if mine & set(m.get("models") or []) and any(acc(m, t) is not None for t, _ in TASKS):
            scores = " · ".join(f"{label} {acc(m, t) * 100:.0f}" for t, label in TASKS if acc(m, t) is not None)
            return f"{esc(scores)}<span class=desc>same weights: {esc(m['bench'].replace('homelab-', ''))}</span>"
    return '<span class="na">pending: same weights not measured yet</span>'


# scores that measure the grader, not the model: (bench prefix, task) -> why. Left out of every table and score
INVALID = {
    ("homelab-vllm-muse-glimmer", "gsm8k"): "Muse answers in one short line (\"$18 per day\"), no \"####\" and no final "
                                            "number QualityBench's GSM8K grader takes: correct replies scored wrong (~35%)",
}


def invalid(m: dict, task: str) -> str | None:
    return next((why for (prefix, t), why in INVALID.items() if t == task and m.get("bench", "").startswith(prefix)), None)


def acc(m: dict, task: str, thinking: str | None = None):
    """The task's accuracy, thinking off; thinking on only for models that can't turn it off (gpt-oss)."""
    if invalid(m, task):
        return None
    found = {}
    for key, value in m.items():
        if key.startswith("acc.") and f".{task}." in key + "." and not key.endswith(".ci95"):
            found["on" if "think-on" in key else "off"] = value
    if thinking:
        return found.get(thinking)
    return found.get("off", found.get("on"))


def thinking_only(m: dict) -> bool:
    keys = [k for k in m if k.startswith("acc.") and not k.endswith(".ci95")]
    return bool(keys) and all("think-on" in k for k in keys)


SCORE_TASKS = ["gsm8k", "ifeval", "mmlu_pro", "humaneval", "mbpp", "lcb"]


def quality_weights(models: list[dict]) -> dict[str, float]:
    """Each task's weight in the quality score: its spread across the models relative to its mean (coefficient
    of variation), normalized. A task every model aces (GSM8K) tells them apart less than one where they spread
    out (LiveCodeBench): it counts less."""
    import statistics

    cv = {}
    for t in SCORE_TASKS:
        values = [acc(m, t) for m in models if acc(m, t) is not None]
        if len(values) > 2 and statistics.mean(values) > 0:
            cv[t] = statistics.stdev(values) / statistics.mean(values)
    total = sum(cv.values())
    return {t: v / total for t, v in cv.items()} if total else {}


QUANT_TAGS = re.compile(r"-(?:it|instruct|fp8|block|dynamic|nvfp4|w4a16|w8a8|quantized|awq|4bit|qat|ct|gguf|bf16)(?=-|$)")


def base_weights(m: dict) -> str:
    """The model the build's weights come from, quantization and packaging aside: nvidia/Qwen3.8-27B-NVFP4,
    Qwen/Qwen3.8-27B-FP8 and Qwen/Qwen3.8-27B are all qwen3.8-27b."""
    main = (m.get("models") or [m["bench"]])[0].split("/")[-1].lower()
    main = re.sub(r"^(?:meta-|nvidia-)", "", main)
    while (shorter := QUANT_TAGS.sub("", main)) != main:
        main = shorter
    return main


def share_full_lcb(models: list[dict]) -> None:
    """A build without its own full LiveCodeBench run takes the full score of another build of the same weights
    (quantization moves accuracy a point or two; the quick tier's 18 problems, ±20). Marked lcb_full_from."""
    full = {}
    for m in models:
        if acc(m, "lcb_full") is not None or acc(m, "lcb_full", "on") is not None:
            full.setdefault(base_weights(m), m)
    for m in models:
        src = full.get(base_weights(m))
        if src is None or src is m:
            continue
        for mode in ("off", "on"):  # per thinking mode: a model may have run one in full and not the other
            if acc(m, "lcb_full", mode) is None and acc(src, "lcb_full", mode) is not None:
                for k, v in src.items():
                    if k.startswith("acc.qualitybench.lcb_full.") and f".think-{mode}" in k:
                        m[k] = v
                m["lcb_full_from"] = src["bench"]


def quality_score(m: dict, weights: dict[str, float], min_share=0.75):
    """Weighted mean accuracy (%) over the tasks the model has; None when they carry less than min_share of
    the weight (a partial run)."""
    have = {t: acc(m, t) for t in weights if acc(m, t) is not None}
    if "lcb" in weights and acc(m, "lcb_full") is not None:  # the full 175 problems over the quick tier's 18
        have["lcb"] = acc(m, "lcb_full")
    share = sum(weights[t] for t in have)
    if not have or share < min_share:
        return None
    return 100 * sum(weights[t] * v for t, v in have.items()) / share


def mode_score(m: dict, weights: dict[str, float], mode: str, min_share=0.75, full: bool = False):
    """The quality score of one thinking mode's run (thinking.yaml models have both), quick-tier tasks only; with
    full, LiveCodeBench's 175 problems in that mode where run (lcb-full.yaml *-think-lcbfull)."""
    have = {t: acc(m, t, mode) for t in weights if acc(m, t, mode) is not None}
    if full and "lcb" in weights and acc(m, "lcb_full", mode) is not None:
        have["lcb"] = acc(m, "lcb_full", mode)
    share = sum(weights[t] for t in have)
    return 100 * sum(weights[t] * v for t, v in have.items()) / share if have and share >= min_share else None


def pareto(points: list[tuple[str, float, float]]) -> set[str]:
    """Names no other point beats on both axes (faster and better)."""
    return {n for n, x, y in points if not any(x2 >= x and y2 >= y and (x2, y2) != (x, y) for _, x2, y2 in points)}


VEGA = ('<script src="https://cdn.jsdelivr.net/npm/vega@5.30.0/build/vega.min.js"></script>'
        '<script src="https://cdn.jsdelivr.net/npm/vega-lite@5.21.0/build/vega-lite.min.js"></script>'
        '<script src="https://cdn.jsdelivr.net/npm/vega-embed@6.26.0/build/vega-embed.min.js"></script>')

# pass colors: the validated categorical order (dataviz reference palette), light / dark; dots use the first
# three only (all-pairs safe) and a shape each
PASS_SERIES = [("single", "1 user"), ("fill4k", "1 user, 4k prompt"), ("conc8", "8 users"), ("conc32", "32 users")]
DOT_SERIES = [("single", "1 user"), ("conc8", "8 users"), ("conc32", "32 users")]


def _vllm_tokens(server_data: str, t0: float, t1: float):
    """Tokens vLLM generated between t0 and t1 (epoch s): its periodic "Avg generation throughput" log lines,
    each the rate since the one before, integrated. None without such lines (llama.cpp)."""
    import datetime

    year = datetime.datetime.fromtimestamp(t0).year
    total, prev, seen = 0.0, None, False
    try:
        f = open(server_data, errors="replace")
    except OSError:
        return None
    with f:
        for line in f:
            if "generation throughput" not in line:
                continue
            m = re.search(r"INFO (\d\d)-(\d\d) (\d\d):(\d\d):(\d\d) .*?generation throughput: ([\d.]+) tokens/s", line)
            if not m:
                continue
            mo, d, hh, mm, ss, rate = m.groups()
            t = datetime.datetime(year, int(mo), int(d), int(hh), int(mm), int(ss)).timestamp()
            if t0 < t <= t1 + 10 and prev is not None and t > prev:
                total += float(rate) * min(t - prev, 30)
                seen = True
            prev = t
    return total if seen else None


def lcb_solutions(rows: list[dict], run_dir: str) -> list[dict]:
    """LiveCodeBench runs, what each solved problem cost: wall time, GPU energy and generated tokens over the
    problems solved. The full run (175 problems) is its pass; the quick tier's 18 are one task of the accuracy
    pass (its run.log gives the task's minutes, the pass's energy is shared out by time)."""
    import datetime
    import glob

    out = []
    for r in rows:
        bench = r["bench"]
        keys = [k for k in r if k.startswith("acc.") and not k.endswith(".ci95") and re.search(r"\.lcb(_full)?\.", k + ".")]
        # > 0: a rerun appends to the data file, so while it runs the row mixes the old attempt's end with its start
        if not keys or (r.get("duration_s") or 0) <= 0:
            continue
        key = keys[0]
        full = ".lcb_full." in key + "."
        mode = "on" if "think-on" in key else "off"
        n = 175 if full else 18
        solved = r[key] * n
        if solved < 0.5:
            continue
        qb = glob.glob(os.path.join(run_dir, f"{bench}.qualitybench", "*.qb"))
        t1 = r.get("end") or 0
        seconds = r["duration_s"]
        tokens = None
        state = {}
        if qb:
            try:
                state = json.load(open(os.path.join(qb[0], "state.json"))).get("caps", {}).get("lcb", {})
            except (OSError, ValueError):
                state = {}
        if not full:
            done = None
            try:
                for line in open(os.path.join(qb[0], "run.log"), errors="replace") if qb else []:
                    if m := re.match(r"\[([\d-]+ [\d:]+)\] lcb: done in ([\d.]+) min", line):
                        done = m
            except OSError:
                pass
            if done is None:
                continue
            t1 = datetime.datetime.strptime(done.group(1), "%Y-%m-%d %H:%M:%S").timestamp()
            seconds = float(done.group(2)) * 60
        energy = r.get("energy_j") and r["energy_j"] * min(seconds / r["duration_s"], 1)
        tokens = state.get("tokens") or _vllm_tokens(os.path.join(run_dir, f"{split(bench)[0]}.0.server.nolog.data"),
                                                      t1 - seconds, t1)
        out.append({"head": model_key(bench)[0], "mode": mode, "tier": "full (175)" if full else "quick (18)",
                    "solved": round(solved), "n": n, "LCB %": round(100 * r[key], 1), "s/solve": round(seconds / solved, 1),
                    "kJ/solve": None if not energy else round(energy / solved / 1000, 2),
                    "tok/solve": None if not tokens else round(tokens / solved)})
    return out


def gguf_sizes(run_dir: str, update: dict | None = None) -> dict:
    """GGUF sizes in bytes ("repo/file" -> all shards), kept in <run>/report/gguf_sizes.json: weights deleted from the
    cache after their run keep their size in the memory charts."""
    path = os.path.join(run_dir, "report", "gguf_sizes.json")
    if update is not None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path + ".tmp", "w") as f:
            json.dump(update, f, indent=1, sort_keys=True)
        os.replace(path + ".tmp", path)
        return update
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        return {}


def weights_gib(m: dict, run_dir: str):
    """Memory the model's weights take: vLLM's own count (draft model included), or for llama.cpp the GGUF file's
    size (all its shards; the repo folder may hold other quantizations)."""
    if m.get("vllm.weights_gib"):
        return m["vllm.weights_gib"]
    import glob

    hub = os.environ.get("HF_HUB_CACHE", "/opt/milabench/data/hub")
    for bench in sorted(m.get("benches") or []):
        try:
            with open(os.path.join(run_dir, f"{bench}.0.data"), errors="replace") as f:
                argv = (json.loads(f.readline().replace("\0", "")).get("data") or {}).get("server", {}).get("argv") or {}
        except (OSError, ValueError):
            continue
        repo, name = argv.get("--repo"), argv.get("--file")
        if not (repo and name):
            continue
        pattern = re.sub(r"-00001-of-(\d+)\.gguf$", r"-*-of-\1.gguf", name)
        files = glob.glob(os.path.join(hub, "models--" + repo.replace("/", "--"), "snapshots", "*", pattern))
        sizes = {os.path.basename(f): os.path.getsize(f) for f in files if os.path.exists(f)}
        known = gguf_sizes(run_dir)
        if sum(sizes.values()) > 2**29:
            if known.get(f"{repo}/{name}") != sum(sizes.values()):
                known[f"{repo}/{name}"] = sum(sizes.values())
                gguf_sizes(run_dir, known)
            return sum(sizes.values()) / 2**30
        if known.get(f"{repo}/{name}"):  # the GGUF since deleted: its size when it was there
            return known[f"{repo}/{name}"] / 2**30
    # the GGUF since deleted: the model folder's size recorded at run time
    return m.get("disk_gb") if (m.get("disk_gb") or 0) > 0.5 else None


LCB_FULL, LCB_QUICK = "thinking off · LiveCode full (175 problems)", "thinking off · LiveCode quick (18)"
THINK_ON, THINK_QUICK = "thinking on", "thinking on · LiveCode quick (18)"

# the summary leaderboard: one column per criterion, each scored 0-100 across the models (accuracies as they are;
# speeds, efficiency, memory and cost on a log scale, since they span orders of magnitude; lower-is-better ones
# flipped), and an overall score from weights the reader sets (kept in their browser and in the page's address)
BOARD = [  # key, label, unit, higher is better, log scale, default weight, what it is
    ("quality", "Quality", "%", True, False, 3, "the quality score: weighted accuracy across the QualityBench tasks (red −N: N tasks not run or cut by the deadline; grey −N: left out because the grader can't read the model's answers; either way the mean of the others)"),
    ("lcb", "Coding", "%", True, False, 3, "LiveCodeBench pass rate: the full 175 problems where run (175*: run on another build of the same weights), else the quick 18; N⌛: replies of the full run past the 1 h time-to-solution window, scored as wrong"),
    ("speed", "1-user speed", "tok/s", True, True, 2, "output tokens per second one user sees, alone on the server"),
    ("tput", "32-user throughput", "tok/s", True, True, 1, "output tokens per second of all 32 users sharing the server"),
    ("eff", "Energy efficiency", "tok/J", True, True, 1, "output tokens per joule of GPU energy, at the model's most efficient load"),
    ("solve", "Cost per solve", "kJ", False, True, 1, "GPU energy per LiveCodeBench problem solved (failures included)"),
    ("mem", "Memory", "GiB", False, True, 1, "memory the weights take, of the machine's 119.6 GiB"),
]
PRESETS = {
    "Balanced": {k: w for k, *_, w, _ in BOARD},
    "Coding agent, 1 user": {"quality": 2, "lcb": 4, "speed": 3, "tput": 0, "eff": 0, "solve": 1, "mem": 0},
    "Serving many users": {"quality": 2, "lcb": 1, "speed": 0, "tput": 4, "eff": 2, "solve": 0, "mem": 0},
    "Most efficient": {"quality": 2, "lcb": 1, "speed": 0, "tput": 0, "eff": 3, "solve": 3, "mem": 1},
    "Quality only": {"quality": 3, "lcb": 2, "speed": 0, "tput": 0, "eff": 0, "solve": 0, "mem": 0},
}


def leaderboard(models: list[dict], weights: dict[str, float], run_dir: str, rows: list[dict]) -> str:
    solves = defaultdict(list)
    for e in lcb_solutions(rows, run_dir):
        if e.get("kJ/solve"):
            solves[e["head"]].append(e)
    data = []
    for m in models:
        score = quality_score(m, weights)
        if score is None:
            continue
        think = thinking_only(m)
        single = m["passes"].get("single") or m["passes"].get("conc1")
        conc32 = m["passes"].get("conc32")
        effs = [r.get("output_tokens_per_joule") for r in m["passes"].values() if r.get("output_tokens_per_joule")]
        lcb_full, lcb = acc(m, "lcb_full"), acc(m, "lcb")
        # thinking off unless the model always thinks; the full run over the quick one
        mine = [x for x in solves.get(m["bench"], []) if x["mode"] == ("on" if think else "off")]
        e = max(mine, key=lambda x: x["tier"].startswith("full"), default=None)
        gib = weights_gib(m, run_dir)
        name = ("* " if think else "") + m["bench"].replace("homelab-", "").replace("vllm-", "").replace("llamacpp-", "llama.cpp ")
        label = dict(TASKS)
        gone = [t for t in weights if acc(m, t) is None and not (t == "lcb" and acc(m, "lcb_full") is not None)]
        missing = [label.get(t, t) for t in gone if not invalid(m, t)]  # not run, or cut by the run deadline
        excluded = [f"{label.get(t, t)}: {invalid(m, t)}" for t in gone if invalid(m, t)]  # run, but the grader fails
        data.append({"model": name, "quality": round(score, 1), "missing": missing, "excluded": excluded,
                     "lcb": None if (lcb_full if lcb_full is not None else lcb) is None else round(100 * (lcb_full if lcb_full is not None else lcb), 1),
                     "lcb_full": lcb_full is not None, "lcb_shared": bool(m.get("lcb_full_from")),
                     "lcb_timeouts": lcb_full is not None and lcb_full_timeouts(run_dir, m.get("lcb_full_from") or m["bench"], think) or 0,
                     "lcb_queued": lcb_full is not None and queue_wait(run_dir, (m.get("lcb_full_from") or m["bench"]) + "-lcbfull") or None,
                     "speed": per_user(single) and round(per_user(single), 1),
                     "tput": conc32 and conc32.get("output_throughput") and round(conc32["output_throughput"], 1),
                     "eff": max(effs) if effs else None,
                     "solve": e and e.get("kJ/solve"),
                     "mem": gib and round(gib, 1)})
    if not data:
        return ""
    cols = [{"key": k, "label": l, "unit": u, "up": up, "log": lg, "w": w, "what": what} for k, l, u, up, lg, w, what in BOARD]
    head = "".join(
        f"<th class=crit title='{esc(c['what'])}'><div>{esc(c['label'])}</div><div class=unit>{esc(c['unit'])}"
        f"{'' if c['up'] else ', lower better'}</div>"
        f"<input type=range min=0 max=4 step=0.5 value={c['w']} data-key={c['key']} aria-label='weight of {esc(c['label'])}'>"
        f"<output>{c['w']:g}</output></th>" for c in cols)
    presets = "".join(f"<button type=button data-preset='{esc(n)}'>{esc(n)}</button>" for n in PRESETS)
    return ("<h2>Leaderboard</h2><div class=note>Every model with a quality score, one column per criterion. Each column "
            "is scored 0–100 across the models (speeds, efficiency, cost and memory on a log scale; lower-is-better columns "
            "flipped), the bar shows that score, the number is the measurement. Set a weight per column (0 leaves it out) "
            "or pick a preset: the overall score is the weighted mean of the column scores the model has, and the table "
            "re-ranks. A missing measurement does not count against a model, it just has fewer columns (marked –). Your "
            "weights stay in this browser and in the page address, to share a ranking.</div>"
            f"<div class=presets>{presets}</div>"
            "<div class=wrap><table class=board id=board><thead><tr><th>#</th><th class=left>model</th><th class=crit>Overall</th>"
            + head + "</tr></thead><tbody></tbody></table></div>"
            "<script>(function(){const COLS=" + json.dumps(cols) + ",DATA=" + json.dumps(data).replace("</", "<\\/")
            + ",PRESETS=" + json.dumps(PRESETS) + ";" + BOARD_JS + "})();</script>")


BOARD_JS = r"""
const tb=document.querySelector('#board tbody'),inputs=[...document.querySelectorAll('#board input[type=range]')];
const range={};for(const c of COLS){const v=DATA.map(d=>d[c.key]).filter(x=>x!=null&&x>0).map(x=>c.log?Math.log(x):x);
 range[c.key]=[Math.min(...v),Math.max(...v)];}
function norm(c,x){if(x==null||!(x>0))return null;const [lo,hi]=range[c.key];if(hi<=lo)return 100;
 let t=((c.log?Math.log(x):x)-lo)/(hi-lo);if(!c.up)t=1-t;return Math.round(100*t);}
function fmtv(c,x){if(x==null)return '–';return c.key==='eff'?x.toFixed(2):c.key==='solve'?x.toFixed(2):x>=100?Math.round(x).toLocaleString():x.toFixed(1);}
function weights(){const w={};for(const i of inputs)w[i.dataset.key]=+i.value;return w;}
function save(w){const q=new URLSearchParams(location.hash.slice(1));q.set('w',COLS.map(c=>w[c.key]).join(','));
 try{history.replaceState(null,'','#'+q.toString());}catch(e){} try{localStorage.setItem('board-w',JSON.stringify(w));}catch(e){}}
function load(){let w=null;const q=new URLSearchParams(location.hash.slice(1)).get('w');
 if(q){const v=q.split(',').map(Number);if(v.length===COLS.length&&v.every(x=>x>=0&&x<=4))w=Object.fromEntries(COLS.map((c,i)=>[c.key,v[i]]));}
 if(!w){try{w=JSON.parse(localStorage.getItem('board-w')||'null');}catch(e){}}
 if(w)for(const i of inputs)if(w[i.dataset.key]!=null){i.value=w[i.dataset.key];i.nextElementSibling.textContent=(+i.value).toString();}}
function esc(t){return String(t).replace(/&/g,'&amp;').replace(/'/g,'&#39;').replace(/"/g,'&quot;').replace(/</g,'&lt;');}
function cell(s,txt,extra){return `<td class="bc${s==null?' na':''}"><div class=bar-bg>${s==null?'':`<i style="width:${s}%"></i>`}</div><span>${txt}</span>${extra||''}</td>`;}
function render(){const w=weights();const rows=DATA.map(d=>{let num=0,den=0;const s={};
  for(const c of COLS){s[c.key]=norm(c,d[c.key]);if(s[c.key]!=null&&w[c.key]>0){num+=w[c.key]*s[c.key];den+=w[c.key];}}
  return {d,s,overall:den?num/den:null};}).sort((a,b)=>(b.overall??-1)-(a.overall??-1));
 tb.innerHTML=rows.map((r,i)=>`<tr><td>${r.overall==null?'':i+1}</td><td class=left>${r.d.model.replace(/&/g,'&amp;').replace(/</g,'&lt;')}</td>`
  +cell(r.overall==null?null:Math.round(r.overall),r.overall==null?'–':`<b>${r.overall.toFixed(0)}</b>`,'').replace('class="bc','class="bc overall')
  +COLS.map(c=>cell(r.s[c.key],fmtv(c,r.d[c.key]),c.key==='quality'?(r.d.missing&&r.d.missing.length?`<sup class=bad title='scored without ${esc(r.d.missing.join(', '))} (not run or cut by the run deadline): the mean of the other tasks, so not comparable with a full score'>−${r.d.missing.length}</sup>`:'')+(r.d.excluded&&r.d.excluded.length?`<sup class=na title='left out: ${esc(r.d.excluded.join('; '))}. The mean of the other tasks'>−${r.d.excluded.length}</sup>`:''):c.key==='lcb'&&r.d.lcb!=null?`<sup${r.d.lcb_shared?" title='the full run of another build of the same weights'":''}>${r.d.lcb_full?(r.d.lcb_shared?'175*':'175'):'18'}</sup>`+(r.d.lcb_timeouts?`<sup class=bad title='${r.d.lcb_timeouts} of 175 replies not back within the 1 h time-to-solution window (16 requests in flight), scored as wrong: the score is time-bounded${r.d.lcb_queued?`. Inflated: the server decodes ${r.d.lcb_queued} at a time, the other requests waited in its queue with their hour running (fixed for later runs)`:''}'>${r.d.lcb_timeouts}⌛</sup>`:''):'')).join('')+'</tr>').join('');
 save(w);}
for(const i of inputs)i.addEventListener('input',()=>{i.nextElementSibling.textContent=(+i.value).toString();render();});
for(const b of document.querySelectorAll('.presets button'))b.addEventListener('click',()=>{const p=PRESETS[b.dataset.preset];
 for(const i of inputs){i.value=p[i.dataset.key];i.nextElementSibling.textContent=(+i.value).toString();}render();});
load();render();
"""


def charts(models: list[dict], weights: dict[str, float], run_dir: str = "", rows: list[dict] | None = None) -> str:
    """Quality score, quality vs speed, total and per-user throughput per pass: Vega-Lite specs, data inline."""
    # thinking-on models lead with a * (in front: long axis labels are cut at their end)
    short = lambda m: (("* " if thinking_only(m) else "")
                       + m["bench"].replace("homelab-", "").replace("vllm-", "").replace("llamacpp-", "llama.cpp "))
    family = lambda name: (re.match(r"[a-z]+", name.removeprefix("* ").removeprefix("llama.cpp ")) or [name])[0]
    label = dict(TASKS)
    quality, scatter, total, user, energy, curve, memory = [], [], [], [], [], [], []
    for m in models:
        name = short(m)
        gib = weights_gib(m, run_dir)
        score = quality_score(m, weights)
        single = m["passes"].get("single") or m["passes"].get("conc1")
        if score is not None:
            think = thinking_only(m)
            full = acc(m, "lcb_full") is not None
            on_full = not think and acc(m, "lcb_full", "on") is not None and acc(m, "lcb_full", "off") is not None
            on = None if think else mode_score(m, weights, "on", full=on_full)
            row = {"model": name, "family": family(name), "score": round(score, 1), "thinking": "on" if think else "off",
                   "label": f"{score:.0f}", "tier": (THINK_ON if full else THINK_QUICK) if think else (LCB_FULL if full else LCB_QUICK),
                   "score_on": None if on is None else round(on, 1),
                   # the thinking-on score: LiveCodeBench's 175 where run thinking on (and off), else the quick 18
                   "tier_on": None if on is None else THINK_ON if on_full else THINK_QUICK,
                   "label_on": None if on is None or on <= score else [f"{on:.0f}", "↑", f"{score:.0f}"],  # stacked: narrow
                   "lcb_from": m.get("lcb_full_from", "").replace("homelab-", "").replace("vllm-", "").replace("llamacpp-", "llama.cpp ") or None,
                   "gain": None if on is None else f"{on - mode_score(m, weights, 'off', full=on_full):+.1f}"}
            (t_off, e_off), (t_on, e_on) = m.get("cost.off", (None, None)), m.get("cost.on", (None, None))
            if on is not None and t_off and t_on:
                row["think_cost"] = f"{t_on / t_off:.1f}× time" + (f", {e_on / e_off:.1f}× energy" if e_off and e_on else "")
            row.update({label[t]: round(acc(m, t) * 100, 1) for t in weights if acc(m, t) is not None})
            quality.append(row)
            if gib:
                row["GiB"] = round(gib, 1)
            if per_user(single):
                scatter.append({**row, "speed": round(per_user(single), 1)})
        for key, name_p in PASS_SERIES:
            r = m["passes"].get(key) or (single if key == "single" else None)
            if r and r.get("output_throughput"):
                total.append({"model": name, "family": family(name), "pass": name_p, "tok/s": round(r["output_throughput"], 1)})
        for key, name_p in DOT_SERIES:
            r = m["passes"].get(key) or (single if key == "single" else None)
            if r and r.get("output_throughput") and r.get("tpot_ms_p50"):
                curve.append({"model": name, "family": family(name), "pass": name_p, "order": [k for k, _ in DOT_SERIES].index(key),
                              "tok/s": round(r["output_throughput"], 1), "TPOT ms": round(r["tpot_ms_p50"], 1),
                              "TTFT ms": round(r.get("ttft_ms_p50") or 0), "tok/s/user": round(1000 / r["tpot_ms_p50"], 1)})
            if r and r.get("output_tokens_per_joule"):
                energy.append({"model": name, "family": family(name), "pass": name_p,
                               "tok/J": round(r["output_tokens_per_joule"], 2), "W": r.get("power_mean_w")})
            if gib and r and r.get("output_throughput"):
                memory.append({"model": name, "family": family(name), "pass": name_p, "GiB": round(gib, 1),
                               "order": [k for k, _ in DOT_SERIES].index(key), "tok/s": round(r["output_throughput"], 1)})
            if per_user(r):
                user.append({"model": name, "family": family(name), "pass": name_p, "tok/s/user": round(per_user(r), 1)})
    if not (quality or total):
        return ""
    front = pareto([(q["model"], q["speed"], q["score"]) for q in scatter])
    for q in scatter:
        q["frontier"] = q["model"] in front
    by_score = [q["model"] for q in sorted(quality, key=lambda q: -q["score"])]
    by_peak = [n for n, _ in sorted({t["model"]: t["tok/s"] for t in total if t["pass"] == "32 users"}.items(), key=lambda kv: -kv[1])]
    by_peak += sorted({t["model"] for t in total} - set(by_peak))
    best_eff = {}
    for e in energy:
        best_eff[e["model"]] = max(best_eff.get(e["model"], 0), e["tok/J"])
    by_eff = [n for n, _ in sorted(best_eff.items(), key=lambda kv: -kv[1])]
    q_energy = [{**q, "tok/J": best_eff[q["model"]]} for q in quality if q["model"] in best_eff]
    front_e = pareto([(q["model"], q["tok/J"], q["score"]) for q in q_energy])
    for q in q_energy:
        q["frontier"] = q["model"] in front_e
    tasks_tip = [{"field": label[t], "type": "quantitative", "title": f"{label[t]} (w {weights[t]:.0%})"} for t in weights]
    series_scale = {"domain": [LCB_FULL, LCB_QUICK, THINK_ON, THINK_QUICK],
                    "range": ["SERIES0", "SERIES0PALE", "SERIES1", "SERIES1PALE"]}
    score_tip = ([{"field": "model"}, {"field": "family"}, {"field": "score", "title": "quality score"},
                  {"field": "tier", "title": "LiveCodeBench"},
                  {"field": "lcb_from", "title": "LiveCode full measured on (same weights, other build)"},
                  {"field": "score_on", "title": "score, thinking on"},
                  {"field": "gain", "title": "thinking on gain (same LiveCode tier both ways)"},
                  {"field": "think_cost", "title": "thinking on costs (accuracy pass)"}] + tasks_tip)
    x_model = lambda order: {"field": "model", "type": "nominal", "sort": order, "title": None,
                             "axis": {"labelAngle": -50, "labelLimit": 240}}
    specs = {
        "q-score": {
            "data": {"values": quality}, "height": 300,
            "layer": [
                # thinking on, for the models scored both ways: behind the thinking-off bar, it shows only where it scores higher
                {"transform": [{"filter": "datum.score_on != null"}, {"calculate": "datum.tier_on", "as": "tier"}],
                 "mark": {"type": "bar", "cornerRadiusTopLeft": 4, "cornerRadiusTopRight": 4, "width": {"band": 0.7}},
                 "encoding": {"x": x_model(by_score), "y": {"field": "score_on", "type": "quantitative"},
                              "opacity": {"condition": {"test": "hl == null || datum.family == hl", "value": 1}, "value": 0.15},
                              "color": {"field": "tier", "type": "nominal", "scale": series_scale}, "tooltip": score_tip}},
                {"mark": {"type": "bar", "cornerRadiusTopLeft": 4, "cornerRadiusTopRight": 4, "width": {"band": 0.7}},
                 "encoding": {"x": x_model(by_score),
                              "y": {"field": "score", "type": "quantitative", "title": "quality score (%)", "scale": {"domain": [0, 100]}},
                              "opacity": {"condition": {"test": "hl == null || datum.family == hl", "value": 1}, "value": 0.15},
                              "color": {"field": "tier", "type": "nominal", "scale": series_scale,
                                        "legend": {"title": None, "orient": "top", "labelLimit": 400}},
                              # the quick-tier-only bars: the same blue, pale (less certain), outlined
                              "stroke": {"field": "tier", "type": "nominal", "scale": {"domain": [LCB_FULL, LCB_QUICK, THINK_ON, THINK_QUICK],
                                         "range": ["SERIES0", "SERIES0", "SERIES1", "SERIES1"]},
                                         "legend": {"title": None, "orient": "top", "labelLimit": 400}},
                              "strokeWidth": {"value": 1},
                              "tooltip": score_tip}},
                {"transform": [  # a stacked label's 2 extra lines: 20 px, 7 points on the fixed 0-100 axis of 300 px
                               {"calculate": "max(datum.score, datum.score_on || 0) + (datum.label_on ? 7 : 0)", "as": "top"},
                               {"calculate": "datum.label_on || datum.label", "as": "shown"}],
                 "mark": {"type": "text", "dy": -4, "fontSize": 10, "angle": 0, "baseline": "bottom", "lineHeight": 10},
                 "encoding": {"x": x_model(by_score), "y": {"field": "top", "type": "quantitative"},
                              "text": {"field": "shown"}, "color": {"value": "MUTED"}, "opacity": {"condition": {"test": "hl == null || datum.family == hl", "value": 1}, "value": 0.15}}}]},
        "q-speed": {
            "data": {"values": scatter}, "height": 340,
            "layer": [
                {"mark": {"type": "point", "filled": True, "size": 80, "stroke": "SURFACE", "strokeWidth": 2},
                 "encoding": {"x": {"field": "speed", "type": "quantitative", "title": "1-user output tok/s"},
                              "y": {"field": "score", "type": "quantitative", "title": "quality score (%)", "scale": {"zero": False}},
                              "opacity": {"condition": {"test": "hl == null || datum.family == hl", "value": 1}, "value": 0.15}, "color": {"field": "frontier", "type": "nominal", "scale": {"domain": [True, False], "range": ["SERIES0", "MUTED"]},
                                        "legend": {"title": None, "labelExpr": "datum.value ? 'best trade-off (nothing faster and better)' : 'other'", "orient": "bottom", "labelLimit": 400}},
                              "tooltip": [{"field": "model"}, {"field": "speed", "title": "1-user tok/s"}, {"field": "score", "title": "quality score"}] + tasks_tip}},
                # a frontier point whose coding score is still LiveCodeBench's quick 18 says so: the full 175 is due
                {"transform": [{"filter": "datum.frontier || datum.family == hl"},
                               {"calculate": "datum.model + (datum.frontier && datum.tier && indexof(datum.tier, 'quick') >= 0 ? ' · LCB 18 only' : '')",
                                "as": "label"}],
                 "mark": {"type": "text", "align": "left", "dx": 7, "dy": -4, "fontSize": 10.5},
                 "encoding": {"x": {"field": "speed", "type": "quantitative"}, "y": {"field": "score", "type": "quantitative"},
                              "text": {"field": "label"}, "color": {"value": "TEXT"}}}]},
        "tput-total": {
            "data": {"values": total}, "height": 320,
            "mark": {"type": "bar", "cornerRadiusTopLeft": 3, "cornerRadiusTopRight": 3},
            "encoding": {"x": x_model(by_peak), "xOffset": {"field": "pass", "sort": [p for _, p in PASS_SERIES]},
                         "y": {"field": "tok/s", "type": "quantitative", "title": "output tok/s, all users"},
                         "opacity": {"condition": {"test": "hl == null || datum.family == hl", "value": 1}, "value": 0.15}, "color": {"field": "pass", "type": "nominal", "sort": [p for _, p in PASS_SERIES],
                                   "scale": {"domain": [p for _, p in PASS_SERIES], "range": ["SERIES0", "SERIES1", "SERIES2", "SERIES3"]},
                                   "legend": {"title": None, "orient": "top"}},
                         "tooltip": [{"field": "model"}, {"field": "pass"}, {"field": "tok/s"}]}},
        "tput-user": {
            "data": {"values": user}, "height": 320,
            "layer": [
                {"mark": {"type": "rule", "strokeWidth": 1.5, "opacity": 0.5},
                 "encoding": {"x": x_model(by_peak), "y": {"aggregate": "min", "field": "tok/s/user", "type": "quantitative"},
                              "y2": {"aggregate": "max", "field": "tok/s/user"}, "detail": {"field": "family"}, "color": {"value": "MUTED"}, "opacity": {"condition": {"test": "hl == null || datum.family == hl", "value": 1}, "value": 0.15}}},
                {"mark": {"type": "point", "filled": True, "size": 70, "stroke": "SURFACE", "strokeWidth": 1.5},
                 "encoding": {"x": x_model(by_peak),
                              "y": {"field": "tok/s/user", "type": "quantitative", "title": "output tok/s one user sees"},
                              "opacity": {"condition": {"test": "hl == null || datum.family == hl", "value": 1}, "value": 0.15}, "color": {"field": "pass", "type": "nominal", "scale": {"domain": [p for _, p in DOT_SERIES], "range": ["SERIES0", "SERIES1", "SERIES2"]},
                                        "legend": {"title": None, "orient": "top"}},
                              "shape": {"field": "pass", "type": "nominal", "scale": {"domain": [p for _, p in DOT_SERIES], "range": ["circle", "square", "triangle-up"]}},
                              "tooltip": [{"field": "model"}, {"field": "pass"}, {"field": "tok/s/user"}]}}]},
    }
    specs["q-energy"] = json.loads(json.dumps(specs["q-speed"]))  # the same scatter, efficiency on x
    specs["q-energy"]["data"] = {"values": q_energy}
    for layer in specs["q-energy"]["layer"]:
        layer["encoding"]["x"] = {"field": "tok/J", "type": "quantitative", "title": "best output tokens per joule (GPU)"}
    specs["q-energy"]["layer"][0]["encoding"]["color"]["legend"]["labelExpr"] = (
        "datum.value ? 'best trade-off (nothing more efficient and better)' : 'other'")
    specs["q-energy"]["layer"][0]["encoding"]["tooltip"] = (
        [{"field": "model"}, {"field": "family"}, {"field": "tok/J", "title": "best tokens / joule"},
         {"field": "score", "title": "quality score"}] + tasks_tip)
    # quality vs memory: the same scatter, weights' memory on x (less is better: the frontier minimizes it)
    q_mem = [dict(q) for q in quality if q.get("GiB")]
    front_m = pareto([(q["model"], -q["GiB"], q["score"]) for q in q_mem])
    for q in q_mem:
        q["frontier"] = q["model"] in front_m
    x_mem = {"field": "GiB", "type": "quantitative", "title": "memory the weights take (GiB, of 119.6 GiB unified)"}
    specs["q-memory"] = json.loads(json.dumps(specs["q-speed"]))
    specs["q-memory"]["data"] = {"values": q_mem}
    for layer in specs["q-memory"]["layer"]:
        layer["encoding"]["x"] = dict(x_mem)
    specs["q-memory"]["layer"][0]["encoding"]["color"]["legend"]["labelExpr"] = (
        "datum.value ? 'best trade-off (nothing smaller and better)' : 'other'")
    specs["q-memory"]["layer"][0]["encoding"]["tooltip"] = (
        [{"field": "model"}, {"field": "family"}, {"field": "GiB", "title": "weights (GiB)"},
         {"field": "score", "title": "quality score"}] + tasks_tip)
    hl_test = "hl == null || datum.family == hl"
    on_color = {"condition": {"test": "hl != null && datum.family == hl", "value": "SERIES0"}, "value": "MUTED"}
    x_tput = {"field": "tok/s", "type": "quantitative", "title": "output tok/s, all users"}
    y_lat = {"field": "TPOT ms", "type": "quantitative", "title": "time between tokens for one user, p50 (ms)",
             "scale": {"type": "log", "nice": False, "padding": 8}}
    front_c = pareto([(f"{c['model']}|{c['pass']}", c["tok/s"], -c["TPOT ms"]) for c in curve])
    for c in curve:
        c["frontier"] = f"{c['model']}|{c['pass']}" in front_c
    specs["tput-latency"] = {
        "data": {"values": curve}, "height": 380,
        "layer": [
            {"mark": {"type": "line", "strokeWidth": 1.5},
             "encoding": {"x": x_tput, "y": y_lat, "detail": {"field": "model"}, "order": {"field": "order"},
                          "color": on_color, "opacity": {"condition": {"test": hl_test, "value": 0.6}, "value": 0.12}}},
            {"mark": {"type": "point", "filled": True, "size": 55, "stroke": "SURFACE", "strokeWidth": 1.5},
             "encoding": {"x": x_tput, "y": y_lat, "color": on_color,
                          "shape": {"field": "pass", "type": "nominal", "scale": {"domain": [p for _, p in DOT_SERIES], "range": ["circle", "square", "triangle-up"]},
                                    "legend": {"title": None, "orient": "top"}},
                          "opacity": {"condition": {"test": hl_test, "value": 1}, "value": 0.15},
                          "tooltip": [{"field": "model"}, {"field": "pass"}, {"field": "tok/s", "title": "output tok/s, all users"},
                                      {"field": "tok/s/user", "title": "tok/s one user sees"}, {"field": "TPOT ms", "title": "TPOT p50 (ms)"},
                                      {"field": "TTFT ms", "title": "TTFT p50 (ms)"}]}},
            {"transform": [{"filter": "hl != null && datum.family == hl && datum.order == 2"}],
             "mark": {"type": "text", "align": "left", "dx": 7, "fontSize": 10.5},
             "encoding": {"x": x_tput, "y": y_lat, "text": {"field": "model"}, "color": {"value": "TEXT"}}}]}
    specs["tput-energy"] = {
        "data": {"values": energy}, "height": 320,
        "layer": [
            {"mark": {"type": "rule", "strokeWidth": 1.5, "opacity": 0.5},
             "encoding": {"x": x_model(by_eff), "y": {"aggregate": "min", "field": "tok/J", "type": "quantitative"},
                          "y2": {"aggregate": "max", "field": "tok/J"}, "detail": {"field": "family"}, "color": {"value": "MUTED"},
                          "opacity": {"condition": {"test": "hl == null || datum.family == hl", "value": 1}, "value": 0.15}}},
            {"mark": {"type": "point", "filled": True, "size": 70, "stroke": "SURFACE", "strokeWidth": 1.5},
             "encoding": {"x": x_model(by_eff),
                          "y": {"field": "tok/J", "type": "quantitative", "title": "output tokens per joule (GPU)"},
                          "opacity": {"condition": {"test": "hl == null || datum.family == hl", "value": 1}, "value": 0.15},
                          "color": {"field": "pass", "type": "nominal", "scale": {"domain": [p for _, p in DOT_SERIES], "range": ["SERIES0", "SERIES1", "SERIES2"]},
                                    "legend": {"title": None, "orient": "top"}},
                          "shape": {"field": "pass", "type": "nominal", "scale": {"domain": [p for _, p in DOT_SERIES], "range": ["circle", "square", "triangle-up"]}},
                          "tooltip": [{"field": "model"}, {"field": "pass"}, {"field": "tok/J", "title": "tokens / joule"},
                                      {"field": "W", "title": "GPU power (W)"}]}}]}
    dot_scale = {"domain": [p for _, p in DOT_SERIES], "range": ["SERIES0", "SERIES1", "SERIES2"]}
    specs["tput-memory"] = {
        "data": {"values": memory}, "height": 360,
        "layer": [
            {"mark": {"type": "rule", "strokeWidth": 1.5},
             "encoding": {"x": dict(x_mem), "y": {"aggregate": "min", "field": "tok/s", "type": "quantitative"},
                          "y2": {"aggregate": "max", "field": "tok/s"}, "detail": {"field": "model"}, "color": {"value": "MUTED"},
                          "opacity": {"condition": {"test": hl_test, "value": 0.45}, "value": 0.1}}},
            {"mark": {"type": "point", "filled": True, "size": 60, "stroke": "SURFACE", "strokeWidth": 1.5},
             "encoding": {"x": dict(x_mem),
                          "y": {"field": "tok/s", "type": "quantitative", "title": "output tok/s, all users (log scale)",
                                "scale": {"type": "log", "domain": [5, 3000]},
                                "axis": {"values": [5, 10, 20, 50, 100, 200, 500, 1000, 2000]}},
                          "color": {"field": "pass", "type": "nominal", "scale": dot_scale, "legend": {"title": None, "orient": "top"}},
                          "shape": {"field": "pass", "type": "nominal", "scale": {"domain": [p for _, p in DOT_SERIES], "range": ["circle", "square", "triangle-up"]}},
                          "opacity": {"condition": {"test": hl_test, "value": 1}, "value": 0.15},
                          "tooltip": [{"field": "model"}, {"field": "pass"}, {"field": "GiB", "title": "weights (GiB)"},
                                      {"field": "tok/s", "title": "output tok/s, all users"}]}},
            {"transform": [{"filter": "hl != null && datum.family == hl && datum.order == 2"}],
             "mark": {"type": "text", "align": "left", "dx": 7, "fontSize": 10.5},
             "encoding": {"x": dict(x_mem), "y": {"field": "tok/s", "type": "quantitative"}, "text": {"field": "model"},
                          "color": {"value": "TEXT"}}}]}
    # LiveCodeBench: what one solved problem costs, thinking off vs on (the full run where there is one)
    by_head = {m["bench"]: m for m in models}
    solve = {}
    for e in lcb_solutions(rows or [], run_dir):
        m = by_head.get(e["head"])
        if m is None:
            continue
        name = short(m)
        # one point per model and mode: the full run when there is one (the quick tier's 18 problems are an
        # estimate); off and on are joined only when both come from the same tier
        k = (name, e["mode"])
        if k not in solve or e["tier"].startswith("full"):
            solve[k] = {**e, "model": name, "family": family(name), "pair": f"{name}|{e['tier']}",
                        "thinking": "thinking on" if e["mode"] == "on" else "thinking off"}
    solve = list(solve.values())
    mode_scale = {"domain": ["thinking off", "thinking on"], "range": ["SERIES0", "SERIES1"]}
    solve_tip = [{"field": "model"}, {"field": "thinking"}, {"field": "tier", "title": "LiveCodeBench"},
                 {"field": "LCB %", "title": "pass rate (%)"},
                 {"field": "solved", "title": "problems solved"}, {"field": "s/solve", "title": "seconds per solve"},
                 {"field": "kJ/solve", "title": "kJ per solve"}, {"field": "tok/solve", "title": "tokens per solve"}]
    x_lcb = {"field": "LCB %", "type": "quantitative", "title": "LiveCodeBench pass rate (%)", "scale": {"zero": False}}
    for sid, field, title, ticks in (
            ("solve-time", "s/solve", "wall seconds per problem solved (log)", [3, 10, 30, 100, 300, 1000]),
            ("solve-energy", "kJ/solve", "GPU kJ per problem solved (log)", [0.1, 0.3, 1, 3, 10, 30]),
            ("solve-tokens", "tok/solve", "tokens generated per problem solved (log)", [1000, 3000, 10000, 30000, 100000, 300000])):
        data = [dict(d) for d in solve if d.get(field)]
        front = pareto([(f"{d['pair']}|{d['mode']}", d["LCB %"], -d[field]) for d in data])
        for d in data:
            d["frontier"] = f"{d['pair']}|{d['mode']}" in front
        y = {"field": field, "type": "quantitative", "title": title, "scale": {"type": "log"},
             "axis": {"values": ticks, "grid": True}}
        specs[sid] = {
            "data": {"values": data}, "height": 340,
            "layer": [
                # one model's thinking off and on, joined
                {"mark": {"type": "line", "strokeWidth": 1.5},
                 "encoding": {"x": x_lcb, "y": y, "detail": {"field": "pair"}, "color": {"value": "MUTED"},
                              "opacity": {"condition": {"test": "hl == null || datum.family == hl", "value": 0.5}, "value": 0.1}}},
                {"mark": {"type": "point", "filled": True, "size": 75, "stroke": "SURFACE", "strokeWidth": 1.5},
                 "encoding": {"x": x_lcb, "y": y,
                              "color": {"field": "thinking", "type": "nominal", "scale": mode_scale,
                                        "legend": {"title": None, "orient": "top"}},
                              "shape": {"field": "tier", "type": "nominal", "scale": {"domain": ["full (175)", "quick (18)"], "range": ["circle", "diamond"]},
                                        "legend": {"title": "LiveCodeBench", "orient": "top", "symbolFillColor": "MUTED", "symbolStrokeColor": "MUTED"}},
                              "opacity": {"condition": {"test": "hl == null || datum.family == hl", "value": 1}, "value": 0.15},
                              "tooltip": solve_tip}},
                {"transform": [{"filter": "datum.frontier || datum.family == hl"}],
                 "mark": {"type": "text", "align": "left", "dx": 7, "dy": -4, "fontSize": 10.5},
                 "encoding": {"x": x_lcb, "y": y, "text": {"field": "model"}, "color": {"value": "TEXT"}}}]}
    # the optimal curve: a dashed line through the frontier points (nothing beats them on both axes), behind the
    # points, on every scatter that has one
    for sid in ("q-speed", "q-energy", "q-memory", "solve-time", "solve-energy", "solve-tokens", "tput-latency"):
        spec = specs.get(sid)
        if not spec or not any(d.get("frontier") for d in spec["data"]["values"]):
            continue
        points = next(l for l in spec["layer"] if (l.get("mark") or {}).get("type") == "point")
        x, y = dict(points["encoding"]["x"]), dict(points["encoding"]["y"])
        spec["layer"].insert(0, {
            "transform": [{"filter": "datum.frontier"}],
            "mark": {"type": "line", "strokeDash": [6, 4], "strokeWidth": 2, "interpolate": "linear"},
            "encoding": {"x": x, "y": y, "order": {"field": x["field"], "type": "quantitative"}, "color": {"value": "SERIES0"},
                         "opacity": {"value": 0.55}}})
    weights_text = ", ".join(f"{label[t]} {w:.0%}" for t, w in sorted(weights.items(), key=lambda kv: -kv[1]))
    h = ["<h2>Quality</h2>",
         "<figure><figcaption>Quality score: weighted accuracy across tasks</figcaption><div class=vega id=q-score></div></figure>",
         f"<div class=note>Weighted mean of the QualityBench quick-tier accuracies, each task weighted by how much the models "
         f"spread out on it relative to its mean (coefficient of variation): {esc(weights_text)}. LiveCodeBench tells the "
         f"models apart most, GSM8K least (nearly all ace it). Its 18 quick-tier problems leave about ±11 points of noise per "
         f"model, so a few points between two models is not a difference; the models with the full 175-problem run (lcb-full.yaml, about ±4) use it instead: their bars are blue, the "
         f"quick-tier-only ones pale, which are the less certain. Blue: thinking off; orange: thinking on. Models that always "
         f"think (Muse, gpt-oss: no switch) have an orange bar and a * before their name in every chart. For the models "
         f"that have the switch, the orange bar behind the blue one is the same model's score thinking on (thinking.yaml, "
         f"quick tier, long replies allowed): it shows above the blue only where thinking scores higher, and stays hidden "
         f"where it scores lower (the gap is in the tooltip and the thinking table). What thinking costs, in time and "
         f"energy, is in the tooltip and the thinking table. Hover a bar for the per-task scores.</div>",
         "<figure><figcaption>Quality vs single-user speed</figcaption><div class=vega id=q-speed></div></figure>",
         "<div class=note>Up and to the right is better. Labeled, on the dashed line: the optimal curve, the models no other model beats on both speed and quality. Hover any model to highlight its family (every Gemma, every Qwen...) in all the charts.</div>",
         "<h2>Throughput</h2>",
         "<figure><figcaption>Output tokens per second, all users, per pass</figcaption><div class=vega id=tput-total></div></figure>",
         "<figure><figcaption>Output tokens per second one user sees, per pass</figcaption><div class=vega id=tput-user></div></figure>",
         "<div class=note>Models ordered by their 32-user throughput. Per user: the line spans one model's dots, from one user "
         "alone to 32 sharing the server; a short line means batching costs each user little.</div>",
         "<figure><figcaption>Throughput vs latency</figcaption><div class=vega id=tput-latency></div></figure>",
         "<div class=note>One line per model, from one user (●) to 8 (■) and 32 (▲) sharing the server: total output "
         "tokens per second against the time each user waits between two tokens (log scale). Down and to the right is "
         "better; a line that climbs steeply buys its throughput with each user's speed. The dashed line: the optimal curve, the "
         "most throughput for each latency over every model and load. Hover a model to bring its family "
         "out and name its lines (at their 32-user end).</div>",
         "<h2>Cost per solved problem (LiveCodeBench)</h2>",
         "<figure><figcaption>Time to solution</figcaption><div class=vega id=solve-time></div></figure>",
         "<figure><figcaption>Energy to solution</figcaption><div class=vega id=solve-energy></div></figure>",
         "<figure><figcaption>Tokens to solution</figcaption><div class=vega id=solve-tokens></div></figure>",
         "<div class=note>What one solved LiveCodeBench problem cost each model: the run's wall time, GPU energy and "
         "generated tokens, over the problems it solved (a model that solves fewer pays for its failures too). Blue: "
         "thinking off, orange: thinking on, joined by a line for the models scored both ways on the same problems; one point per model and mode, the full run when there is one; ● the full 175-problem "
         "run, ◆ the quick tier's 18 (noisier). Across: the run's LiveCodeBench pass rate; up: its cost per solve. "
         "Bottom right is best (solves more, for less); labeled, on the dashed line: the optimal curve, the runs no other beats on both (for each pass rate, the cheapest way to get there). Wall time is with QualityBench's 16 requests in parallel, so it is the "
         "machine's time per solve, not one request's latency. Tokens: counted by vLLM (its periodic throughput log, "
         "integrated over the task); llama.cpp runs have no count yet. Costs on a log scale: thinking typically "
         "multiplies the tokens per solve, and the time and energy with them; the line shows what it bought in pass rate. "
         "Hover a model to bring out its family.</div>",
         "<h2>Memory</h2>",
         "<figure><figcaption>Quality vs memory</figcaption><div class=vega id=q-memory></div></figure>",
         "<figure><figcaption>Throughput vs memory</figcaption><div class=vega id=tput-memory></div></figure>",
         "<div class=note>Memory: what the model's weights take (vLLM's count, the speculative-decoding draft model "
         "included; llama.cpp: the GGUF file). The KV cache comes on top, a budget set per run, so it is not part of the "
         "model's footprint here; the machine's 119.6 GiB is shared with the CPU. Quality vs memory: up and to the left is "
         "better, labeled on the dashed line (the optimal curve) the models nothing smaller beats. Throughput vs memory: one model per vertical line, from one user "
         "(●) to 8 (■) and 32 (▲) sharing the server; small models with short lines are the fast ones. Hover a model to bring "
         "out its family.</div>",
         "<h2>Energy efficiency</h2>",
         "<figure><figcaption>Output tokens per joule, per pass</figcaption><div class=vega id=tput-energy></div></figure>",
         "<figure><figcaption>Quality vs energy efficiency</figcaption><div class=vega id=q-energy></div></figure>",
         "<div class=note>Generated tokens per joule of GPU energy (the GPU's own power reading over the pass; the rest of "
         "the machine is not counted), best model first. Quality vs efficiency: each model at its most efficient pass (nearly always 32 users); labeled on the dashed line (the optimal curve), the models nothing beats on both. Batching is what pays: the same model makes several times more "
         "tokens per joule at 32 users than for one. * before a name: the model always thinks (scored thinking on).</div>",
         VEGA,
         "<script>(function(){const specs=" + json.dumps(specs).replace("</", "<\\/") + ";"
         + "const families=" + json.dumps({r["model"]: r["family"] for r in quality + total + user}) + ";",
         r"""const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
function theme(){const dark=document.documentElement.dataset.theme==='dark'||(document.documentElement.dataset.theme!=='light'&&matchMedia('(prefers-color-scheme: dark)').matches);
 return {SERIES0:dark?'#3987e5':'#2a78d6',SERIES1:dark?'#d95926':'#eb6834',SERIES2:dark?'#199e70':'#1baf7a',SERIES3:dark?'#c98500':'#eda100',SERIES0PALE:dark?'#244463':'#b4d0f1',SERIES1PALE:dark?'#5c3420':'#f8cab8',
  TEXT:css('--fg'),MUTED:css('--muted'),SURFACE:css('--surface'),LINE:css('--line')};}
function fill(o,t){if(typeof o==='string')return t[o]??o;if(Array.isArray(o))return o.map(x=>fill(x,t));if(o&&typeof o==='object'){const r={};for(const k in o)r[k]=fill(o[k],t);return r;}return o;}
function render(){if(!window.vegaEmbed)return;const t=theme();for(const v of views)v.finalize();views=[];current=null;
 for(const id in specs){const el=document.getElementById(id);if(!el)continue;
  const spec=Object.assign({$schema:'https://vega.github.io/schema/vega-lite/v5.json',width:'container',autosize:{type:'fit-x',contains:'padding'}},fill(specs[id],t));
  spec.params=[{name:'hl',value:null}];spec.config={background:'transparent',font:'Inter, system-ui, sans-serif',view:{stroke:null},
   axis:{labelColor:t.MUTED,titleColor:t.MUTED,gridColor:t.LINE,domainColor:t.LINE,tickColor:t.LINE,labelFontSize:10.5,titleFontSize:11,titleFontWeight:600},
   axisX:{grid:false},legend:{labelColor:t.TEXT,titleColor:t.MUTED,titleFontSize:11,titleFontWeight:600,labelFontSize:11,symbolStrokeWidth:0},text:{font:'JetBrains Mono, ui-monospace, monospace'}};
  vegaEmbed(el,spec,{actions:false,renderer:'svg'}).then(r=>{views.push(r.view);
   const pick=(e,item)=>{const d=item&&item.datum;let m=d&&(d.family??(item.mark&&item.mark.role==='axis-label'?families[d.value]:undefined));highlight(m??null);};
   r.view.addEventListener('pointermove',pick);r.view.addEventListener('pointerout',()=>highlight(null));})
   .catch(e=>{el.textContent='chart failed to render: '+e.message;});}}
// one model family highlighted across every chart (every Gemma, every Qwen...): hover a bar, a dot or an axis label
let views=[],current=null;
function highlight(m){if(m===current)return;current=m;for(const v of views)v.signal('hl',m).runAsync();}
render();matchMedia('(prefers-color-scheme: dark)').addEventListener('change',render);
new MutationObserver(render).observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});
})();</script>"""]
    return "\n".join(h)


TIMEOUT_END = "timeout"  # qb_launch.py's finish_reason past the window

# --- time to solution per reply (qb_launch.py's timing log) ----------------------------------------

LOOP_RATIO = 0.10  # zlib ratio of a reply's last 20k characters below this: a loop (normal answers 0.25-0.4)


def reply_timings(run_dir: str) -> list[dict]:
    """The full LiveCodeBench runs with a timing log: per reply, minutes from request to end, how it ended, loop
    or not, and solved or not (LiveCodeBench's grading, joined on the prompt's hash: lcb_prompt_sha1.json maps each
    problem's question_id to the hash of its messages, rebuilt with lcb_prompt_sha1.py)."""
    import glob

    try:
        ids = {v: k for k, v in json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                          "lcb_prompt_sha1.json"))).items()}
    except (OSError, ValueError):
        ids = {}
    runs = []
    for log in sorted(glob.glob(os.path.join(run_dir, "*.qualitybench", "*.qb", "timing", "lcb.jsonl"))):
        qb = os.path.dirname(os.path.dirname(log))
        bench = os.path.basename(os.path.dirname(qb)).removesuffix(".qualitybench")
        head, _ = split(bench)
        if not head.endswith("-lcbfull"):
            continue
        try:
            graded = {e["question_id"]: bool(e.get("graded_list") and e["graded_list"][0])
                      for e in json.load(open(os.path.join(qb, "lcb", "custom_codegeneration_output_eval_all.json")))}
        except (OSError, ValueError, KeyError):
            graded = {}
        replies = []
        for line in open(log):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if not r.get("t_start") or not r.get("t_end"):
                continue
            qid = ids.get(r.get("prompt_sha1"))
            replies.append({"min": (r["t_end"] - r["t_start"]) / 60, "end": r.get("finish_reason"),
                            "loop": (r.get("loop_ratio") or 1) < LOOP_RATIO, "qid": qid,
                            "ok": graded.get(qid) if graded else None})
        if replies:
            name = head.removesuffix("-lcbfull").replace("homelab-", "").replace("vllm-", "").replace("llamacpp-", "llama.cpp ")
            runs.append({"name": name.replace("-think", " (thinking)"), "bench": head, "replies": replies,
                         "graded": bool(graded), "running": not graded and head in live_benches(os.path.basename(run_dir.rstrip("/")))})
    return runs


# hover a model in the legend: its line stays, the others fade (empty: every line at full strength)
TIMING_HOVER = {"params": [{"name": "hl", "select": {"type": "point", "fields": ["name"], "on": "mouseover", "clear": "mouseout"},
                            "bind": "legend"}]}
TIMING_EMPH = {"opacity": {"condition": {"param": "hl", "empty": True, "value": 1}, "value": 0.12},
               "strokeWidth": {"condition": {"param": "hl", "empty": False, "value": 3.5}, "value": 2.25}}

# the page's chart style draws legend symbols with no stroke (its scatters use filled points); a line chart's
# swatch is a stroke, so these legends set their own: one row per model, a thick short line in its color
TIMING_LEGEND = {"title": None, "orient": "bottom", "direction": "vertical", "columns": 2, "labelLimit": 400,
                 "symbolType": "stroke", "symbolStrokeWidth": 4, "symbolSize": 300, "labelFontSize": 11.5}


def timing_section(run_dir: str) -> str:
    runs = reply_timings(run_dir)
    if not runs:
        return ""
    specs, cdf, curve = {}, [], []
    rows = []
    for run in runs:
        rs = run["replies"]
        done = sorted(r["min"] for r in rs if r["end"] not in (TIMEOUT_END, "error"))
        back = sorted(r["min"] for r in rs if r["end"] != TIMEOUT_END)  # a timed-out reply never comes back
        for i, m in enumerate(back):
            cdf.append({"name": run["name"], "min": round(m, 2), "share": round(100 * (i + 1) / 175, 2)})
        if back and run["graded"]:  # flat to the window: the gap below 100% is what never came back
            cdf.append({"name": run["name"], "min": 60, "share": round(100 * len(back) / 175, 2)})
        if run["graded"]:
            for limit in range(0, 61):
                ok = sum(1 for r in rs if r["ok"] and r["min"] <= limit)
                curve.append({"name": run["name"], "limit": limit, "acc": round(100 * ok / 175, 1)})
        q = lambda p: done[min(len(done) - 1, int(p * len(done)))] if done else None
        capped = [r for r in rs if r["end"] == "length"]
        at = lambda limit: (100 * sum(1 for r in rs if r["ok"] and r["min"] <= limit) / 175) if run["graded"] else None
        queued = queue_wait(run_dir, run["bench"])
        flags = (" <span class=run>running</span>" if run["running"] else " <span class=na>stopped</span>" if not run["graded"] else "")
        if queued:
            flags += (f" <span class=bad title='the server decodes {queued} requests at a time and QualityBench sent 16: "
                      "the others waited in its queue with their hour running, so these times and timeouts include queue "
                      "wait (fixed for later runs)'>queue wait</span>")
        rows.append(f"<tr><td>{esc(run['name'])}{flags}</td>"
                    f"<td>{len(rs)}</td><td>{fmt(q(0.5), 1)}</td><td>{fmt(q(0.9), 1)}</td>"
                    f"<td>{len(capped)}</td><td>{sum(r['loop'] for r in capped)}</td>"
                    f"<td>{sum(r['end'] == TIMEOUT_END for r in rs)}</td>"
                    + "".join(f"<td>{fmt(at(x), 1)}</td>" for x in (10, 20, 30, 60)) + "</tr>")
    # one color per model, the same in both charts (a fixed domain), and 20 distinct ones before any repeats
    color_scale = {"domain": [run["name"] for run in runs], "scheme": "tableau20"}
    specs["t-cdf"] = {"data": {"values": cdf}, "height": 300, **TIMING_HOVER, "mark": {"type": "line", "interpolate": "step-after", "strokeWidth": 2.25},
                      "encoding": {**TIMING_EMPH, "x": {"field": "min", "type": "quantitative", "title": "minutes from request to reply", "scale": {"domain": [0, 62]}},
                                   "y": {"field": "share", "type": "quantitative", "title": "% of the 175 replies back", "scale": {"domain": [0, 100]}},
                                   "color": {"field": "name", "type": "nominal", "scale": color_scale, "legend": TIMING_LEGEND},
                                   "tooltip": [{"field": "name"}, {"field": "min", "title": "minutes", "format": ".1f"},
                                               {"field": "share", "title": "% back", "format": ".1f"}]}}
    out = ["<h2>Time to solution per reply</h2><div class=note>The full LiveCodeBench runs with a per-reply timing log "
           "(from 2026-11-18): minutes from a request to its reply, with every other request of the run in flight "
           "(vLLM 16, llama.cpp as many as its slots). A reply cut at the token cap counts as back, a timed-out one "
           "never comes back. The second chart is the score the model would get with a shorter time-to-solution "
           "window: problems solved by a reply back within that many minutes. The third puts both together, one point "
           "per model: its score with the full hour, against the minutes it takes for 90% of that score to come back; "
           "top left is fast and right, and the dashed line joins the models nothing beats on both.</div>",
           figure("t-cdf", "Replies back within N minutes")]
    if curve:
        specs["t-acc"] = {"data": {"values": curve}, "height": 300, **TIMING_HOVER, "mark": {"type": "line", "interpolate": "step-after", "strokeWidth": 2.25},
                          "encoding": {**TIMING_EMPH, "x": {"field": "limit", "type": "quantitative", "title": "time-to-solution window, minutes"},
                                       "y": {"field": "acc", "type": "quantitative", "title": "LiveCodeBench pass rate, %", "scale": {"zero": True}},
                                       "color": {"field": "name", "type": "nominal", "scale": color_scale,
                                                 "legend": {**TIMING_LEGEND, "values": sorted({d["name"] for d in curve}, key=[r["name"] for r in runs].index)}},
                                       "tooltip": [{"field": "name"}, {"field": "limit", "title": "window, min"},
                                                   {"field": "acc", "title": "pass rate %", "format": ".1f"}]}}
        out.append(figure("t-acc", "Coding score with a shorter window"))
        # fast and right: per model, the minutes until 90% of its final score is in (the curve above), and that score
        fast = []
        for run in runs:
            if not run["graded"]:
                continue
            at = lambda limit: 100 * sum(1 for r in run["replies"] if r["ok"] and r["min"] <= limit) / 175
            final = at(60)
            if final <= 0:
                continue
            t90 = next(limit for limit in range(0, 61) if at(limit) >= 0.9 * final)
            fast.append({"name": run["name"], "t90": t90, "score": round(final, 1),
                         "at10": round(at(10), 1), "at20": round(at(20), 1), "at30": round(at(30), 1)})
        front = pareto([(d["name"], -d["t90"], d["score"]) for d in fast])
        for d in fast:
            d["frontier"] = d["name"] in front
        labels = []  # points at about the same spot share one label (names joined), so none overprints another
        for d in sorted(fast, key=lambda d: -d["score"]):
            near = next((g for g in labels if abs(g["t90"] - d["t90"]) <= 1.5 and abs(g["score"] - d["score"]) <= 1.5), None)
            if near:
                near["name"] += " · " + d["name"]
            else:
                labels.append({"name": d["name"], "t90": d["t90"], "score": d["score"]})
        if fast:
            fx = {"field": "t90", "type": "quantitative", "title": "minutes until 90% of its score is in (lower is better)",
                  "scale": {"domain": [0, 62]}}
            fy = {"field": "score", "type": "quantitative", "title": "LiveCodeBench pass rate, % (1 h window)",
                  "scale": {"zero": False}}
            specs["t-fast"] = {"data": {"values": fast}, "height": 320, "layer": [
                {"transform": [{"filter": "datum.frontier"}], "mark": {"type": "line", "strokeDash": [6, 4], "strokeWidth": 2},
                 "encoding": {"x": fx, "y": fy, "order": {"field": "t90", "type": "quantitative"}, "color": {"value": "MUTED"}}},
                {"mark": {"type": "point", "filled": True, "size": 110, "stroke": "SURFACE", "strokeWidth": 2},
                 "encoding": {"x": fx, "y": fy, "color": {"field": "name", "type": "nominal", "scale": color_scale, "legend": None},
                              "tooltip": [{"field": "name", "title": "model"}, {"field": "score", "title": "score at 60 min"},
                                          {"field": "t90", "title": "minutes to 90% of it"}, {"field": "at10", "title": "score at 10 min"},
                                          {"field": "at20", "title": "score at 20 min"}, {"field": "at30", "title": "score at 30 min"}]}},
                {"data": {"values": labels}, "mark": {"type": "text", "align": "left", "dx": 8, "dy": -6, "fontSize": 10},
                 "encoding": {"x": fx, "y": fy, "text": {"field": "name"}, "color": {"value": "TEXT"}}}]}
            out.append(figure("t-fast", "Fast and right: score against the wait for most of it"))
    out.append("<div class=wrap><table><tr><th>model</th><th>replies</th><th>min p50</th><th>min p90</th>"
               "<th>cut at cap</th><th>of which loops</th><th>timed out</th><th>score at 10 min</th><th>20 min</th>"
               "<th>30 min</th><th>60 min</th></tr>" + "".join(rows) + "</table></div>"
               f"<div class=note>min p50 / p90: minutes of the replies that came back. Loops: a cut reply whose last 20k "
               f"characters compress below {LOOP_RATIO} (repeating itself); the others were still working when cut. "
               "Score at N min: % of the 175 problems solved by a reply back within N minutes.</div>")
    out.append(UC_JS.replace("__SPECS__", json.dumps(specs).replace("</", "<\\/")))
    return "".join(out)


TB_TOTAL = 89  # Terminal-Bench 2.0 tasks
TB_PARTS = [("solved", "SERIES0"), ("failed", "MUTED"), ("agent timed out", "SERIES3"), ("error", "SERIES1"),
            ("not run yet", "LINE")]


def tb_trials(job_dir: str) -> list[dict]:
    """One dict per finished trial of a Harbor job folder (its <trial>/result.json): works while the job runs."""
    out = []
    for path in sorted(glob.glob(os.path.join(job_dir, "*", "result.json"))):
        try:
            t = json.load(open(path))
        except (OSError, ValueError):
            continue
        if not t.get("finished_at"):
            continue
        reward = float(((t.get("verifier_result") or {}).get("rewards") or {}).get("reward", 0.0))
        agent = t.get("agent_result") or {}
        out.append({"reward": reward, "exception": (t.get("exception_info") or {}).get("exception_type"),
                    "in": agent.get("n_input_tokens") or 0, "out": agent.get("n_output_tokens") or 0,
                    "end": t["finished_at"]})
    return out


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if not n:
        return 0.0, 0.0
    p, d = k / n, 1 + z * z / n
    c, w = (p + z * z / (2 * n)) / d, z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return max(0.0, c - w), min(1.0, c + w)


def tb_section(run_dir: str, models: list[dict], weights: dict[str, float]) -> str:
    """Terminal-Bench 2.0 (terminalbench.yaml: the top models only, Terminus-2 agent): solved / failed / timed out
    per model, over the 89 tasks. Not part of the quality score (most models don't have it)."""
    live = live_benches(os.path.basename(run_dir.rstrip("/")))
    by_name = {}
    for m in models:
        name = re.sub(r"^homelab-(vllm-|llamacpp-)?", "", m["bench"])
        by_name[name] = m
    data, rows = [], []
    for folder in sorted(glob.glob(os.path.join(run_dir, "homelab-tb2-*.terminalbench.terminalbench"))):
        bench = os.path.basename(folder).split(".")[0]
        name = bench[len("homelab-tb2-"):]
        if name.startswith("smoke-"):
            continue
        job = os.path.join(folder, bench + ".terminalbench")
        trials = tb_trials(job)
        if not trials:
            continue
        done = len(trials)
        state = "running" if bench in live else "done" if os.path.exists(os.path.join(folder, "trials.jsonl")) else "stopped"
        if state != "done":  # finished runs only: a partial rate over a few tasks reads as a result
            continue
        try:
            jr = json.load(open(os.path.join(job, "result.json")))
            t0 = datetime.datetime.fromisoformat(jr["started_at"])
            # unfinished: up to now while it runs, else to its last finished task
            t1 = (datetime.datetime.fromisoformat(jr["finished_at"]) if jr.get("finished_at")
                  else datetime.datetime.now() if state == "running"
                  else datetime.datetime.fromisoformat(max(t["end"] for t in trials)))
            hours = (t1 - t0).total_seconds() / 3600
        except (OSError, ValueError, KeyError, TypeError):
            hours = None
        solved = sum(t["reward"] >= 1.0 for t in trials)
        timeouts = sum(t["reward"] < 1.0 and t["exception"] == "AgentTimeoutError" for t in trials)
        errors = sum(t["reward"] < 1.0 and t["exception"] not in (None, "AgentTimeoutError") for t in trials)
        counts = {"solved": solved, "agent timed out": timeouts, "error": errors,
                  "failed": done - solved - timeouts - errors, "not run yet": max(0, TB_TOTAL - done)}
        lo, hi = _wilson(solved, done)
        m = by_name.get(name)
        quality = quality_score(m, weights) if m else None
        coding = (acc(m, "lcb_full") or acc(m, "lcb")) if m else None
        coding = coding * 100 if coding is not None and coding <= 1 else coding
        label = name + ("" if state == "done" else f"  ({state}, {done}/{TB_TOTAL})")
        info = {"model": label, "family": (re.match(r"[a-z]+", name) or [name])[0],
                "rate": round(100 * solved / done, 1) if done else None,
                "ci": f"{100 * lo:.0f}–{100 * hi:.0f}%", "done": done, "state": state,
                "tokens": f"{sum(t['in'] for t in trials) / 1e6:.1f} M in, {sum(t['out'] for t in trials) / 1e6:.2f} M out",
                "hours": round(hours, 1) if hours is not None else None,
                "quality": round(quality, 1) if quality is not None else None,
                "coding": round(coding, 1) if coding is not None else None}
        rows.append(info)
        for i, (part, _) in enumerate(TB_PARTS):
            data.append({**info, "part": part, "order": i, "n": counts[part]})
    if not rows:
        return ""
    # finished runs first, best solve rate on top; partial runs after, their rate over the tasks done so far
    order = [r["model"] for r in sorted(rows, key=lambda r: (["done", "running", "stopped"].index(r["state"]), -(r["rate"] or 0)))]
    y = {"field": "model", "type": "nominal", "sort": order, "title": None, "axis": {"labelLimit": 360}}
    tip = [{"field": "model"}, {"field": "part", "title": "tasks"}, {"field": "n", "title": "count"},
           {"field": "rate", "title": "solved, % of tasks run"}, {"field": "ci", "title": "95% interval"},
           {"field": "done", "title": f"tasks run (of {TB_TOTAL})"}, {"field": "hours", "title": "wall hours"},
           {"field": "tokens", "title": "agent tokens"}, {"field": "coding", "title": "LiveCodeBench %"},
           {"field": "quality", "title": "quality score"}]
    spec = {"height": {"step": 26}, "data": {"values": data}, "layer": [
        {"mark": {"type": "bar", "stroke": "SURFACE", "strokeWidth": 1.5},
         "encoding": {"y": y, "x": {"field": "n", "type": "quantitative", "stack": "zero", "title": f"tasks (of {TB_TOTAL})",
                                    "scale": {"domain": [0, TB_TOTAL], "nice": False}},
                      "color": {"field": "part", "type": "nominal", "title": None,
                                "scale": {"domain": [p for p, _ in TB_PARTS], "range": [c for _, c in TB_PARTS]},
                                "legend": {"orient": "bottom", "direction": "horizontal"}},
                      "order": {"field": "order", "type": "quantitative"},
                      "opacity": {"condition": {"test": "datum.part == 'failed'", "value": 0.45}, "value": 1},
                      "tooltip": tip}}]}
    spec["layer"].append({"transform": [{"filter": "datum.part == 'solved' && datum.n >= 6"},
                                       {"calculate": "format(datum.rate, '.0f') + '%'", "as": "pct"}],
                        "mark": {"type": "text", "align": "right", "dx": -5, "fontSize": 10.5, "fontWeight": 600},
                        "encoding": {"y": y, "x": {"field": "n", "type": "quantitative"}, "color": {"value": "SURFACE"},
                                     "text": {"field": "pct"}}})
    return ("<h3>Agentic coding: Terminal-Bench 2.0</h3>"
            + figure("q-tb", "Terminal-Bench 2.0: tasks solved, per model")
            + f"<div class=note>The {TB_TOTAL} tasks of Terminal-Bench 2.0, each a real job in its own container "
            "(build a project, fix a bug, train a small model...), done by the Terminus-2 agent driving a shell with the "
            "model, 20 minutes or so per task (the task's own limit), 4 tasks at a time. Only the best models run it "
            "(terminalbench.yaml: one build per model, about 7 h each), so it is not part of the quality score. "
            "The number on a bar: the share solved. "
            "Timed out: the agent was still working at the task's limit (a slow model loses tasks here); error: "
            "the agent or its container failed. Finished runs only: a model shows up once all its tasks are done. With 89 tasks, a model's rate is about ±10 points (the 95% "
            "interval is in the tooltip), so close bars are not a difference. Hover for tokens, wall time and the "
            "model's LiveCodeBench and quality scores.</div>"
            + UC_JS.replace("__SPECS__", json.dumps({"q-tb": spec}).replace("</", "<\\/")))


def release_section(models: list[dict], weights: dict[str, float]) -> str:
    """Quality score by the model's release date (release_dates.json, from release_dates.py): one point per set of
    weights, its best-scoring build here, and the best score any model released by that date reaches."""
    try:
        dates = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "release_dates.json")))
    except (OSError, ValueError):
        return ""
    best = {}
    for m in models:
        key, score = base_weights(m), quality_score(m, weights)
        if score is None or key not in dates:
            continue
        if key not in best or score > best[key]["score"]:
            build = m["bench"].replace("homelab-", "").replace("vllm-", "").replace("llamacpp-", "llama.cpp ")
            best[key] = {"model": key, "family": (re.match(r"[a-z]+", key) or [key])[0], "date": dates[key]["date"],
                         "score": round(score, 1), "build": ("* " if thinking_only(m) else "") + build,
                         "repo": dates[key]["repo"]}
    points = sorted(best.values(), key=lambda d: d["date"])
    if len(points) < 3:
        return ""
    day = lambda d: datetime.date.fromisoformat(d["date"]).toordinal()
    start_d = day(points[0]) - 30
    end_d = datetime.date.today().toordinal() + 200
    lo, hi = min(d["score"] for d in points), max(d["score"] for d in points)
    # labels placed greedily, highest score first, on an ~820 x 375 px plot: above, below, or left of the point,
    # the first spot whose box hits no label already placed (boxes in px, ~6.2 px per character, 11 px tall)
    px = lambda d: (day(d) - start_d) / (end_d - start_d) * 820
    py = lambda d: (hi + 2 - d["score"]) / (hi - lo + 4) * 375
    boxes = [(px(d) - 5, py(d) - 5, px(d) + 5, py(d) + 5) for d in points]  # the dots: labels avoid them too
    hit = lambda a, b: max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    for d in sorted(points, key=lambda d: -d["score"]):
        w, x0, y0 = len(d["model"]) * 6.2, px(d), py(d)
        spots = {"above": (x0 + 7, y0 - 17), "below": (x0 + 7, y0 + 4), "left": (x0 - 7 - w, y0 - 6),
                 "above-left": (x0 - 7 - w, y0 - 17), "below-left": (x0 - 7 - w, y0 + 4)}
        own = (x0 - 5, y0 - 5, x0 + 5, y0 + 5)
        cost = {pos: sum(hit((bx, by, bx + w, by + 12), b) for b in boxes if b != own) for pos, (bx, by) in spots.items()}
        d["pos"] = min(spots, key=lambda pos: (cost[pos] > 0, cost[pos]))  # the first free spot, else the least covered
        bx, by = spots[d["pos"]]
        boxes.append((bx, by, bx + w, by + 12))
    sota, top = [], None
    for d in points:  # the best quality reachable with a model released by then
        if top is None or d["score"] > top:
            top = d["score"]
            sota.append({"date": d["date"], "score": top, "model": d["model"]})
    sota.append({"date": datetime.date.today().isoformat(), "score": top, "model": sota[-1]["model"]})
    end = datetime.date.fromordinal(end_d).isoformat()  # room for the last labels
    start = datetime.date.fromordinal(start_d).isoformat()
    x = {"field": "date", "type": "temporal", "title": "release date of the weights",
         "scale": {"domain": [start, end]}, "axis": {"format": "%b %Y", "labelAngle": 0}}
    y = {"field": "score", "type": "quantitative", "title": "quality score", "scale": {"zero": False}}
    spec = {"height": 380, "layer": [
        {"data": {"values": sota}, "mark": {"type": "line", "interpolate": "step-after", "strokeDash": [6, 4], "strokeWidth": 2},
         "encoding": {"x": x, "y": y, "color": {"value": "MUTED"}}},
        {"data": {"values": points}, "mark": {"type": "point", "filled": True, "size": 90, "stroke": "SURFACE", "strokeWidth": 2},
         "encoding": {"x": x, "y": y, "color": {"field": "family", "type": "nominal", "scale": {"scheme": "tableau10"},
                                                  "legend": {"title": None, "orient": "bottom", "columns": 6}},
                      "tooltip": [{"field": "model", "title": "weights"}, {"field": "date", "type": "temporal", "format": "%Y-%m-%d", "title": "released"},
                                  {"field": "score", "title": "quality score"}, {"field": "build", "title": "best build here"},
                                  {"field": "repo", "title": "original repo"}]}},
        *({"data": {"values": points}, "transform": [{"filter": f"datum.pos == '{pos}'"}],
           "mark": {"type": "text", "align": align, "dx": dx, "dy": dy, "fontSize": 10},
           "encoding": {"x": x, "y": y, "text": {"field": "model"}, "color": {"value": "TEXT"}}}
          for pos, align, dx, dy in (("above", "left", 7, -6), ("below", "left", 7, 11), ("left", "right", -7, 0),
                                     ("above-left", "right", -7, -6), ("below-left", "right", -7, 11)))]}
    return ("<h2>Quality by release date</h2><div class=note>Each point is one set of weights at its best-scoring build "
            "on this machine (the quality score of the leaderboard), placed at the day the original weights were "
            "published on Hugging Face (quantizations followed back to what they quantize; a fine-tune is a release "
            "of its own). The dashed line is the best score reachable with a model released by then.</div>"
            + figure("q-release", "Quality score by release date")
            + UC_JS.replace("__SPECS__", json.dumps({"q-release": spec}).replace("</", "<\\/")))


def reply_cap(run_dir: str, bench: str) -> str | None:
    """The thinking-on run's reply cap (tokens), from its QualityBench results file; None when there is none (the
    first runs, at 8k, whose files are archived)."""
    import glob

    for path in glob.glob(os.path.join(run_dir, f"{bench}-think.quality.qualitybench", "*.thinking-on.json")):
        try:
            cap = json.load(open(path)).get("env", {}).get("thinking_on_min_max_tokens")
        except (OSError, ValueError):
            continue
        if cap:
            return f"{int(cap) // 1024}k"
    return None


THINKING_LOWER = """<div class=note><b>Why thinking on sometimes scores lower.</b> Thinking does not make a model worse
at the task; in these runs a lower score has three causes, the first by far the largest:
<ol>
<li><b>The thinking budget.</b> The reasoning and the answer share one limit on the reply's length (the reply cap).
A model that reasons past it is cut off before it writes the answer, and that item scores as wrong, however good the
reasoning was. The hard items are the ones it thinks longest on, so the cap takes exactly those: LiveCodeBench and
MMLU-Pro lose most. Qwen3.6-35B, for one: LiveCodeBench 44% thinking off, 22% thinking on at an 8k cap, 39% at 24k,
with 11 of its 18 replies still cut. The <i>replies cut</i> column counts them; the reruns raise the cap to 32k
(QualityBench's default), and what a model still cannot fit there is a real cost of its thinking on this budget.</li>
<li><b>Loops.</b> Scored at temperature 0 (greedy, the same for every model), a model can get stuck checking its own
draft over and over until the cap: Gemma 4 26B (see its note). Sampling avoids it, at the cost of repeatable scores.</li>
<li><b>Noise.</b> The quick tier scores 18 LiveCodeBench problems: one problem is 5.6 points, and its interval is about
±11 points. A gap of a few points between thinking off and on is not a difference.</li>
</ol>
Read a thinking-on drop with the cut replies next to it: when many replies were cut, the score measures the budget,
not the model.</div>"""


def lcb_full_timeouts(run_dir: str, bench: str, think: bool = False) -> int:
    """Replies of the model's full LiveCodeBench run (lcb-full.yaml) not back within the time-to-solution window:
    scored as wrong, so a slow model's full score is time-bounded (qb_launch.py counts them in state.json)."""
    import glob

    # an always-thinking model: either name; an always-reasoning one's "off" is its effort-none run
    suffixes = ("-lcbfull", "-think-lcbfull") if think else ("-lcbfull", "-effort-none-lcbfull")
    for path in (p for x in suffixes
                 for d in (f"{bench}{x}.lcbfull.qualitybench", f"{bench}{x}.qualitybench")  # vLLM pass / llama.cpp bench
                 for p in glob.glob(os.path.join(run_dir, d, "*.qb", "state.json"))):
        try:
            caps = json.load(open(path)).get("caps") or {}
        except (OSError, ValueError):
            continue
        return sum(c.get("timeouts", 0) for c in caps.values())
    return 0


def queue_wait(run_dir: str, bench: str) -> int | None:
    """llama-server slots when the bench's QualityBench sent more requests at once than the server decodes (before
    benchmarks/llamacpp/main.py capped --concurrency at --parallel, 2026-11-19): the extra requests waited in
    llama-server's queue with their time-to-solution clock running, so its timeouts include queue time."""
    import glob

    for path in glob.glob(os.path.join(run_dir, f"{bench}.*.data")):
        try:
            text = open(path, errors="replace").read()
        except OSError:
            continue
        slots = re.search(r'"--parallel", "(\d+)"', text)
        cmd = re.search(r"QUALITY: [^\n{]*", text)
        if not slots or not cmd:
            continue
        conc = re.search(r"--concurrency (\d+)", cmd.group(0))
        if int(conc.group(1) if conc else 16) > int(slots.group(1)):
            return int(slots.group(1))
    return None


def replies_cut(run_dir: str, bench: str) -> tuple[int, int, int] | None:
    """The thinking-on run's replies stopped at the reply cap, out of all, and those past the time-to-solution
    window (QualityBench's state.json; qb_launch.py counts the timeouts)."""
    import glob

    for path in glob.glob(os.path.join(run_dir, f"{bench}-think.quality.qualitybench", "*.qb", "state.json")):
        try:
            caps = json.load(open(path)).get("caps") or {}
        except (OSError, ValueError):
            continue
        if caps:
            return (sum(c.get("length", 0) for c in caps.values()), sum(c.get("replies", 0) for c in caps.values()),
                    sum(c.get("timeouts", 0) for c in caps.values()))
    return None


def thinking_section(models: list[dict], weights: dict[str, float], run_dir: str = "") -> str:
    """Models scored both ways (thinking.yaml): accuracy off vs on, and the accuracy pass's time and energy."""
    both = [m for m in models if any(acc(m, t, "off") is not None for t in weights)
            and any(acc(m, t, "on") is not None for t in weights)]
    if not both:
        return ("<h2>Thinking on vs off</h2><p class=sub>No model scored both ways yet (config/homelab/thinking.yaml "
                "runs the models whose chat template has a thinking switch).</p>")
    label = dict(TASKS)

    score = lambda m, mode: mode_score(m, weights, mode)

    h = ["<h2>Thinking on vs off</h2><div class=wrap><table><tr><th>model</th><th class=grp>score off</th><th>score on</th>"
         "<th>gain</th><th>reply cap</th><th>replies cut</th>" + "".join(f"<th>{esc(label[t])} off→on</th>" for t in weights)
         + "<th class=grp>time off</th><th>time on</th><th>× time</th><th>energy off kJ</th><th>energy on kJ</th><th>× energy</th></tr>"]
    caveats = {}
    for m in both:
        off, on = score(m, "off"), score(m, "on")
        cap = reply_cap(run_dir, m["bench"]) or "8k"
        cut = replies_cut(run_dir, m["bench"])
        cut_cell = fmt(None) if not cut or not cut[1] else (
            f"<span class={'bad' if cut[0] / cut[1] > 0.05 else ''}>{cut[0]} of {cut[1]}</span>"
            + (f" <span class=bad title='replies not back within the time-to-solution window (1 h): scored as wrong'>"
               f"+ {cut[2]} timed out</span>" if cut[2] else ""))
        mark = ""
        for prefix, why in THINK_CAVEAT.items():
            if m["bench"].startswith(prefix):
                caveats.setdefault(prefix, (len(caveats) + 1, why))
                mark = f" <sup>{caveats[prefix][0]}</sup>"
        (t_off, e_off), (t_on, e_on) = m.get("cost.off", (None, None)), m.get("cost.on", (None, None))
        ratio = lambda a, b: f"{b / a:.1f}×" if a and b else fmt(None)
        gain = (on - off) if off is not None and on is not None else None
        h.append(f"<tr><td>{esc(m['bench'].replace('homelab-', ''))}{mark}</td><td class=grp>{fmt(off)}</td><td>{fmt(on)}</td>"
                 f"<td class={'ok' if (gain or 0) > 0 else 'bad' if (gain or 0) < 0 else ''}>{fmt(gain, 1, suffix='') if gain is None else f'{gain:+.1f}'}</td><td>{cap}</td><td>{cut_cell}</td>"
                 + "".join(f"<td>{fmt(acc(m, t, 'off'), 0, 100)}→{fmt(acc(m, t, 'on'), 0, 100)}</td>" for t in weights)
                 + f"<td class=grp>{fmt(t_off and t_off / 60, 0, suffix=' min')}</td><td>{fmt(t_on and t_on / 60, 0, suffix=' min')}</td>"
                 f"<td>{ratio(t_off, t_on)}</td><td>{fmt(e_off and e_off / 1000, 0)}</td><td>{fmt(e_on and e_on / 1000, 0)}</td>"
                 f"<td>{ratio(e_off, e_on)}</td></tr>")
    h.append("</table></div><div class=note>The same QualityBench quick-tier tasks on the same server, thinking off and on "
             "(reply cap: the most tokens a reply, reasoning included, may take; one cut there scores as wrong. The reruns of "
             "2026-11-15 use 24k in a 32k context; rows still at 8k are the first runs, cut short on hard items). Score: the quality score's weights. Time and "
             "energy: the whole accuracy pass, the price of the gain on this machine. The main tables and charts use the "
             "thinking-off scores (models that cannot turn it off are scored on, marked *). "
             + "".join(f"Left out: {esc(p.replace('homelab-vllm-', ''))} ({esc(why)}). " for p, why in THINK_INVALID.items())
             + "".join(f"<br><sup>{n}</sup> {esc(why)}." for n, why in caveats.values()) + "</div>")
    h.append(THINKING_LOWER)
    return "\n".join(h)


GEN_DIR = os.environ.get("MILABENCH_COMFY_OUTPUT", "/opt/milabench/extra/comfyui/ComfyUI/output")
GEN_KINDS = {".png": "image", ".jpg": "image", ".jpeg": "image", ".webp": "image",
             ".mp4": "video", ".webm": "video", ".flac": "audio", ".wav": "audio", ".mp3": "audio", ".ogg": "audio"}
SRC_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def _bench_argv(run_dir: str, bench: str) -> dict:
    path = os.path.join(run_dir, f"{bench}.0.data")
    try:
        with open(path, errors="replace") as f:
            first = json.loads(f.readline().replace("\0", ""))
        return (first.get("data") or {}).get("argv") or {}
    except (OSError, ValueError):
        return {}


def _prompt(run_dir: str, bench: str) -> str | None:
    """The workflow's positive prompt (the first long text input)."""
    workflow = _bench_argv(run_dir, bench).get("--workflow")
    if not workflow:
        return None
    path = os.path.join(SRC_ROOT, "benchmarks", "comfyui", workflow)
    try:
        graph = json.load(open(path))
    except (OSError, ValueError):
        return None
    for node in graph.values():
        for key in ("text", "prompt", "tags", "positive_prompt"):
            value = (node.get("inputs") or {}).get(key) if isinstance(node, dict) else None
            if isinstance(value, str) and len(value) > 10:
                return value
    return None


def _samples(run_dir: str, row: dict, limit: int) -> list[str]:
    """A bench's generations: <bench>.outputs/ (the bench keeps its first few), else the ComfyUI output files
    written during its run (older runs kept none)."""
    bench = row["bench"]
    kept = os.path.join(run_dir, f"{bench}.outputs")
    if os.path.isdir(kept):
        files = sorted(os.path.join(kept, f) for f in os.listdir(kept))
    else:
        start, dur = row.get("start"), row.get("duration_s") or 0
        if not start or not os.path.isdir(GEN_DIR):
            return []
        files = []
        for root, _, names in os.walk(GEN_DIR):
            for n in names:
                f = os.path.join(root, n)
                if start - 5 <= os.path.getmtime(f) <= start + dur + 60:
                    files.append(f)
        files.sort(key=os.path.getmtime)
    files = [f for f in files if os.path.splitext(f)[1].lower() in GEN_KINDS]
    if len(files) > limit:  # spread over the run, not only its first seconds
        step = len(files) / limit
        files = [files[int(i * step)] for i in range(limit)]
    return files


def _publish_sample(src: str, out_dir: str, bench: str) -> tuple[str, str]:
    """(kind, path relative to the page): images as 512 px JPEG thumbnails, videos and audio copied."""
    import shutil

    kind = GEN_KINDS[os.path.splitext(src)[1].lower()]
    folder = os.path.join(out_dir, "gallery", bench)
    stale = lambda dest: os.path.exists(dest) and os.path.getmtime(dest) < os.path.getmtime(src)  # a rerun's file, same name
    os.makedirs(folder, exist_ok=True)
    stem = os.path.splitext(os.path.basename(src))[0]
    if kind == "image":
        dest = os.path.join(folder, stem + ".jpg")
        if not os.path.exists(dest) or stale(dest):
            try:
                from PIL import Image
                im = Image.open(src).convert("RGB")
                im.thumbnail((512, 512))
                im.save(dest, "JPEG", quality=85)
            except ImportError:
                dest = os.path.join(folder, os.path.basename(src))
                shutil.copy2(src, dest)
    elif kind == "audio":  # lossless clips are ~10 MB: 96 kbps MP3 plays everywhere at a tenth of that
        dest = os.path.join(folder, stem + ".mp3")
        if (not os.path.exists(dest) or stale(dest)) and subprocess.run(
                ["ffmpeg", "-v", "error", "-y", "-i", src, "-vn", "-b:a", "96k", dest], capture_output=True).returncode != 0:
            dest = os.path.join(folder, os.path.basename(src))
            shutil.copy2(src, dest)
    else:  # video: re-encoded only when large
        dest = os.path.join(folder, stem + ".mp4")
        if not os.path.exists(dest) or stale(dest):
            if os.path.getsize(src) > 3e6 and subprocess.run(
                    ["ffmpeg", "-v", "error", "-y", "-i", src, "-vf", "scale=-2:480", "-c:v", "libx264", "-crf", "28",
                     "-preset", "veryfast", "-an", "-movflags", "+faststart", dest], capture_output=True).returncode == 0:
                pass
            else:
                shutil.copy2(src, dest)
    return kind, os.path.relpath(dest, out_dir)


def gen_numbers(unit: str) -> list[tuple[str, object]]:
    """The generation tables' number columns: time, GPU energy and peak memory per output, and the engine."""
    per = lambda r: r.get("rate_mean") and 1 / r["rate_mean"]
    return [(f"s / {unit}", lambda r: fmt(per(r), 1, suffix=" s")),
            (f"kJ / {unit}", lambda r: fmt(r.get("power_mean_w") and per(r) and r["power_mean_w"] * per(r) / 1000, 2, suffix=" kJ")),
            ("peak memory", lambda r: fmt(r.get("mem_peak_gib"), 0, suffix=" GiB")),
            ("engine", lambda r: esc(((r.get("what") or {}).get("engine") or "")
                                     + (f" · {(r.get('what') or {}).get('precision')}" if (r.get("what") or {}).get("precision") else "")))]


def gen_cls(name: str) -> str:
    return "eng" if name == "engine" else "num"


def gallery(run_dir: str, rows: list[dict], out_dir: str | None, limit=4) -> tuple[str, list[str]]:
    """What the image, video and audio models made: a few samples each, with the prompt and the speed."""
    gen = [r for r in rows if r["bench"].startswith(("homelab-comfyui-", "homelab-omni-", "homelab-tp2-omni-"))
           and "." not in r["bench"] and use_case(r) not in NOT_GALLERY]  # editing, upscaling, 3D: their own tables
    if not gen or not out_dir:
        return "", []
    files, cards = [], []
    by_prompt = defaultdict(dict)  # prompt -> {bench: image path}: the side-by-side view
    perf = {}  # bench -> its row: the comparison table's speed, energy and memory lines
    for r in sorted(gen, key=lambda r: r["bench"]):
        bench = r["bench"]
        found = _samples(run_dir, r, limit)
        if found and GEN_KINDS[os.path.splitext(found[0])[1].lower()] == "audio":
            found = _samples(run_dir, r, 2)  # 2 clips: they are minutes long
        samples = [_publish_sample(f, out_dir, bench) for f in found]
        prompts_file = os.path.join(run_dir, f"{bench}.outputs", "prompts.json")
        sample_prompts = json.load(open(prompts_file)) if os.path.exists(prompts_file) else {}
        captions = [sample_prompts.get(os.path.basename(f), "") for f in found]
        files += [path for _, path in samples]
        rate = r.get("rate_mean")
        speed = f"{1 / rate:.1f} s per item" if rate else ""
        media = []
        for (kind, path), caption in zip(samples, captions):
            if kind == "image" and caption:
                by_prompt[caption].setdefault(bench, path)
            if kind == "image":
                media.append(f"<a href='{esc(path)}' target=_blank title='{esc(caption)}'><img src='{esc(path)}' "
                             f"alt='{esc(caption or bench + ' sample')}' loading=lazy></a>")
            elif kind == "video":
                media.append(f"<video src='{esc(path)}' controls muted loop playsinline preload=metadata></video>")
            else:
                media.append(f"<audio src='{esc(path)}' controls preload=none></audio>")
        if samples and all(kind == "image" for kind, _ in samples) and any(captions):
            perf[bench] = r
            continue  # an image model: in the comparison table, not a card
        if not media:
            why = ("hung the machine: no output" if r["status"] in ("missing-end",) else
                   "ran before the client kept its outputs: rerun to see them" if "omni" in bench else
                   "no output kept")
            media.append(f"<span class=na>{esc(why)}</span>")
        prompt = _prompt(run_dir, bench) or ("a different prompt per sample: hover one to read it" if any(captions) else None)
        kind = GEN_KINDS[os.path.splitext(found[0])[1].lower()] if found else "video" if "wan" in bench else "other"
        cards.append({"kind": kind, "bench": bench, "row": r, "media": media, "prompt": prompt})
    # the same prompt (and seed) in every image model, side by side: rows = models, columns = their numbers, then
    # one column per prompt (more models than prompts: the table grows down, not across)
    shared = {p: m for p, m in by_prompt.items() if len(m) > 1}
    compare = ""
    if shared:
        try:
            from benchmate.prompts import IMAGE_PROMPTS
        except ImportError:
            IMAGE_PROMPTS = []
        order = [p for p in IMAGE_PROMPTS if p in shared] + sorted(set(shared) - set(IMAGE_PROMPTS))
        models = sorted({b for m in shared.values() for b in m}, key=lambda b: -(perf.get(b, {}).get("rate_mean") or 0))
        numbers = gen_numbers("image")
        short = lambda i, p: f"{i}. " + (" ".join(p.split()[:4]) + ("…" if len(p.split()) > 4 else ""))
        head = ("<tr><th class=model>model</th>" + "".join(f"<th class={gen_cls(n)}>{esc(n)}</th>" for n, _ in numbers)
                + "".join(f"<th class=prompt title='{esc(p)}'>{esc(short(i, p))}</th>" for i, p in enumerate(order, 1)) + "</tr>")
        body = "".join(
            f"<tr><td class=model>{esc(b.replace('homelab-', ''))}</td>"
            + "".join(f"<td class={gen_cls(n)}>{value(perf[b]) if b in perf else fmt(None)}</td>" for n, value in numbers)
            + "".join(f"<td><a href='{esc(shared[p][b])}' target=_blank><img src='{esc(shared[p][b])}' alt='{esc(b)}: {esc(p)}' "
                      f"title='{esc(p)}' loading=lazy></a></td>" if b in shared[p] else "<td class=na>–</td>" for p in order)
            + "</tr>" for b in models)
        prompts = "<ol class=prompts>" + "".join(f"<li>{esc(p)}</li>" for p in order) + "</ol>"
        compare = ("<h3>Image models: the same prompts, side by side</h3>" + prompts
                   + "<div class=wrap><table class=same-prompt>" + head + body + "</table></div>"
                   "<div class=note>One row per model, fastest first; one column per prompt (numbered as listed above; "
                   "hover a header or an image for its prompt). Every image model renders the same prompts in the same "
                   "order with the same seeds (benchmate.prompts), so a column compares the models on one prompt; click "
                   "an image for the full size. Seconds per image: one image at a time, the run's mean. Energy: the GPU's "
                   "mean power over the run times the seconds per image. Peak memory: the machine's unified memory at its "
                   "highest during the run.</div>")
    others = ""
    for kind, title, unit in (("video", "Video models", "clip"), ("audio", "Audio models", "clip"), ("other", "Other generators", "item")):
        entries = sorted((c for c in cards if c["kind"] == kind), key=lambda c: -(c["row"].get("rate_mean") or 0))
        if not entries:
            continue
        numbers = gen_numbers(unit)
        n = max(len(c["media"]) for c in entries)
        head = ("<tr><th class=model>model</th>" + "".join(f"<th class={gen_cls(k)}>{esc(k)}</th>" for k, _ in numbers)
                + "<th class=gprompt>prompt</th>" + "".join(f"<th>{unit} {i}</th>" for i in range(1, n + 1)) + "</tr>")
        body = "".join(
            f"<tr><td class=model>{esc(c['bench'].replace('homelab-', ''))}"
            + ("" if c["row"]["status"] == "ok" else f"<br><span class=bad>{esc(c['row']['status'])}</span>") + "</td>"
            + "".join(f"<td class={gen_cls(k)}>{value(c['row'])}</td>" for k, value in numbers)
            + f"<td class=gprompt><div title='{esc(c['prompt'] or '')}'>{esc(c['prompt'] or '–')}</div></td>"
            + "".join(f"<td class={kind}>{m}</td>" for m in c["media"]) + "<td></td>" * (n - len(c["media"])) + "</tr>"
            for c in entries)
        others += (f"<h3>{title}</h3><div class=wrap><table class='same-prompt gen-{kind}'>{head}{body}</table></div>")
    if others:
        others += ("<div class=note>One row per model, fastest first; a few outputs spread over its run. Each video and "
                   "audio model runs its own workflow's stock prompt (hover a prompt for all of it), so a row is one model, "
                   "not a same-prompt comparison. Seconds and energy per clip as for the images.</div>")
    html_ = "<h2>Generations: what the image, video and audio models made</h2>" + compare + others
    return html_, files


# --- use cases: tool calling, embeddings, long context, vision, OCR, speech, editing, upscaling, 3D, batch, fine-tuning
# (config/homelab/{toolcall,embeddings,longctx,vision,ocr,tts,image-edit,upscale,3d,batch,finetune,finetune-diffusion}.yaml).
# Each section shows only when its benchmarks have results.

# weights that are not free for every use: bench name pattern -> note (from the configs' comments)
LICENSES = [
    (r"qwen-image-21", "Qwen-Image 2.1: Qwen research license (non-commercial)"),
    (r"flux1?-dev(?!.*kontext)", "FLUX.1 dev: FLUX non-commercial license"),
    (r"flux2-dev", "FLUX.2 dev: FLUX non-commercial license"),
    (r"klein-9b", "FLUX.2 klein 9B: FLUX non-commercial license"),
    (r"kontext", "FLUX.1 Kontext dev: FLUX non-commercial license"),
    (r"fibo", "Fibo 2.0: CC BY-NC 4.0 (non-commercial)"),
    (r"minimax-h3", "MiniMax H3-Image: MiniMax community license (written permission needed above 1M USD revenue "
                    "or for hosted services)"),
    (r"breeze-tts", "Breeze-TTS-2: research and non-commercial license"),
    (r"higgs-audio", "Higgs Audio v3: research and non-commercial license"),
    (r"voxtral", "Voxtral 4B TTS: CC BY-NC 4.0 (non-commercial)"),
    (r"omnivoice", "OmniVoice: weights CC BY-NC (non-commercial)"),
    (r"fish-drama|drama-?3", "Fish Audio Drama 3: research license (non-commercial)"),
    (r"ultrasharp", "4x-UltraSharpV2: CC BY-NC-SA 4.0 (non-commercial)"),
    (r"pid-flux|pixeldit", "PiD (PixelDiT): NVIDIA NSCL v1 (non-commercial)"),
    (r"chandra-ocr", "chandra-ocr-2: modified OpenRAIL-M (free for research, personal use and companies under "
                     "2M USD; not to compete with Datalab's API)"),
    (r"nanonets", "Nanonets-OCR2: no license on the model repository (check before commercial use)"),
    (r"hunyuan3d", "Hunyuan3D 2.1: Tencent Hunyuan community license, not licensed in the EU, the UK and South Korea"),
]


def license_note(bench: str) -> str | None:
    name = bench.lower()
    return next((note for pattern, note in LICENSES if re.search(pattern, name)), None)


def lic(bench: str, seen: dict | None = None) -> str:
    """A small "license" marker (the note in its tooltip) after a model name; seen collects the section's notes."""
    note = license_note(bench)
    if not note:
        return ""
    if seen is not None:
        seen.setdefault(note, None)
    return f" <span class=lic title='{esc(note)}'>license</span>"


def license_footer(seen: dict) -> str:
    if not seen:
        return ""
    return ("<div class=note><b>Licenses.</b> Marked <span class=lic>license</span>: weights that are not free for "
            "every use. " + " ".join(esc(n) + "." for n in seen) + "</div>")


def uc_split(bench: str) -> tuple[str, str | None]:
    """bench -> (benchmark, pass); a dot inside a model's name (qwen3-0.6b, minicpm-o-4.5) is not a pass."""
    if "." in bench:
        head, tail = bench.rsplit(".", 1)
        if tail and not tail[0].isdigit():
            return head, tail
    return bench, None


UC_PREFIX = re.compile(r"^homelab-(?:toolcall-|embed-|rerank-|longctx-|vlm-|docparse-|tts-|voice-|batch-|ft-|ftd-|"
                       r"comfyui-edit-|omni-edit-|comfyui-sr-|comfyui-)?")


def uc_name(head: str) -> str:
    return UC_PREFIX.sub("", head)


def use_case(r: dict) -> str | None:
    """Which use-case section a row belongs to (None: the LLM serving / generation sections)."""
    bench, tags = r.get("bench") or "", set(r.get("tags") or [])
    if bench.startswith("homelab-toolcall-") or "toolcall" in tags:
        return "toolcall"
    if "retrieval" in tags or bench.startswith(("homelab-embed-", "homelab-rerank-")):
        return "embed"
    if "longctx" in tags or bench.startswith("homelab-longctx-"):
        return "longctx"
    if bench.startswith("homelab-vlm-"):
        return "vision"
    if "docparse" in tags or bench.startswith("homelab-docparse-"):
        return "ocr"
    if "tts" in tags or bench.startswith(("homelab-tts-", "homelab-voice-")):
        return "tts"
    if "image-edit" in tags or "-edit-" in bench:
        return "edit"
    if "upscale" in tags or bench.startswith(("homelab-comfyui-sr-", "homelab-comfyui-seedvr2-")):
        return "upscale"
    if "3d" in tags:
        return "3d"
    if "batch" in tags or bench.startswith("homelab-batch-"):
        return "batch"
    if bench.startswith(("homelab-ft-", "homelab-ftd-")) or any(k.startswith("ft.") for k in r):
        return "finetune"
    return None


# the use cases whose rows stay out of the LLM serving tables and charts (served models, but not chat serving)
NOT_LLM_SERVING = {"toolcall", "embed", "longctx", "vision", "ocr", "tts", "batch"}
# and out of the text-to-image / video / audio gallery (their own tables below)
NOT_GALLERY = {"edit", "upscale", "3d"}


def uc_groups(rows: list[dict], case: str) -> dict[str, dict]:
    """benchmark -> {"passes": {pass: row}, "rows": [...], "tags": set, "evals": [...]}, for one use case."""
    out = {}
    for r in rows:
        if use_case(r) != case:
            continue
        head, part = uc_split(r["bench"])
        g = out.setdefault(head, {"bench": head, "passes": {}, "rows": [], "tags": set(), "evals": [], "status": set()})
        g["rows"].append(r)
        g["passes"][part or "default"] = r
        g["tags"] |= set(r.get("tags") or [])
        g["status"].add(r["status"])
        g["evals"] += r.get("evals") or []
        for key in ("models", "what", "disk_gb"):
            if r.get(key) and key not in g:
                g[key] = r[key]
    return out


def uc_cell(g: dict, seen: dict | None = None) -> str:
    """The first cell of a use-case table: the model's name, its license marker, its status when not ok."""
    bad = sorted(s for s in g["status"] if s not in ("ok",))
    flag = (f" <span class={'run' if bad == ['running'] else 'bad'} style='font-size:11px'>· {esc(', '.join(bad))}</span>"
            if bad else "")
    return f"<td class=model>{esc(uc_name(g['bench']))}{lic(g['bench'], seen)}{flag}{describe_html(g)}</td>"


def ev_find(g: dict, suite: str | None, task: str, thinking=None) -> dict | None:
    """The last eval summary of a suite / task (and thinking mode) in a benchmark's passes."""
    found = None
    for e in g["evals"]:
        if (suite is None or e.get("suite") == suite) and e.get("task") == task and (thinking is None or e.get("thinking") == thinking):
            found = e
    return found


def per_pass_energy(r: dict | None, items) -> float | None:
    """GPU joules per item over a pass (its energy over its own window)."""
    if not r or not r.get("energy_j") or not items:
        return None
    return r["energy_j"] / items


def pass_items(r: dict | None, rate_key: str) -> float | None:
    """Items a pass handled: its rate times its own duration."""
    if not r or r.get(rate_key) is None or not r.get("pass_duration_s"):
        return None
    return r[rate_key] * r["pass_duration_s"]


def scatter_spec(data: list[dict], x: str, y: str, x_title: str, y_title: str, tooltip: list[dict],
                 group: str | None = None, log_x: bool = False, height: int = 340) -> dict:
    """A quality-vs-speed scatter in the charts' style: one dashed optimal curve (pareto) per group, the curve's
    points labeled; colored by group (or by frontier when there is no group)."""
    groups = defaultdict(list)
    for d in data:
        groups[d.get(group) if group else None].append(d)
    for members in groups.values():
        front = pareto([(d["name"], d[x], d[y]) for d in members])
        for d in members:
            d["frontier"] = d["name"] in front
    xe = {"field": x, "type": "quantitative", "title": x_title, **({"scale": {"type": "log"}} if log_x else {})}
    ye = {"field": y, "type": "quantitative", "title": y_title, "scale": {"zero": False}}
    if group:
        domain = sorted({d[group] for d in data})
        scale = {"domain": domain}
        if len(domain) <= 4:  # the validated categorical colors (more groups: Vega's own scheme)
            scale["range"] = ["SERIES0", "SERIES1", "SERIES2", "SERIES3"][:len(domain)]
        color = {"field": group, "type": "nominal", "scale": scale, "legend": {"title": None, "orient": "top", "labelLimit": 400}}
    else:
        color = {"field": "frontier", "type": "nominal", "scale": {"domain": [True, False], "range": ["SERIES0", "MUTED"]},
                 "legend": {"title": None, "labelExpr": "datum.value ? 'best trade-off (nothing faster and better)' : 'other'",
                            "orient": "bottom", "labelLimit": 400}}
    line_color = {"field": group, "type": "nominal", "scale": color["scale"], "legend": color["legend"]} if group else {"value": "SERIES0"}
    return {"data": {"values": data}, "height": height, "layer": [
        {"transform": [{"filter": "datum.frontier"}],
         "mark": {"type": "line", "strokeDash": [6, 4], "strokeWidth": 2},
         "encoding": {"x": xe, "y": ye, "order": {"field": x, "type": "quantitative"}, "color": line_color,
                      **({"detail": {"field": group}} if group else {}), "opacity": {"value": 0.55}}},
        {"mark": {"type": "point", "filled": True, "size": 80, "stroke": "SURFACE", "strokeWidth": 2},
         "encoding": {"x": xe, "y": ye, "color": color, "tooltip": tooltip}},
        # a frontier point whose coding score is still LiveCodeBench's quick 18 says so (the full 175 is due)
        {"transform": [{"filter": "datum.frontier"},
                       {"calculate": "datum.name + (datum.tier && indexof(datum.tier, 'quick') >= 0 ? ' · LCB 18 only' : '')",
                        "as": "label"}],
         "mark": {"type": "text", "align": "left", "dx": 7, "dy": -4, "fontSize": 10.5},
         "encoding": {"x": xe, "y": ye, "text": {"field": "label"}, "color": {"value": "TEXT"}}}]}


def line_spec(data: list[dict], x: str, y: str, x_title: str, y_title: str, tooltip: list[dict], log_y=False,
              x_values=None, height=320) -> dict:
    """One line per model over the context lengths."""
    xe = {"field": x, "type": "quantitative", "title": x_title, "scale": {"type": "log", "base": 2},
          "axis": {"values": x_values} if x_values else {}}
    ye = {"field": y, "type": "quantitative", "title": y_title, **({"scale": {"type": "log"}} if log_y else {})}
    return {"data": {"values": data}, "height": height, "layer": [
        {"mark": {"type": "line", "strokeWidth": 1.5, "point": {"filled": True, "size": 45}},
         "encoding": {"x": xe, "y": ye, "color": {"field": "name", "type": "nominal",
                                                   "legend": {"title": None, "orient": "right", "labelLimit": 260}},
                      "order": {"field": x, "type": "quantitative"}, "tooltip": tooltip}}]}


UC_JS = r"""<script>(function(){const specs=__SPECS__;
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
function theme(){const dark=document.documentElement.dataset.theme==='dark'||(document.documentElement.dataset.theme!=='light'&&matchMedia('(prefers-color-scheme: dark)').matches);
 return {SERIES0:dark?'#3987e5':'#2a78d6',SERIES1:dark?'#d95926':'#eb6834',SERIES2:dark?'#199e70':'#1baf7a',SERIES3:dark?'#c98500':'#eda100',
  TEXT:css('--fg'),MUTED:css('--muted'),SURFACE:css('--surface'),LINE:css('--line')};}
function fill(o,t){if(typeof o==='string')return t[o]??o;if(Array.isArray(o))return o.map(x=>fill(x,t));if(o&&typeof o==='object'){const r={};for(const k in o)r[k]=fill(o[k],t);return r;}return o;}
let views=[];
function render(){if(!window.vegaEmbed)return;const t=theme();for(const v of views)v.finalize();views=[];
 for(const id in specs){const el=document.getElementById(id);if(!el)continue;
  const spec=Object.assign({$schema:'https://vega.github.io/schema/vega-lite/v5.json',width:'container',autosize:{type:'fit-x',contains:'padding'}},fill(specs[id],t));
  spec.config={background:'transparent',font:'Inter, system-ui, sans-serif',view:{stroke:null},
   axis:{labelColor:t.MUTED,titleColor:t.MUTED,gridColor:t.LINE,domainColor:t.LINE,tickColor:t.LINE,labelFontSize:10.5,titleFontSize:11,titleFontWeight:600},
   axisX:{grid:false},legend:{labelColor:t.TEXT,titleColor:t.MUTED,titleFontSize:11,titleFontWeight:600,labelFontSize:11,symbolStrokeWidth:0},text:{font:'JetBrains Mono, ui-monospace, monospace'}};
  vegaEmbed(el,spec,{actions:false,renderer:'svg'}).then(r=>views.push(r.view)).catch(e=>{el.textContent='chart failed to render: '+e.message;});}}
render();matchMedia('(prefers-color-scheme: dark)').addEventListener('change',render);
new MutationObserver(render).observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});
})();</script>"""


def figure(sid: str, caption: str) -> str:
    return f"<figure><figcaption>{esc(caption)}</figcaption><div class=vega id={sid}></div></figure>"


def pct(v, digits=1) -> str:
    return fmt(v, digits, 100)


# --- tool calling ------------------------------------------------------------------------------------

BFCL_GROUPS = [("overall", "overall"), ("non_live", "single-turn"), ("live", "live (user-written)"),
               ("multi_turn", "multi-turn"), ("irrelevance", "no call needed")]


def toolcall_section(rows: list[dict], specs: dict) -> str:
    groups = uc_groups(rows, "toolcall")

    def qb(g, mode):  # QualityBench's bfcl task (400 items, single turn)
        for r in g["rows"]:
            if r.get(f"acc.qualitybench.bfcl.think-{mode}") is not None:
                return r[f"acc.qualitybench.bfcl.think-{mode}"], r.get(f"acc.qualitybench.bfcl.think-{mode}.ci95")
        return None, None

    shown = [g for g in groups.values() if qb(g, "off")[0] is not None or qb(g, "on")[0] is not None
             or any(e.get("suite") == "bfcl" for e in g["evals"])]
    if not shown:
        return ""
    shown.sort(key=lambda g: -max(qb(g, "off")[0] or 0, qb(g, "on")[0] or 0))
    cats = []
    for g in shown:
        for e in g["evals"]:
            t = e.get("task", "")
            if e.get("suite") == "bfcl" and t.startswith("bfcl:") and t != "bfcl:overall" and t[5:] not in cats:
                cats.append(t[5:])
    extended = any(ev_find(g, "bfcl", "bfcl:overall") for g in shown)
    seen = {}
    ci = lambda c: f" title='95% interval {c[0] * 100:.0f}–{c[1] * 100:.0f}'" if c and len(c) == 2 else ""
    h = ["<h2>Tool calling</h2><div class=wrap><table class=uc><tr><th>model</th><th class=grp>BFCL, thinking off</th>"
         "<th>BFCL, thinking on</th>"
         + ("".join(f"<th class={'grp' if i == 0 else ''}>{esc(label)}</th>" for i, (_, label) in enumerate(BFCL_GROUPS)) if extended else "")
         + "".join(f"<th{' class=grp' if i == 0 else ''}>{esc(c.replace('_', ' '))}</th>" for i, c in enumerate(cats))
         + "<th class=grp>s per call p50</th><th>s per call p90</th><th>s per multi-turn turn p50</th></tr>"]
    bars = []
    for g in shown:
        (off, off_ci), (on, on_ci) = qb(g, "off"), qb(g, "on")
        name = uc_name(g["bench"])
        for mode, v in (("thinking off", off), ("thinking on", on)):
            if v is not None:
                bars.append({"name": name, "thinking": mode, "BFCL %": round(100 * v, 1)})
        overall = ev_find(g, "bfcl", "bfcl:overall")
        groups_ = (overall or {}).get("scores") or {}
        # latency: the single-stream pass when there is one (bfcl-lat, threads 1), else the extended pass
        single = (g["passes"].get("bfcl-lat") or {}).get("evals") or []
        lat = next((e["latency"] for e in single + g["evals"] if e.get("suite") == "bfcl" and (e.get("latency") or {}).get("p50_s")), None) or {}
        turn = next((e["latency"].get("turn_p50_s") for e in g["evals"] if (e.get("latency") or {}).get("turn_p50_s")), None)
        h.append("<tr>" + uc_cell(g, seen) + f"<td class=grp{ci(off_ci)}>{pct(off)}</td><td{ci(on_ci)}>{pct(on)}</td>"
                 + ("".join(f"<td class={'grp' if i == 0 else ''}>{pct(groups_.get(k))}</td>" for i, (k, _) in enumerate(BFCL_GROUPS)) if extended else "")
                 + "".join(f"<td{' class=grp' if i == 0 else ''}>{pct(((ev_find(g, 'bfcl', 'bfcl:' + c, 'off') or ev_find(g, 'bfcl', 'bfcl:' + c) or {}).get('scores') or {}).get('acc'))}</td>"
                           for i, c in enumerate(cats))
                 + f"<td class=grp>{fmt(lat.get('p50_s'), 2)}</td><td>{fmt(lat.get('p90_s'), 2)}</td><td>{fmt(turn, 2)}</td></tr>")
    specs["uc-bfcl"] = {"data": {"values": bars}, "height": 300,
                        "mark": {"type": "bar", "cornerRadiusTopLeft": 3, "cornerRadiusTopRight": 3},
                        "encoding": {"x": {"field": "name", "type": "nominal", "sort": [uc_name(g["bench"]) for g in shown], "title": None,
                                           "axis": {"labelAngle": -50, "labelLimit": 240}},
                                     "xOffset": {"field": "thinking"},
                                     "y": {"field": "BFCL %", "type": "quantitative", "title": "BFCL accuracy (%)", "scale": {"domain": [0, 100]}},
                                     "color": {"field": "thinking", "type": "nominal", "scale": {"domain": ["thinking off", "thinking on"], "range": ["SERIES0", "SERIES1"]},
                                               "legend": {"title": None, "orient": "top"}},
                                     "tooltip": [{"field": "name", "title": "model"}, {"field": "thinking"}, {"field": "BFCL %"}]}}
    h.append("</table></div>")
    if bars:
        h.append(figure("uc-bfcl", "BFCL accuracy, thinking off and on"))
    h.append("<div class=note>BFCL: the Berkeley Function Calling Leaderboard's single-turn categories (QualityBench's 400 "
             "items: simple, multiple, parallel and parallel-multiple calls), scored on each model's agent server: the "
             "server's tool-call parser turns the reply into calls, so a call the parser misreads counts as wrong. Hover a "
             "score for its 95% interval. "
             + ("The group and category columns come from the extended pass (every second item of 9 categories): "
                "<i>no call needed</i> rewards not calling a tool when none fits (made-up calls score wrong); <i>live</i> "
                "are user-written requests; <i>multi-turn</i> runs simulated tool APIs over several turns. " if extended or cats else "")
             + "Seconds per call: BFCL's client-side time per request, from the single-stream pass when there is one "
               "(else with 16 requests in flight); a multi-turn turn sums every call of the turn.</div>")
    h.append(license_footer(seen))
    return "\n".join(h)


# --- embeddings and reranking ------------------------------------------------------------------------

EMBED_PASSES = ["query-c1", "query-c32", "doc512-b32", "doc512-b256", "doc2k-b32", "doc8k-b4", "rr-d16-c1", "rr-d100-c1", "rr-d100-c4"]


def embed_section(rows: list[dict], specs: dict) -> str:
    groups = uc_groups(rows, "embed")
    if not any(g["passes"].get(p, {}).get("docs_per_s") for g in groups.values() for p in g["passes"]) \
            and not any(e.get("suite") == "retrieval" for g in groups.values() for e in g["evals"]):
        return ""
    seen, h, points = {}, ["<h2>Embeddings and reranking</h2>"], []
    for kind, title in (("embed", "Embedding models"), ("rerank", "Rerankers")):
        members = [g for g in groups.values() if (kind in g["tags"]) or (kind == "embed" and "rerank" not in g["tags"]
                                                                          and g["bench"].startswith("homelab-embed-"))]
        if not members:
            continue
        passes = [p for p in EMBED_PASSES if any((g["passes"].get(p) or {}).get("docs_per_s") is not None for g in members)]
        for g in members:
            for p, r in g["passes"].items():
                if r.get("docs_per_s") is not None and p not in passes:
                    passes.append(p)
        datasets = []
        for g in members:
            for e in g["evals"]:
                if e.get("suite") == "retrieval" and e.get("task") not in datasets:
                    datasets.append(e["task"])

        def best(g):
            rs = [r for r in g["passes"].values() if r.get("docs_per_s")]
            return max(rs, key=lambda r: r["docs_per_s"]) if rs else None

        def j1k(r):
            docs = pass_items(r, "docs_per_s")
            e = per_pass_energy(r, docs)
            return e * 1000 if e is not None else None

        members.sort(key=lambda g: -((best(g) or {}).get("docs_per_s") or 0))
        h.append(f"<h3>{title}</h3><div class=wrap><table class=uc><tr><th>model</th>"
                 + "".join(f"<th{' class=grp' if i == 0 else ''} title='documents per second'>{esc(p)} docs/s</th>" for i, p in enumerate(passes))
                 + "<th class=grp>best J per 1k docs</th><th>peak memory GiB</th>"
                 + "".join(f"<th class=grp>{esc(d)} nDCG@10</th><th>{esc(d)} recall@100</th>"
                           + (f"<th>{esc(d)} BM25 nDCG@10</th>" if kind == "rerank" else "") for d in datasets)
                 + ("<th class=grp>indexing docs/s</th>" if kind == "embed" else "<th class=grp>pairs/s</th>") + "</tr>")
        for g in members:
            evs = {d: ev_find(g, "retrieval", d) or {} for d in datasets}
            effs = [j1k(r) for r in g["passes"].values() if j1k(r) is not None]
            mem = max((r.get("mem_peak_gib") or 0 for r in g["rows"]), default=0) or None
            speed_cell = next((e.get("index_docs_per_s") if kind == "embed" else e.get("rerank_pairs_per_s")
                               for e in evs.values() if e), None)
            h.append("<tr>" + uc_cell(g, seen)
                     + "".join(f"<td{' class=grp' if i == 0 else ''}>{fmt((g['passes'].get(p) or {}).get('docs_per_s'), 0)}</td>" for i, p in enumerate(passes))
                     + f"<td class=grp>{fmt(min(effs) if effs else None, 1)}</td><td>{fmt(mem)}</td>"
                     + "".join(f"<td class=grp>{pct((e.get('scores') or {}).get('ndcg@10'))}</td><td>{pct((e.get('scores') or {}).get('recall@100'))}</td>"
                               + (f"<td>{pct((e.get('bm25') or {}).get('ndcg@10'))}</td>" if kind == "rerank" else "")
                               for e in evs.values())
                     + f"<td class=grp>{fmt(speed_cell, 0)}</td></tr>")
            nd = [(e.get("scores") or {}).get("ndcg@10") for e in evs.values() if (e.get("scores") or {}).get("ndcg@10") is not None]
            b = best(g)
            if nd and b:
                points.append({"name": uc_name(g["bench"]), "kind": "embedding" if kind == "embed" else "reranker",
                               "docs/s": round(b["docs_per_s"], 1), "nDCG@10": round(100 * sum(nd) / len(nd), 1),
                               "pass": uc_split(b["bench"])[1], "J/1k docs": round(min(effs), 1) if effs else None})
        h.append("</table></div>")
    if points:
        specs["uc-embed"] = scatter_spec(points, "docs/s", "nDCG@10", "documents per second (best pass, log scale)",
                                         "retrieval quality: mean nDCG@10 (%)",
                                         [{"field": "name", "title": "model"}, {"field": "kind"}, {"field": "nDCG@10"},
                                          {"field": "docs/s"}, {"field": "pass", "title": "fastest pass"},
                                          {"field": "J/1k docs", "title": "best J per 1k docs"}], group="kind", log_x=True)
        h.append(figure("uc-embed", "Retrieval quality vs documents per second"))
    h.append("<div class=note>Speed passes: real text cut to a fixed length, sent in batches (query-c1: one 32-token "
             "query at a time, the latency a search box sees; doc512-b256: 256 documents of 512 tokens per request, "
             "indexing a corpus). Rerankers: one query with 16 or 100 candidate documents per request. J per 1k docs: GPU "
             "energy over the pass, at the model's most efficient pass. Quality: BEIR test sets as MTEB ships them, whole "
             "corpus; nDCG@10 (how well the first 10 results are ordered) and recall@100 (share of the relevant documents "
             "in the first 100). Rerankers reorder the same 50 BM25 candidates: the BM25 column is the starting point. "
             "Scatter: up and to the right is better; the dashed line per kind is the optimal curve (nothing faster and "
             "better).</div>")
    h.append(license_footer(seen))
    return "\n".join(h)


# --- long context ------------------------------------------------------------------------------------

CTX_PASSES = [("ctx8k", 8), ("ctx32k", 32), ("ctx64k", 64), ("ctx128k", 128), ("ctx256k", 256)]
RULER_LENGTHS = ["8192", "32768", "65536", "131072"]


def kv_capacity(run_dir: str) -> dict:
    """benchmarks/longctx/kv_capacity.py --json output, when saved with the run (report/ or the run folder)."""
    for path in (os.path.join(run_dir, "report", "kv_capacity.json"), os.path.join(run_dir, "kv_capacity.json")):
        try:
            return {r["bench"]: r for r in json.load(open(path))}
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return {}


def longctx_section(rows: list[dict], specs: dict, run_dir: str) -> str:
    groups = uc_groups(rows, "longctx")
    shown = [g for g in groups.values() if any(p in g["passes"] for p, _ in CTX_PASSES)
             or any(r.get("acc.qualitybench.needle.think-off") is not None for r in g["rows"]) or g["evals"]]
    if not shown:
        return ""
    kv = kv_capacity(run_dir)
    ttft, prefill, decode = [], [], []
    ctxs = [(p, k) for p, k in CTX_PASSES if any(p in g["passes"] for g in shown)]

    def ruler(g, length):
        vals = {}
        for e in g["evals"]:
            t = str(e.get("task", ""))
            if t.startswith("longctx_ruler"):
                v = next((v for k, v in (e.get("scores") or {}).items() if k.split(",")[0] == length), None)
                if v is not None and v >= 0:
                    vals[t.removeprefix("longctx_ruler_")] = v
        return vals

    lengths = [n for n in RULER_LENGTHS if any(ruler(g, n) for g in shown)]
    kv_lengths = ["32768", "131072", "262144"] if kv else []
    h = ["<h2>Long context</h2><div class=wrap><table class=uc><tr><th>model</th>"
         + "".join(f"<th class=grp>{k}k TTFT s</th><th>{k}k decode tok/s</th>" for _, k in ctxs)
         + "<th class=grp>4 × 32k tok/s</th><th class=grp>needle %</th>"
         + "".join(f"<th{' class=grp' if i == 0 else ''}>RULER {int(n) // 1024}k %</th>" for i, n in enumerate(lengths))
         + "".join(f"<th{' class=grp' if i == 0 else ''}>sessions at {int(n) // 1024}k</th>" for i, n in enumerate(kv_lengths)) + "</tr>"]
    for g in shown:
        name = uc_name(g["bench"])
        cells = ""
        for p, k in ctxs:
            r = g["passes"].get(p) or {}
            t = r.get("ttft_ms_p50")
            cells += f"<td class=grp>{fmt(t and t / 1000, 1)}</td><td>{fmt(per_user(r), 1)}</td>"
            tip = [{"field": "name", "title": "model"}, {"field": "k tokens", "title": "context (k tokens)"}]
            if t:
                ttft.append({"name": name, "k tokens": k, "TTFT s": round(t / 1000, 2)})
                if r.get("mean_input_len"):
                    prefill.append({"name": name, "k tokens": k, "prefill tok/s": round(r["mean_input_len"] / (t / 1000))})
            if per_user(r):
                decode.append({"name": name, "k tokens": k, "decode tok/s": round(per_user(r), 1)})
        needle = next((r.get("acc.qualitybench.needle.think-off") for r in g["rows"] if r.get("acc.qualitybench.needle.think-off") is not None), None)
        by_len = next((ev.get("scores") or {} for r in g["rows"] for ev in r.get("evals") or []
                       if ev.get("task") == "needle_by_length"), {})
        needle_tip = ", ".join(f"{k.removeprefix('acc_')} {100 * v:.0f}%" for k, v in sorted(by_len.items(), key=lambda kv: int(kv[0].removeprefix('acc_').removesuffix('k') or 0)) if isinstance(v, (int, float)))
        conc4 = (g["passes"].get("conc4-32k") or {}).get("output_throughput")
        rul = ""
        for i, n in enumerate(lengths):
            vals = ruler(g, n)
            mean = sum(vals.values()) / len(vals) if vals else None
            tip = ", ".join(f"{k.replace('_', ' ')} {100 * v:.0f}" for k, v in vals.items())
            rul += f"<td{' class=grp' if i == 0 else ''} title='{esc(tip)}'>{pct(mean, 0)}</td>"
        cap = kv.get(g["bench"]) or {}
        sess = "".join(f"<td{' class=grp' if i == 0 else ''}>{fmt(((cap.get('sessions') or {}).get(n) or {}).get('fit'), 0)}</td>"
                       for i, n in enumerate(kv_lengths))
        h.append("<tr>" + uc_cell(g) + cells + f"<td class=grp>{fmt(conc4, 0)}</td><td class=grp title='{esc(needle_tip)}'>{pct(needle, 0)}</td>" + rul + sess + "</tr>")
    h.append("</table></div>")
    ticks = [k for _, k in ctxs]
    tip = lambda f: [{"field": "name", "title": "model"}, {"field": "k tokens", "title": "context (k tokens)"}, {"field": f}]
    if ttft:
        specs["uc-ctx-ttft"] = line_spec(ttft, "k tokens", "TTFT s", "prompt length (k tokens)", "time to first token (s, log scale)",
                                         tip("TTFT s"), log_y=True, x_values=ticks)
        h.append(figure("uc-ctx-ttft", "Time to first token by prompt length"))
    if prefill:
        specs["uc-ctx-prefill"] = line_spec(prefill, "k tokens", "prefill tok/s", "prompt length (k tokens)", "prefill tokens per second",
                                            tip("prefill tok/s"), x_values=ticks)
        h.append(figure("uc-ctx-prefill", "Prefill speed by prompt length"))
    if decode:
        specs["uc-ctx-decode"] = line_spec(decode, "k tokens", "decode tok/s", "prompt length (k tokens)", "output tokens per second, one user",
                                           tip("decode tok/s"), x_values=ticks)
        h.append(figure("uc-ctx-decode", "Decode speed at depth"))
    h.append("<div class=note>One user, a real-text prompt of the given length, 256 tokens out, prefix caching off: the "
             "time to first token is reading the whole document; decode tok/s is the writing speed once that deep "
             "(1000 / TPOT p50). Prefill tok/s: prompt tokens over the time to first token. 4 × 32k: total output tokens "
             "per second with four 32k-token sessions at once. Needle: QualityBench's needle test (a 7-digit number at 10 "
             "depths of 32k and 128k haystacks), thinking off. RULER: a fixed slice (multi-key and multi-value needles, "
             "variable tracking, HotpotQA among distractors), 8 samples per task and length, mean of the tasks (hover for "
             "each). " + ("Sessions: how many sessions of that length fit in the memory left beside the weights "
                          "(benchmarks/longctx/kv_capacity.py, an estimate from the model's config)." if kv else "") + "</div>")
    return "\n".join(h)


# --- vision Q&A --------------------------------------------------------------------------------------

VQA_TASKS = [("docvqa", "DocVQA (ANLS)"), ("chartqa", "ChartQA"), ("ocrbench", "OCRBench"), ("mmmu", "MMMU")]


def vision_section(rows: list[dict], specs: dict) -> str:
    groups = uc_groups(rows, "vision")
    shown = [g for g in groups.values() if any(e.get("suite") == "vqa" for e in g["evals"])
             or any(r.get("output_throughput") for r in g["passes"].values())]
    if not shown:
        return ""
    mean_acc = lambda g: (lambda v: sum(v) / len(v) if v else None)(
        [s for t, _ in VQA_TASKS if (s := ((ev_find(g, "vqa", t) or {}).get("scores") or {}).get("acc")) is not None])
    shown.sort(key=lambda g: -(mean_acc(g) or 0))
    seen, points = {}, []
    h = ["<h2>Vision: questions about images</h2><div class=wrap><table class=uc><tr><th>model</th>"
         + "".join(f"<th{' class=grp' if i == 0 else ''}>{esc(label)}</th>" for i, (_, label) in enumerate(VQA_TASKS))
         + "<th>mean</th><th class=grp>s per image, 1 user</th><th>TTFT s, 1 user</th><th>images/s, 8 users</th>"
           "<th>s per image, Q&amp;A pass</th><th>image + prompt tokens</th><th>describe tok/s, 1 user</th></tr>"]
    for g in shown:
        evs = {t: ev_find(g, "vqa", t) or {} for t, _ in VQA_TASKS}
        timings = [e.get("timing") or {} for e in evs.values() if e]
        e2e = [t["e2el_ms_p50"] for t in timings if t.get("e2el_ms_p50")]
        tokens = [t["prompt_tokens_mean"] for t in timings if t.get("prompt_tokens_mean")]
        one = g["passes"].get("img1") or {}
        m = mean_acc(g)
        h.append("<tr>" + uc_cell(g, seen)
                 + "".join(f"<td{' class=grp' if i == 0 else ''}>{pct((e.get('scores') or {}).get('acc'))}</td>" for i, e in enumerate(evs.values()))
                 + f"<td><b>{pct(m)}</b></td><td class=grp>{fmt(one.get('e2el_ms_p50') and one['e2el_ms_p50'] / 1000, 2)}</td>"
                 f"<td>{fmt(one.get('ttft_ms_p50') and one['ttft_ms_p50'] / 1000, 2)}</td>"
                 f"<td>{fmt((g['passes'].get('img8') or {}).get('request_throughput'), 2)}</td>"
                 f"<td>{fmt(sum(e2e) / len(e2e) / 1000 if e2e else None, 2)}</td><td>{fmt(sum(tokens) / len(tokens) if tokens else None, 0)}</td>"
                 f"<td>{fmt(per_user(g['passes'].get('desc1')), 1)}</td></tr>")
        if m is not None and one.get("e2el_ms_p50"):
            points.append({"name": uc_name(g["bench"]), "accuracy %": round(100 * m, 1),
                           "images/min": round(60000 / one["e2el_ms_p50"], 1),
                           **{label: round(100 * ((evs[t].get("scores") or {}).get("acc") or 0), 1) for t, label in VQA_TASKS if evs[t]}})
    h.append("</table></div>")
    if points:
        specs["uc-vision"] = scatter_spec(points, "images/min", "accuracy %", "images per minute, one user (a document page + question)",
                                          "mean accuracy (%)", [{"field": "name", "title": "model"}, {"field": "accuracy %"},
                                                                {"field": "images/min"}] + [{"field": l} for _, l in VQA_TASKS])
        h.append(figure("uc-vision", "Accuracy vs speed, one user"))
    h.append("<div class=note>Accuracy: fixed slices (about 100 items each) of DocVQA (document pages, ANLS score), ChartQA "
             "(charts, numbers within 5%), OCRBench (reading text in images) and MMMU (college-level questions with an "
             "image), greedy, thinking off where the model has the switch; mean: the four tasks' plain mean. s per image, "
             "1 user: a document page and its question, 64 tokens out (p50). Q&amp;A pass: the accuracy run's time per "
             "image with 8 in flight. Image + prompt tokens: what the server counted per request (the image's size in "
             "tokens drives the prefill cost). Describe: decode speed writing a 256-token description.</div>")
    h.append(license_footer(seen))
    return "\n".join(h)


# --- OCR / document parsing ---------------------------------------------------------------------------

OCR_TASKS = [("olmocr", "olmOCR-bench"), ("omnidocbench", "OmniDocBench"), ("opb_fr", "French pages"), ("opb_hw", "handwriting")]
OCR_FAMILY = [("ocr-pipeline", "OCR pipeline (no LLM)"), ("vlm-general", "general VLM"), ("ocr-dedicated", "dedicated OCR model")]


def ocr_section(rows: list[dict], specs: dict) -> str:
    groups = uc_groups(rows, "ocr")
    shown = [g for g in groups.values() if any(r.get("pages_per_min") for r in g["passes"].values())
             or any(e.get("suite") == "docparse" for e in g["evals"])]
    if not shown:
        return ""
    family = lambda g: next((label for tag, label in OCR_FAMILY if tag in g["tags"]), "other")
    score = lambda g, t: ((ev_find(g, "docparse", t) or {}).get("scores") or {}).get("acc")
    shown.sort(key=lambda g: -(score(g, "olmocr") or 0))
    seen, points = {}, []
    h = ["<h2>OCR and document parsing</h2><div class=wrap><table class=uc><tr><th>model</th><th>family</th>"
         "<th class=grp>pages/min, 1 at a time</th><th>pages/min, 8 at once</th><th>s per page p50, 1 at a time</th>"
         "<th>J per page, 8 at once</th><th>peak memory GiB</th>"
         + "".join(f"<th{' class=grp' if i == 0 else ''}>{esc(label)} %</th>" for i, (_, label) in enumerate(OCR_TASKS)) + "</tr>"]
    for g in shown:
        c1, c8 = g["passes"].get("c1") or {}, g["passes"].get("c8") or {}
        jpp = per_pass_energy(c8, c8.get("completed"))
        mem = max((r.get("mem_peak_gib") or 0 for r in g["rows"]), default=0) or None
        h.append("<tr>" + uc_cell(g, seen) + f"<td class=left>{esc(family(g))}</td>"
                 f"<td class=grp>{fmt(c1.get('pages_per_min'), 1)}</td><td>{fmt(c8.get('pages_per_min'), 1)}</td>"
                 f"<td>{fmt(c1.get('e2el_ms_p50') and c1['e2el_ms_p50'] / 1000, 1)}</td><td>{fmt(jpp, 0)}</td><td>{fmt(mem)}</td>"
                 + "".join(f"<td{' class=grp' if i == 0 else ''}>{pct(score(g, t))}</td>" for i, (t, _) in enumerate(OCR_TASKS)) + "</tr>")
        if score(g, "olmocr") is not None and c8.get("pages_per_min"):
            points.append({"name": uc_name(g["bench"]), "family": family(g), "olmOCR %": round(100 * score(g, "olmocr"), 1),
                           "pages/min": round(c8["pages_per_min"], 1), "J/page": round(jpp) if jpp else None,
                           **{label: round(100 * score(g, t), 1) for t, label in OCR_TASKS[1:] if score(g, t) is not None}})
    h.append("</table></div>")
    if points:
        # one optimal curve over every family (the trade-off a user picks from), colored by family
        front = pareto([(p["name"], p["pages/min"], p["olmOCR %"]) for p in points])
        spec = scatter_spec(points, "pages/min", "olmOCR %", "pages per minute, 8 pages in flight (log scale)",
                            "olmOCR-bench pass rate (%)",
                            [{"field": "name", "title": "model"}, {"field": "family"}, {"field": "olmOCR %"},
                             {"field": "pages/min"}, {"field": "J/page", "title": "GPU joules per page"}]
                            + [{"field": label} for _, label in OCR_TASKS[1:]], group="family", log_x=True)
        for p in points:
            p["frontier"] = p["name"] in front
        line = spec["layer"][0]["encoding"]
        line["color"] = {"value": "SERIES0"}
        line.pop("detail", None)
        specs["uc-ocr"] = spec
        h.append(figure("uc-ocr", "OCR quality vs pages per minute"))
    h.append("<div class=note>Speed: a fixed set of 56 pages (old scans, tables, math, multi-column, French forms), rendered "
             "before the clock starts; 1 at a time: the first 4 pages of each group. J per page: GPU energy over the "
             "8-in-flight pass. Quality, quick tier, each with its official scorer: olmOCR-bench (share of its unit "
             "tests passed), OmniDocBench (text, tables and formulas combined), and two omni_parse_bench slices, French "
             "and handwriting. Each model runs its own recipe (prompt, resolution, output format). Scatter: up and to the "
             "right is better; the dashed line is the optimal curve across every family.</div>")
    h.append(license_footer(seen))
    return "\n".join(h)


# --- text to speech ----------------------------------------------------------------------------------

TTS_CLIPS = (("single", "a short reply"), ("paragraph", "a paragraph"))  # passes whose first clip is played


def tts_clips(run_dir: str, out_dir: str | None, head: str) -> tuple[list[dict], list[str]]:
    """The model's first clip of the short-reply and paragraph passes (<bench>.<pass>.outputs/tts_001.wav and
    texts.json, the text it spoke), published like the gallery's audio."""
    clips, files = [], []
    if not out_dir:
        return clips, files
    for part, what in TTS_CLIPS:
        folder = os.path.join(run_dir, f"{head}.{part}.outputs")
        wavs = sorted(f for f in os.listdir(folder) if f.endswith(".wav")) if os.path.isdir(folder) else []
        if not wavs:
            continue
        try:
            texts = json.load(open(os.path.join(folder, "texts.json")))
        except (OSError, ValueError):
            texts = {}
        _, path = _publish_sample(os.path.join(folder, wavs[0]), out_dir, f"{head}.{part}")
        files.append(path)
        clips.append({"what": what, "path": path, "text": texts.get(wavs[0]) if isinstance(texts, dict) else None})
    return clips, files


def tts_section(rows: list[dict], run_dir: str = "", out_dir: str | None = None) -> tuple[str, list[str]]:
    groups = uc_groups(rows, "tts")
    shown = [g for g in groups.values() if any(r.get("tts.rtf_x") or r.get("tts.wer") is not None for r in g["rows"])]
    if not shown:
        return "", []
    get = lambda g, p, k: (g["passes"].get(p) or {}).get(k)
    shown.sort(key=lambda g: -(get(g, "single", "tts.rtf_x") or 0))
    seen = {}
    h = ["<h2>Text to speech</h2><div class=wrap><table class=uc><tr><th>model</th>"
         "<th class=grp>real-time factor, 1 user</th><th>first audio s, 1 user</th><th>first audio s p90</th>"
         "<th>real-time factor, paragraph</th><th class=grp>× real time, 4 users</th><th>× real time, 16 users</th>"
         "<th>first audio s, 16 users</th><th class=grp>word error rate %</th><th>J per audio minute</th><th>peak memory GiB</th></tr>"]
    for g in shown:
        wer = next((r.get("tts.wer") for r in g["rows"] if r.get("tts.wer") is not None), None)
        single = g["passes"].get("single") or {}
        jpm = (single.get("energy_j") / single["tts.audio_seconds"] * 60) if single.get("energy_j") and single.get("tts.audio_seconds") else None
        mem = max((r.get("mem_peak_gib") or 0 for r in g["rows"]), default=0) or None
        xr = lambda p: get(g, p, "tts.x_realtime") or get(g, p, "rate_mean")
        h.append("<tr>" + uc_cell(g, seen)
                 + f"<td class=grp>{fmt(get(g, 'single', 'tts.rtf_x'), 1, suffix='×')}</td><td>{fmt(get(g, 'single', 'tts.ttfa_s_p50'), 2)}</td>"
                 f"<td>{fmt(get(g, 'single', 'tts.ttfa_s_p90'), 2)}</td><td>{fmt(get(g, 'paragraph', 'tts.rtf_x'), 1, suffix='×')}</td>"
                 f"<td class=grp>{fmt(xr('conc4'), 1, suffix='×')}</td><td>{fmt(xr('conc16'), 1, suffix='×')}</td>"
                 f"<td>{fmt(get(g, 'conc16', 'tts.ttfa_s_p50'), 2)}</td><td class=grp>{pct(wer)}</td><td>{fmt(jpm, 0)}</td><td>{fmt(mem)}</td></tr>")
    h.append("</table></div><div class=note>Real-time factor: seconds of audio made per second of work, per request (above "
             "1×: faster than it plays). First audio: time until the first audio chunk arrives (streaming), what a voice "
             "assistant waits before it starts talking. × real time with 4 or 16 users: seconds of audio per wall second, "
             "all requests together. Word error rate: every text spoken once, transcribed back with Whisper and compared "
             "with the input (lower is better; voice-assistant models: against their own written answer). J per audio "
             "minute: GPU energy of the 1-user pass per minute of audio.</div>")
    # the voices: one short reply and one paragraph per model, the text it was given under each
    files, voices = [], []
    for g in shown:
        clips, more = tts_clips(run_dir, out_dir, g["bench"])
        files += more
        if clips:
            cells = "".join(
                f"<td class=left><audio src='{esc(c['path'])}' controls preload=none></audio>"
                f"<div class=desc>“{esc(c['text'] or '?')}”</div></td>" for c in clips)
            voices.append(f"<tr><td>{esc(g['bench'].replace('homelab-tts-', ''))}</td>{cells}</tr>")
    if voices:
        h.append("<h3>Voices</h3><div class=wrap><table class=uc><tr><th>model</th><th class=left>short reply</th>"
                 "<th class=left>paragraph</th></tr>" + "".join(voices) + "</table></div>"
                 "<div class=note>Each model's first clip of the 1-user passes, with the text it was given: the same "
                 "texts for every model. Each speaks in its preset voice or a fixed voice-design prompt (tts.yaml); no voice cloning.</div>")
    h.append(license_footer(seen))
    return "\n".join(h), files


# --- image editing, upscaling, 3D: the gallery machinery ----------------------------------------------

EDIT_DIR = os.path.join(SRC_ROOT, "benchmarks", "comfyui_edit")


def media_rows(run_dir: str, rows: list[dict], out_dir: str, case: str, limit: int) -> tuple[list[dict], list[str]]:
    """The use case's benchmarks (single-pass, like the gallery's) with their kept outputs published."""
    entries, files = [], []
    for r in sorted((r for r in rows if use_case(r) == case and "." not in r["bench"]), key=lambda r: -(r.get("rate_mean") or 0)):
        found = _samples(run_dir, r, limit)
        samples = [_publish_sample(f, out_dir, r["bench"]) for f in found]
        prompts_file = os.path.join(run_dir, f"{r['bench']}.outputs", "prompts.json")
        try:
            captions = json.load(open(prompts_file)) if os.path.exists(prompts_file) else {}
        except (OSError, ValueError):
            captions = {}
        files += [path for _, path in samples]
        entries.append({"row": r, "samples": [(kind, path, captions.get(os.path.basename(f), ""))
                                               for (kind, path), f in zip(samples, found)]})
    return entries, files


def joules(unit: str):
    per = lambda r: r.get("rate_mean") and 1 / r["rate_mean"]
    return (f"J / {unit}", lambda r: fmt(r.get("power_mean_w") and per(r) and r["power_mean_w"] * per(r), 0, suffix=" J"))


def edit_section(run_dir: str, rows: list[dict], out_dir: str | None) -> tuple[str, list[str]]:
    if not out_dir or not any(use_case(r) == "edit" for r in rows):
        return "", []
    entries, files = media_rows(run_dir, rows, out_dir, "edit", 8)
    try:
        edits = json.load(open(os.path.join(EDIT_DIR, "edits.json"))).get("edits") or []
    except (OSError, ValueError):
        edits = []
    order = [e["instruction"] for e in edits]
    order += sorted({c for e in entries for _, _, c in e["samples"] if c and c not in order})
    if not order:
        return "", files
    inputs = {}
    for e in edits:
        src = os.path.join(EDIT_DIR, "inputs", e.get("image", ""))
        if os.path.exists(src) and e["image"] not in inputs:
            _, path = _publish_sample(src, out_dir, "edit-inputs")
            inputs[e["image"]] = path
            files.append(path)
    kind_of = {e["instruction"]: e.get("kind", "") for e in edits}
    image_of = {e["instruction"]: e.get("image") for e in edits}
    numbers = gen_numbers("edit")
    seen = {}
    head = ("<tr><th class=model>model</th>" + "".join(f"<th class={gen_cls(n)}>{esc(n)}</th>" for n, _ in numbers)
            + "".join(f"<th class=prompt title='{esc(p)}'>{i}. {esc(kind_of.get(p) or ' '.join(p.split()[:4]))}</th>"
                      for i, p in enumerate(order, 1)) + "</tr>")
    inrow = ("<tr class=inputs><td class=model>input image</td>" + "".join(f"<td class={gen_cls(n)}></td>" for n, _ in numbers)
             + "".join(f"<td><img src='{esc(inputs[image_of[p]])}' alt='input: {esc(image_of[p])}' title='input: {esc(image_of[p])}' loading=lazy></td>"
                       if image_of.get(p) in inputs else "<td class=na>–</td>" for p in order) + "</tr>")
    body = ""
    for e in entries:
        r = e["row"]
        by = {c: path for kind, path, c in e["samples"] if kind == "image" and c}
        status = "" if r["status"] == "ok" else f"<br><span class=bad>{esc(r['status'])}</span>"
        body += (f"<tr><td class=model>{esc(uc_name(r['bench']))}{lic(r['bench'], seen)}{status}</td>"
                 + "".join(f"<td class={gen_cls(n)}>{value(r)}</td>" for n, value in numbers)
                 + "".join(f"<td><a href='{esc(by[p])}' target=_blank><img src='{esc(by[p])}' alt='{esc(r['bench'])}: {esc(p)}' "
                           f"title='{esc(p)}' loading=lazy></a></td>" if p in by else "<td class=na>–</td>" for p in order) + "</tr>")
    html_ = ("<h2>Image editing</h2><ol class=prompts>" + "".join(f"<li>{esc(p)}</li>" for p in order) + "</ol>"
             "<div class=wrap><table class=same-prompt>" + head + inrow + body + "</table></div>"
             "<div class=note>One row per model, fastest first; one column per edit (numbered as listed above; hover a "
             "header or an image for the instruction), the first row the input image. Every model gets the same input "
             "images, instructions and seeds in the same order (benchmarks/comfyui_edit/edits.json), on ComfyUI or "
             "vLLM-Omni. Seconds per edit: one at a time, the run's mean, from the input image to the edited one. "
             "Energy and memory as for the text-to-image table.</div>" + license_footer(seen))
    return html_, files


def media_table(run_dir: str, rows: list[dict], out_dir: str | None, case: str, title: str, unit: str, note: str,
                limit: int = 3) -> tuple[str, list[str]]:
    if not out_dir or not any(use_case(r) == case for r in rows):
        return "", []
    entries, files = media_rows(run_dir, rows, out_dir, case, limit)
    if not entries:
        return "", files
    numbers = gen_numbers(unit)
    numbers[1] = joules(unit)
    n = max([len(e["samples"]) for e in entries] + [1])
    seen = {}
    head = ("<tr><th class=model>model</th>" + "".join(f"<th class={gen_cls(k)}>{esc(k)}</th>" for k, _ in numbers)
            + "".join(f"<th>{esc(unit)} {i}</th>" for i in range(1, n + 1)) + "</tr>")
    body = ""
    for e in entries:
        r = e["row"]
        status = "" if r["status"] == "ok" else f"<br><span class=bad>{esc(r['status'])}</span>"
        media = [f"<a href='{esc(path)}' target=_blank title='{esc(c)}'><img src='{esc(path)}' alt='{esc(r['bench'])} output' loading=lazy></a>"
                 if kind == "image" else f"<video src='{esc(path)}' controls muted loop playsinline preload=metadata></video>"
                 if kind == "video" else f"<span class=na>{esc(kind)}</span>" for kind, path, c in e["samples"]]
        if not media:
            media = ["<span class=na>no output kept</span>"]
        body += (f"<tr><td class=model>{esc(uc_name(r['bench']))}{lic(r['bench'], seen)}{status}</td>"
                 + "".join(f"<td class={gen_cls(k)}>{value(r)}</td>" for k, value in numbers)
                 + "".join(f"<td>{m}</td>" for m in media) + "<td></td>" * (n - len(media)) + "</tr>")
    html_ = (f"<h2>{esc(title)}</h2><div class=wrap><table class='same-prompt uc-{case}'>" + head + body + "</table></div>"
             f"<div class=note>{note}</div>" + license_footer(seen))
    return html_, files


# --- batch -------------------------------------------------------------------------------------------

BATCH_PASSES = [("summarize", "summarize"), ("classify", "classify"), ("longdoc", "long documents")]


def batch_section(rows: list[dict]) -> str:
    groups = uc_groups(rows, "batch")
    shown = [g for g in groups.values() if any(p in g["passes"] for p, _ in BATCH_PASSES)]
    if not shown:
        return ""
    passes = [(p, label) for p, label in BATCH_PASSES if any(p in g["passes"] for g in shown)]
    shown.sort(key=lambda g: -((g["passes"].get("summarize") or {}).get("request_throughput") or 0))
    h = ["<h2>Batch jobs</h2><div class=wrap><table class=uc><tr><th>model</th>"
         + "".join(f"<th class=grp>{esc(label)}: docs/hour</th><th>J per doc</th><th>kJ per 1M tokens</th><th>output tok/s</th>" for _, label in passes)
         + "<th class=grp>peak memory GiB</th></tr>"]
    for g in shown:
        cells = ""
        for p, _ in passes:
            r = g["passes"].get(p) or {}
            tokens = (r.get("total_throughput") or 0) * (r.get("pass_duration_s") or 0)
            cells += (f"<td class=grp>{fmt(r.get('request_throughput') and r['request_throughput'] * 3600, 0)}</td>"
                      f"<td>{fmt(per_pass_energy(r, r.get('completed')), 0)}</td>"
                      f"<td>{fmt(r.get('energy_j') and tokens and r['energy_j'] / tokens * 1e6 / 1000, 1)}</td>"
                      f"<td>{fmt(r.get('output_throughput'), 0)}</td>")
        mem = max((r.get("mem_peak_gib") or 0 for r in g["rows"]), default=0) or None
        h.append("<tr>" + uc_cell(g) + cells + f"<td class=grp>{fmt(mem)}</td></tr>")
    h.append("</table></div><div class=note>Offline jobs where only volume matters: every request sent at once, the server "
             "batching up to 256 (64 for long documents), no speculative decoding. summarize: 2,048-token documents to "
             "256-token summaries; classify: 1,024-token documents to an 8-token label; long documents: 8,192 tokens to "
             "512. docs/hour: completed requests per second × 3,600 in steady state (warm-up excluded). J per doc and kJ "
             "per 1M tokens (prompt and output tokens together): GPU energy over the pass.</div>")
    return "\n".join(h)


# --- fine-tuning -------------------------------------------------------------------------------------

def finetune_section(rows: list[dict]) -> str:
    groups = uc_groups(rows, "finetune")
    llm = [g for g in groups.values() if any(r.get("ft.train_tokens_per_s") for r in g["rows"])]
    diff = [g for g in groups.values() if g not in llm and any(r.get("ft.steps_per_s") or r.get("rate_mean") for r in g["rows"])]
    if not (llm or diff):
        return ""
    seen = {}
    h = ["<h2>Fine-tuning</h2>"]
    if llm:
        first = lambda g, k: next((r.get(k) for r in g["rows"] if r.get(k) is not None), None)
        llm.sort(key=lambda g: -(first(g, "ft.train_tokens_per_s") or 0))
        h.append("<h3>Language models (LoRA / QLoRA)</h3><div class=wrap><table class=uc><tr><th>model</th><th>method</th>"
                 "<th class=grp>training tok/s</th><th>s per step p50</th><th>tokens per step</th><th class=grp>peak memory GiB</th>"
                 "<th>allocated by training GiB</th><th class=grp>kWh per 1M tokens</th><th>loss first → last</th></tr>")
        for g in llm:
            ok = first(g, "ft.loss_ok")
            loss = (f"{first(g, 'ft.loss_start'):.2f} → {first(g, 'ft.loss_end'):.2f}"
                    if first(g, "ft.loss_start") is not None and first(g, "ft.loss_end") is not None else None)
            method = "QLoRA (4-bit base)" if first(g, "ft.qlora") or "qlora" in g["tags"] else "LoRA"
            mem = first(g, "ft.system_mem_peak_gib") or max((r.get("mem_peak_gib") or 0 for r in g["rows"]), default=0) or None
            h.append("<tr>" + uc_cell(g, seen) + f"<td class=left>{esc(method)}</td>"
                     f"<td class=grp>{fmt(first(g, 'ft.train_tokens_per_s'), 0)}</td><td>{fmt(first(g, 'ft.step_time_s_p50'), 2)}</td>"
                     f"<td>{fmt(first(g, 'ft.tokens_per_step'), 0)}</td><td class=grp>{fmt(mem)}</td>"
                     f"<td>{fmt(first(g, 'ft.torch_peak_reserved_gib'))}</td><td class=grp>{fmt(first(g, 'ft.kwh_per_1m_tokens'), 3)}</td>"
                     f"<td class={'ok' if ok else 'bad' if ok is False else ''}"
                     f"{' title=' + chr(39) + 'the loss did not drop as expected' + chr(39) if ok is False else ''}>{esc(loss) if loss else fmt(None)}</td></tr>")
        h.append("</table></div><div class=note>The same job for every model: fixed-length packed sequences of real "
                 "instructions, a fixed number of sequences per step, so tokens per step is the same within a size class. "
                 "The first steps (warm-up) are not timed. Peak memory: the machine's unified memory at its highest "
                 "(weights, activations and optimizer); allocated by training: PyTorch's own peak reservation. kWh per 1M "
                 "tokens: GPU energy over the timed steps. Loss: mean of the first and last steps; red when it did not "
                 "drop as expected (the run trained nothing useful).</div>")
    if diff:
        diff.sort(key=lambda g: -max((r.get("ft.images_per_s") or r.get("rate_mean") or 0) for r in g["rows"]))
        h.append("<h3>Image models (diffusion LoRA)</h3><div class=wrap><table class=uc><tr><th>model and tool</th>"
                 "<th class=grp>steps/s</th><th>images/s</th><th>steps</th><th class=grp>peak memory GiB</th>"
                 "<th>kWh per 1000 steps</th><th>loss first → last</th></tr>")
        for g in diff:
            r = max(g["rows"], key=lambda r: r.get("ft.images_per_s") or r.get("rate_mean") or 0)
            steps = r.get("ft.steps")
            kwh = r.get("energy_j") and steps and r["energy_j"] / steps * 1000 / 3.6e6
            h.append("<tr>" + uc_cell(g, seen) + f"<td class=grp>{fmt(r.get('ft.steps_per_s'), 2)}</td>"
                     f"<td>{fmt(r.get('ft.images_per_s') or r.get('rate_mean'), 2)}</td><td>{fmt(steps, 0)}</td>"
                     f"<td class=grp>{fmt(r.get('mem_peak_gib'))}</td><td>{fmt(kwh, 3)}</td><td>{fmt(None)}</td></tr>")
        h.append("</table></div><div class=note>The same LoRA job in each tool (rank 16, batch 1, 1024 px, 200 steps, "
                 "bf16) on the same images. Steps/s and images/s: the tool's whole training run, wall clock (its model "
                 "loading included). kWh per 1000 steps: GPU energy over the run. The tools report no loss to the "
                 "benchmark yet.</div>")
    h.append(license_footer(seen))
    return "\n".join(h)


def use_cases(run_dir: str, rows: list[dict], out_dir: str | None) -> tuple[str, list[str]]:
    """Every use-case section with data, and the media files they link."""
    specs, files, h = {}, [], []

    def safe(build, name):  # a section that fails on new data says so; the rest of the page still renders
        try:
            return build()
        except Exception as exc:  # noqa: BLE001
            return (f"<p class=sub>The {esc(name)} section could not be drawn ({esc(type(exc).__name__)}: "
                    f"{esc(str(exc)[:200])}).</p>", [])

    for name, build in (("tool calling", lambda: toolcall_section(rows, specs)), ("embeddings", lambda: embed_section(rows, specs)),
                        ("long context", lambda: longctx_section(rows, specs, run_dir)),
                        ("vision", lambda: vision_section(rows, specs)), ("OCR", lambda: ocr_section(rows, specs)),
                        ("text to speech", lambda: tts_section(rows, run_dir, out_dir)), ("batch", lambda: batch_section(rows)),
                        ("fine-tuning", lambda: finetune_section(rows))):
        part = safe(build, name)
        h.append(part[0] if isinstance(part, tuple) else part)
        if isinstance(part, tuple) and len(part) > 1 and isinstance(part[1], list) and all(isinstance(f, str) for f in part[1]):
            files += part[1]  # a section with media of its own (text to speech: the voices)
    media = []
    for build in (lambda: edit_section(run_dir, rows, out_dir),
                  lambda: media_table(run_dir, rows, out_dir, "upscale", "Upscaling", "image",
                                      "One row per upscaler, fastest first; 1024 × 1024 in, 4096 × 4096 out (×4). Each "
                                      "output is a crop of the same 512-pixel region: plain resizing (lanczos) on the left, "
                                      "the model on the right. Seconds per image: one at a time, the run's mean. Energy: the "
                                      "GPU's mean power times the seconds per image."),
                  lambda: media_table(run_dir, rows, out_dir, "3d", "3D assets", "asset",
                                      "One row per model, fastest first; the same object photo in, one 3D asset out per "
                                      "generation (shown: a rendered view). Seconds per asset: the run's mean, background "
                                      "removal and mesh processing included. Energy: the GPU's mean power times the seconds "
                                      "per asset.")):
        part, more = safe(build, "image")
        media.append(part)
        files += more
    h = h[:6] + media + h[6:]  # the order on the page: served models, then images, then batch and fine-tuning
    h = [x for x in h if x]
    if specs:
        h.append(UC_JS.replace("__SPECS__", json.dumps(specs).replace("</", "<\\/")))
    return "\n".join(h), files



CSS = """
/* layout: a run dashboard: status strip, model table (summary before detail), charts, failures, per-pass log */
:root{
  --bg:#f6f7f6;--surface:#ffffff;--fg:#17201c;--muted:#5d6a64;--line:#dde3df;
  --accent:#1f6f5c;--ok:#23794f;--bad:#b42318;--run:#a15c07;
  --c0:#1f6f5c;--c1:#d07a2d;--c2:#4f6fb0;--c3:#a8507e;--c4:#7d8f2e;--c5:#6b5bb5;--c6:#2f8fa3;--c7:#8a8f8c;
  --font-display:"Space Grotesk",system-ui,sans-serif;--font-body:"Inter",system-ui,-apple-system,"Segoe UI",sans-serif;
  --font-data:"JetBrains Mono",ui-monospace,"SF Mono",Menlo,monospace;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --bg:#111614;--surface:#18201d;--fg:#e6ece9;--muted:#93a29b;--line:#2a3531;
  --accent:#5cc2a4;--ok:#6fd19c;--bad:#f2867a;--run:#e8b04a;
  --c0:#5cc2a4;--c1:#eb9b55;--c2:#86a3e0;--c3:#d88ab2;--c4:#b3c55c;--c5:#a597e6;--c6:#6cc3d6;--c7:#a3aaa6;color-scheme:dark}}
:root[data-theme="dark"]{
  --bg:#111614;--surface:#18201d;--fg:#e6ece9;--muted:#93a29b;--line:#2a3531;
  --accent:#5cc2a4;--ok:#6fd19c;--bad:#f2867a;--run:#e8b04a;
  --c0:#5cc2a4;--c1:#eb9b55;--c2:#86a3e0;--c3:#d88ab2;--c4:#b3c55c;--c5:#a597e6;--c6:#6cc3d6;--c7:#a3aaa6;color-scheme:dark}
body{background:var(--bg);color:var(--fg);font:14px/1.5 var(--font-body)}
main{max-width:1500px;margin:0 auto;padding-block:24px 48px;
  padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}
h1{font:600 24px/1.2 var(--font-display);margin:0 0 4px;text-wrap:balance}
h2{font:600 17px/1.3 var(--font-display);margin:32px 0 8px;text-wrap:balance}
.sub{color:var(--muted)}
.now{margin:.35rem 0 0;font-size:.95rem;overflow-wrap:anywhere}
.cards{display:flex;flex-wrap:wrap;gap:10px;margin:16px 0}
.card{background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:8px 14px;min-width:110px;
  font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.card b{display:block;font:600 22px/1.3 var(--font-data);color:var(--fg);letter-spacing:0;text-transform:none}
.wrap{overflow-x:auto;background:var(--surface);border:1px solid var(--line);border-radius:6px;min-width:0}
table{border-collapse:collapse;width:100%;font:12.5px/1.4 var(--font-data);font-variant-numeric:tabular-nums}
th,td{padding:5px 9px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th{position:sticky;top:0;background:var(--surface);font:600 11px/1.3 var(--font-body);color:var(--muted);
  text-transform:uppercase;letter-spacing:.04em}
td:first-child,th:first-child{text-align:left;font-family:var(--font-body)}
tr:hover td{background:color-mix(in srgb,var(--accent) 7%,transparent)}
.na{color:var(--muted)}.ok{color:var(--ok)}.bad{color:var(--bad)}.run{color:var(--run)}
/* Vega's tooltip box: the page's colors (its own theme is light whatever the page's) */
#vg-tooltip-element.vg-tooltip{background:var(--surface)!important;color:var(--fg)!important;border:1px solid var(--line)!important;box-shadow:0 4px 14px rgba(0,0,0,.35)!important;border-radius:6px!important;font:12px Inter,system-ui,sans-serif!important}
#vg-tooltip-element.vg-tooltip td.key{color:var(--muted)!important}#vg-tooltip-element.vg-tooltip td.value{color:var(--fg)!important}#vg-tooltip-element.vg-tooltip h2{color:var(--fg)!important}
.grp{border-left:2px solid var(--line)}
figure{margin:14px 0;background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:12px;min-width:0}
figcaption{font:600 14px/1.3 var(--font-display);margin-bottom:6px}
.gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,420px),1fr));gap:12px}
.gen{background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:10px 12px;min-width:0}
.gen-head{display:flex;justify-content:space-between;gap:8px;font-size:13px}.gen-head span{font:12px var(--font-data)}
.gen-prompt{color:var(--muted);font-size:12px;margin:4px 0 8px}
.gen-media{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:6px}
.gen-media img,.gen-media video{width:100%;aspect-ratio:1/1;object-fit:cover;border-radius:4px;display:block;background:var(--line)}
.gen-media video{aspect-ratio:16/9;object-fit:contain}.gen-media audio{width:100%;grid-column:1/-1}
table.same-prompt td{vertical-align:middle}table.same-prompt th.prompt{white-space:normal;min-width:120px;max-width:130px;font-weight:400;font-size:12px}table.same-prompt td.num,table.same-prompt th.num{text-align:right;font-size:12px;white-space:nowrap}table.same-prompt td.eng,table.same-prompt th.eng{text-align:left;font-size:12px;white-space:normal;min-width:12ch;max-width:18ch}table.same-prompt .model{position:sticky;left:0;z-index:1;background:var(--surface);font:12px var(--font-data);white-space:nowrap}table.same-prompt img{width:120px;height:120px;object-fit:cover;border-radius:4px;display:block}ol.prompts{margin:.25rem 0 .75rem;padding-left:1.5rem;font-size:13px;color:var(--muted)}ol.prompts li{margin:.1rem 0}table.same-prompt td.gprompt,table.same-prompt th.gprompt{text-align:left}table.same-prompt td.gprompt div{min-width:26ch;max-width:34ch;text-align:left;font-size:12px;color:var(--muted);display:-webkit-box;-webkit-line-clamp:4;-webkit-box-orient:vertical;overflow:hidden;white-space:normal}table.same-prompt td.video video{width:200px;display:block;border-radius:4px}table.same-prompt td.audio audio{width:220px;display:block}
.vega{width:100%;min-height:120px}.vega svg{max-width:100%}
#vg-tooltip-element{font:12px var(--font-body)}
ul{margin:6px 0;padding-left:20px}li{margin:3px 0}code{font:12px var(--font-data)}
.note{color:var(--muted);font-size:12px;margin-top:6px;max-width:110ch}
.progress{background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:12px 14px;margin:6px 0 4px}
.bar{display:flex;height:12px;border-radius:6px;overflow:hidden;background:var(--line);margin:8px 0 6px}
.bar span{display:block;height:100%}.bar .s-ok{background:var(--ok)}.bar .s-bad{background:var(--bad)}.bar .s-run{background:var(--run)}
.legend{display:flex;flex-wrap:wrap;gap:14px;font-size:12px;color:var(--muted)}
.chips{display:flex;flex-wrap:wrap;gap:4px;justify-content:flex-start}
.chip{display:inline-block;font:11px/1.6 var(--font-data);padding:0 7px;border-radius:9px;border:1px solid var(--line);color:var(--muted)}
.chip.ok{border-color:color-mix(in srgb,var(--ok) 45%,transparent);color:var(--ok);background:color-mix(in srgb,var(--ok) 10%,transparent)}
.chip.bad{border-color:color-mix(in srgb,var(--bad) 45%,transparent);color:var(--bad);background:color-mix(in srgb,var(--bad) 10%,transparent)}
.chip.run{border-color:var(--run);color:var(--run);background:color-mix(in srgb,var(--run) 14%,transparent);font-weight:600}
td.plan{text-align:left;white-space:normal}
.desc{display:block;font:11px/1.35 var(--font-body);color:var(--muted);white-space:normal;max-width:46ch}
td.left{text-align:left;white-space:normal}
.presets{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0}
.presets button{font:12px var(--font-body);color:var(--fg);background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:3px 11px;cursor:pointer}
.presets button:hover{border-color:var(--accent)}
table.board th.crit{text-align:center;vertical-align:bottom;min-width:96px;white-space:normal}
table.board th .unit{font-weight:400;text-transform:none;letter-spacing:0}
table.board th input{width:84px;accent-color:var(--accent);display:block;margin:4px auto 0}
table.board th output{font:11px var(--font-data);color:var(--fg)}
table.board td.left{font-family:var(--font-body)}
table.board td.bc{position:relative;min-width:96px;text-align:right}
table.board td.bc .bar-bg{position:absolute;left:6px;right:6px;top:50%;height:16px;transform:translateY(-50%);border-radius:3px;overflow:hidden}
table.board td.bc .bar-bg i{display:block;height:100%;background:color-mix(in srgb,var(--accent) 22%,transparent)}
table.board td.bc span,table.board td.bc sup{position:relative}
table.board td.overall .bar-bg i{background:color-mix(in srgb,var(--accent) 50%,transparent)}
table.board td.overall{border-right:2px solid var(--line)}
table.board td.bc sup{color:var(--muted);font-size:9px;margin-left:2px}
tr.skip td{color:var(--muted)}
.lic{display:inline-block;font:10px/1.5 var(--font-body);padding:0 5px;margin-left:4px;border-radius:7px;cursor:help;
  border:1px solid color-mix(in srgb,var(--run) 55%,transparent);color:var(--run);background:color-mix(in srgb,var(--run) 10%,transparent);vertical-align:1px}
table.uc td.model{white-space:nowrap}table.uc td.model .desc{white-space:normal}
table.same-prompt tr.inputs td{background:color-mix(in srgb,var(--line) 35%,transparent)}
table.uc-upscale img{width:auto;height:120px;max-width:260px;object-fit:contain}\nsummary{cursor:pointer;margin:32px 0 8px}summary h2{margin:0}
"""

FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
         '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&family=JetBrains+Mono:wght@400;600'
         '&family=Space+Grotesk:wght@600&display=swap">')


def progress(plan: list[dict], rows: list[dict]) -> dict:
    """Every planned result with its state: ok / failed / running / pending / skipped."""
    by_name = {r["bench"]: r for r in rows}
    entries, counts, durations = [], defaultdict(int), []
    for bench in plan:
        results = [f"{bench['bench']}.{p}" for p in bench["passes"]] or [bench["bench"]]
        states = []
        whole = by_name.get(bench["bench"])  # a benchmark that failed as a whole (its server died)
        for name in results:
            row = by_name.get(name) or (whole if whole and whole["status"] not in ("ok", "running") else None)
            if bench.get("skipped"):
                state = "skipped"
            elif row is None:
                state = "pending"
            elif row["status"] == "ok":
                state = "ok"
            elif row["status"] == "running":
                state = "running"
            else:
                state = "failed"
            if row and (row.get("duration_s") or 0) > 0 and state in ("ok", "failed"):
                durations.append(row["duration_s"])
            states.append((name.rsplit(".", 1)[-1] if bench["passes"] else "run", state, row))
            counts[state] += 1
        rows_done = [r for _, _, r in states if r and (r.get("duration_s") or 0) > 0]
        entries.append({**bench, "states": states, "minutes": sum(r["duration_s"] for r in rows_done) / 60 if rows_done else None})
    total = sum(v for k, v in counts.items() if k != "skipped")
    left = counts["pending"] + counts["running"]
    eta_h = (sum(durations) / len(durations)) * left / 3600 if durations else None
    return {"entries": entries, "counts": counts, "total": total, "eta_h": eta_h}


def session_totals(run_dir: str) -> dict:
    """Every monitoring session (nodemon/<session>/nodes/*.gpu.tsv): GPU energy (counter, last minus first),
    time measured (last minus first sample), and the first and last sample of all (epoch s)."""
    import glob

    out = {"energy_j": 0.0, "seconds": 0.0, "first": None, "last": None}
    for path in glob.glob(os.path.join(run_dir, "nodemon", "*", "nodes", "*.gpu.tsv")):
        t0 = t1 = e0 = e1 = None
        try:
            with open(path, errors="replace") as f:
                head = f.readline().rstrip("\n").split("\t")
                ct = head.index("time_ms") if "time_ms" in head else None
                ce = head.index("energy_j") if "energy_j" in head else None
                for line in f:
                    parts = line.split("\t")
                    try:
                        t = float(parts[ct]) / 1000 if ct is not None else None
                    except (IndexError, ValueError):
                        t = None
                    if t is not None:
                        t0 = t if t0 is None else t0
                        t1 = t
                    try:
                        e = float(parts[ce]) if ce is not None else None
                    except (IndexError, ValueError):
                        e = None
                    if e is not None:
                        e0 = e if e0 is None else e0
                        e1 = e
        except OSError:
            continue
        if t0 is not None and t1 > t0:
            out["seconds"] += t1 - t0
            out["first"] = t0 if out["first"] is None else min(out["first"], t0)
            out["last"] = t1 if out["last"] is None else max(out["last"], t1)
        if e0 is not None and e1 > e0:
            out["energy_j"] += e1 - e0
    return out


def fmt_hours(seconds: float) -> str:
    h = seconds / 3600
    return f"{h:.1f} h" if h < 48 else f"{h:.0f} h <span class=sub>({h / 24:.1f} days)</span>"


def energy_spent_j(run_dir: str) -> float:
    """All the GPU energy measured while the suite ran: each monitoring session's energy counter, last minus first
    (reruns, superseded and archived runs, loading and idle time between passes included)."""
    import glob

    total = 0.0
    for path in glob.glob(os.path.join(run_dir, "nodemon", "*", "nodes", "*.gpu.tsv")):
        first = last = None
        try:
            with open(path, errors="replace") as f:
                head = f.readline().rstrip("\n").split("\t")
                col = head.index("energy_j") if "energy_j" in head else None
                if col is None:
                    continue
                for line in f:
                    parts = line.split("\t")
                    try:
                        v = float(parts[col])
                    except (IndexError, ValueError):
                        continue
                    first = v if first is None else first
                    last = v
        except OSError:
            continue
        if first is not None and last is not None and last > first:
            total += last - first
    return total


def live_benches(run: str) -> dict[str, float]:
    """The benchmarks a live `milabench run --run-name <run>` is on (its --select), with seconds since it started.
    A result file can stay untouched for hours while a long pass (QualityBench) runs, so its age says nothing."""
    out = subprocess.run(["ps", "-eo", "etimes=,args="], capture_output=True, text=True).stdout
    live = {}
    for line in out.splitlines():
        etimes, _, args = line.strip().partition(" ")
        argv = args.split()
        if "run" not in argv or f"--run-name" not in argv or argv[argv.index("--run-name") + 1] != run:
            continue
        if not any(a.endswith("milabench") for a in argv[:2]) or "--select" not in argv:
            continue
        for bench in argv[argv.index("--select") + 1].split(","):
            live[bench] = max(live.get(bench, 0), float(etimes))
    return live


def qb_progress(run_dir: str, bench: str) -> str | None:
    """Where a live QualityBench pass is: tasks done of planned, the one in progress (LiveCodeBench: problems answered)."""
    import glob
    logs = sorted(glob.glob(os.path.join(run_dir, f"{bench}.*.qualitybench", "*", "run.log")), key=os.path.getmtime)
    if not logs:
        return None
    text = open(logs[-1], errors="replace").read()
    planned = re.search(r"tasks ([\w,]+) \(", text)
    planned = planned.group(1).split(",") if planned else []
    done = re.findall(r"\] (\w+): done in", text)
    started = re.findall(r"\] (\w+): start", text)
    current = next((t for t in reversed(started) if t not in done), None)
    msg = f"{len(done)}/{len(planned) or '?'} tasks"
    if current:
        msg += f", on {current}"
        qb = os.path.dirname(logs[-1])
        ids, raw = os.path.join(qb, current, "ids.txt"), os.path.join(qb, current, "raw.jsonl")
        if os.path.exists(ids):
            n = sum(1 for l in open(ids) if l.strip())
            got = sum(1 for l in open(raw) if l.strip()) if os.path.exists(raw) else 0
            msg += f" ({got}/{n} answered)"
        t0 = re.findall(rf"\[([\d-]+ [\d:]+)\] {current}: start", text)
        if t0:
            mins = (datetime.datetime.now() - datetime.datetime.strptime(t0[-1], "%Y-%m-%d %H:%M:%S")).total_seconds() / 60
            msg += f" for {mins:.0f} min"
    return msg


def page(run_dir: str, rows: list[dict], out_dir: str | None = None) -> str:
    run = os.path.basename(os.path.normpath(run_dir))
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    live = live_benches(run)
    totals = session_totals(run_dir)
    for r in rows:  # a quiet result file of a benchmark still running: not "missing-end"
        if r["status"] == "missing-end" and r["bench"].split(".")[0] in live:
            r["status"] = "running"
    status = defaultdict(int)
    for r in rows:
        status[r["status"]] += 1
    running = sorted({r["bench"].split(".")[0] for r in rows if r["status"] == "running"} | set(live))
    gpu = next((r.get("gpu") for r in rows if r.get("gpu")), "?")
    alive = subprocess.run(["pgrep", "-f", f"run-name {run}"], capture_output=True).returncode == 0
    quarantine = {}
    qfile = os.path.join(os.environ.get("MILABENCH_BASE", "/opt/milabench"), "homelab-quarantine.json")
    if os.path.exists(qfile):
        quarantine = {k: v for k, v in json.load(open(qfile)).items() if not k.startswith("_")}
    # the use cases' served models (tool calling, embeddings, OCR...) have their own sections, not the LLM tables
    models = model_rows([r for r in rows if use_case(r) not in NOT_LLM_SERVING],
                        {b for b, e in quarantine.items() if e.get("kind") == "unsupported"})
    share_full_lcb(models)

    kills = []
    if os.path.exists(os.path.join(run_dir, "memguard.log")):
        kills = [json.loads(l) for l in open(os.path.join(run_dir, "memguard.log")) if '"kill"' in l]

    # an Artifact page: the host wraps it in its own document (no html / head / body here)
    h = [f'<title>Homelab Benchmarks</title><meta name="claude:viewport-fit-sides" content="cover">{FONTS}<style>{CSS}</style><main>',
         f"<h1>Homelab benchmarks — {esc(gpu)}</h1>",
         f"<div class=sub>run <code>{esc(run)}</code> · updated {now} · "
         f"{'<span class=run>running</span>' if alive else 'finished'}</div>",
         *[f"<div class=now><span class=run>● now</span> <code>{esc(b)}</code>"
           + (f" · {live[b] / 3600:.1f} h in" if b in live else "")
           + (f" · {esc(p)}" if (p := qb_progress(run_dir, b)) else "") + "</div>" for b in running],
         "<div class=cards>",
         f"<div class=card>passed<b class=ok>{status['ok']}</b></div>",
         f"<div class=card>failed<b class=bad>{status['failed'] + status['missing-end'] + status['killed-oom']}</b></div>",
         f"<div class=card>running<b class=run>{status['running']}</b></div>",
         f"<div class=card>models measured<b>{len(models)}</b></div>",
         f"<div class=card>memguard kills<b>{len(kills)}</b></div>",
         f"<div class=card title='GPU energy of every monitored session: reruns, superseded runs, loading and idle time "
         f"included. Only grows.'>energy spent<b>{totals['energy_j'] / 3.6e6:.2f} kWh</b></div>",
         f"<div class=card title='Time the GPU was monitored, all sessions: every benchmark, rerun, superseded run, "
         f"model download and load, and the gaps between passes. Only grows.'>time spent<b>{fmt_hours(totals['seconds'])}</b></div>",
         f"<div class=card title='Duration of the results in the tables, measuring only (model download and load excluded; one "
         f"run per benchmark; a run in progress counts its time so far). The rest of time spent: loading, reruns, "
         f"superseded runs and the gaps between passes.'>time, results shown<b>{fmt_hours(sum(max(r.get('duration_s') or 0, 0) for r in rows))}</b></div>",
         f"<div class=card title='From the first monitored sample to the last: calendar time, the pauses between "
         f"sessions included.'>calendar span<b>{fmt_hours((totals['last'] or 0) - (totals['first'] or 0))}</b></div>",
         f"<div class=card title='GPU energy of the results in the tables (one run per benchmark: a rerun replaces the "
         f"run before it, and a run in progress counts what it has used so far).'>energy, results shown"
         f"<b>{sum(r.get('energy_j') or 0 for r in rows) / 3.6e6:.2f} kWh</b></div>",
         "</div>"]

    plan_file = os.path.join(run_dir, "report", "plan.json")
    prog = progress(json.load(open(plan_file)), rows) if os.path.exists(plan_file) else None
    if prog and prog["total"]:
        c, total = prog["counts"], prog["total"]
        pct = lambda n: f"{100 * n / total:.2f}%"
        eta = f" · about {prog['eta_h']:.0f} h left at the average pace so far" if prog["eta_h"] and alive else ""
        h.append(f"<div class=progress><b>{c['ok'] + c['failed']} of {total} results</b> "
                 f"<span class=sub>({100 * (c['ok'] + c['failed']) / total:.0f}%){eta}</span>"
                 f"<div class=bar role=img aria-label='progress'><span class=s-ok style='width:{pct(c['ok'])}'></span>"
                 f"<span class=s-bad style='width:{pct(c['failed'])}'></span><span class=s-run style='width:{pct(c['running'])}'></span></div>"
                 f"<div class=legend><span class=ok>■ passed {c['ok']}</span><span class=bad>■ failed {c['failed']}</span>"
                 f"<span class=run>■ running {c['running']}</span><span>■ pending {c['pending']}</span>"
                 f"<span>skipped {c['skipped']} (quarantine)</span></div></div>")

    # model tables: what to compare (speed, accuracy), then what it costs (memory, energy)
    passes = [p for p in PASS_ORDER if any(p in m["passes"] for m in models)]
    hung = {b for b, e in quarantine.items() if e.get("kind") == "hang"}

    def state(m):
        if m["benches"] & hung:
            return "bad", "hung the machine (quarantined)"
        if "running" in m["status"]:
            return "run", "running"
        if m["status"] - {"ok", "running"}:
            return "bad", "some passes failed"
        return "ok", ""

    def name_cell(m):
        css, note = state(m)
        flag = f" <span class={css} style='font-size:11px'>· {esc(note)}</span>" if note else ""
        return f"<td><span class={css}>{esc(m['bench'].replace('homelab-', ''))}</span>{flag}{describe_html(m)}</td>"

    speed = [  # (header, getter, digits)
        ("1 user tok/s", lambda m: per_user(m["passes"].get("single") or m["passes"].get("conc1")), 1),
        ("1 user TTFT ms", lambda m: (m["passes"].get("single") or m["passes"].get("conc1") or {}).get("ttft_ms_p50"), 0),
        ("4k prompt TTFT ms", lambda m: (m["passes"].get("fill4k") or {}).get("ttft_ms_p50"), 0),
        ("8 users tok/s", lambda m: (m["passes"].get("conc8") or {}).get("output_throughput"), 0),
        ("8 users tok/s/user", lambda m: per_user(m["passes"].get("conc8")), 1),
        ("32 users tok/s", lambda m: (m["passes"].get("conc32") or {}).get("output_throughput"), 0),
        ("32 users tok/s/user", lambda m: per_user(m["passes"].get("conc32")), 1),
    ]
    tasks = [(t, label) for t, label in TASKS if any(acc(m, t) is not None for m in models)]
    weights = quality_weights(models)
    extra = [("ppl wiki", "ppl.wiki")] if any(m.get("ppl.wiki") for m in models) else []
    shown = [m for m in models if m["passes"] or any(acc(m, t) is not None for t, _ in tasks)]
    h.append(leaderboard(shown, weights, run_dir, rows))
    quality_html = charts(shown, weights, run_dir, rows)
    try:
        tb = tb_section(run_dir, shown, weights)
    except Exception as exc:  # noqa: BLE001  new data: the rest of the page still renders
        tb = f"<p class=sub>The Terminal-Bench chart could not be drawn ({esc(type(exc).__name__)}: {esc(str(exc)[:200])}).</p>"
    h.append(quality_html.replace("<h2>Throughput</h2>", tb + "<h2>Throughput</h2>", 1))
    try:
        h.append(release_section(shown, weights))
    except Exception as exc:  # noqa: BLE001  new data: the rest of the page still renders
        h.append(f"<p class=sub>The release-date chart could not be drawn ({esc(type(exc).__name__)}: {esc(str(exc)[:200])}).</p>")
    h.append(thinking_section(shown, weights, run_dir))
    try:
        h.append(timing_section(run_dir))
    except Exception as exc:  # noqa: BLE001  new data: the rest of the page still renders
        h.append(f"<p class=sub>The time-to-solution section could not be drawn ({esc(type(exc).__name__)}: {esc(str(exc)[:200])}).</p>")
    tables_at = len(h)  # the big per-model tables: built here, placed after the charts and use cases (below)
    h.append("<h2>LLM serving: compare</h2><div class=wrap><table><tr><th>model</th>"
             + "".join(f"<th{' class=grp' if i in (0, 3, 5) else ''}>{esc(label)}</th>" for i, (label, _, _) in enumerate(speed))
             + "<th class=grp>quality score</th><th>thinking</th>"
             + "".join(f"<th>{esc(label)}</th>" for _, label in tasks)
             + "".join(f"<th>{esc(label)}</th>" for label, _ in extra) + "</tr>")
    for m in shown:
        h.append("<tr>" + name_cell(m)
                 + "".join(f"<td{' class=grp' if i in (0, 3, 5) else ''}>{fmt(get(m), digits)}</td>" for i, (_, get, digits) in enumerate(speed))
                 + f"<td class=grp><b>{fmt(quality_score(m, weights))}</b>{'*' if thinking_only(m) and quality_score(m, weights) is not None else ''}</td>"
                 + f"<td>{'<span class=run>always on</span>' if thinking_only(m) else 'off' if acc(m, 'gsm8k') is not None else fmt(None)}</td>"
                 + "".join(f"<td>{fmt(acc(m, t), 1, 100)}{'*' if thinking_only(m) and acc(m, t) is not None else ''}</td>" for t, _ in tasks)
                 + "".join(f"<td>{fmt(m.get(key), 2)}</td>" for _, key in extra) + "</tr>")
    h.append("</table></div><div class=note>tok/s: output tokens per second of all streams; tok/s/user: what one stream "
             "sees while decoding (1000 / TPOT p50); with one user both are the same. TTFT: time to the first token, p50. "
             "Accuracy: QualityBench quick tier (a fixed tenth of each task), thinking off, %; * thinking on, for models that "
             "can't turn it off (gpt-oss, Muse). Quality score: the tasks' weighted mean, weights in the Quality section above. "
             + "".join(f"Left out: {esc(t)} of {esc(prefix.replace('homelab-vllm-', ''))}* ({esc(why)}). " for (prefix, t), why in INVALID.items())
             + "Speculative decoding (MTP, DFlash) speeds are flattered by the fixed-length outputs. "
             "Every pass and its p99s: the per-pass log at the end.</div>")

    h.append("<h2>LLM serving: footprint and energy</h2><div class=wrap><table><tr><th>model</th><th>disk GB</th>"
             "<th>weights GiB</th><th>KV cache GiB</th><th>memory peak GiB</th><th class=grp>power W</th><th>temp peak °C</th>"
             "<th>energy kJ</th><th>tok/J</th></tr>")
    for m in shown:
        best = max(m["passes"].values(), key=lambda p: p.get("output_throughput") or 0, default={})
        h.append("<tr>" + name_cell(m)
                 + f"<td>{fmt(m.get('disk_gb'))}</td><td>{fmt(m.get('vllm.weights_gib'))}</td><td>{fmt(m.get('vllm.kv_cache_gib'))}</td>"
                 f"<td>{fmt(m.get('mem_peak_gib'))}</td><td class=grp>{fmt(best.get('power_mean_w'))}</td><td>{fmt(m.get('temp_peak_c'), 0)}</td>"
                 f"<td>{fmt((m.get('energy_j') or None) and m['energy_j'] / 1000)}</td><td>{fmt(best.get('output_tokens_per_joule'), 2)}</td></tr>")
    h.append("</table></div><div class=note>disk: the model's files (GB). weights / KV cache: vLLM's own report (GiB); the KV cache "
             "is sized for the benchmark's passes on unified memory. memory peak: the machine's memory in use (unified: GPU "
             "and system). power and tok/J: at the model's fastest pass; energy: all its passes, accuracy included.</div>")

    h.append(pass_definitions(models, passes))

    others = [r for r in rows if r.get("rate_mean") is not None and use_case(r) is None]  # use cases: their own sections
    if others:
        h.append("<h2>Other benchmarks</h2><div class=wrap><table><tr><th>benchmark</th><th>metric</th><th>mean</th><th>s per item</th>"
                 "<th>accuracy (QualityBench quick, %)</th>"
                 "<th>disk GB</th><th>mem peak GiB</th><th>power W</th><th>temp °C</th><th>energy kJ</th><th>status</th></tr>")
        for r in others:
            accuracy = same_weights_accuracy(r, models) if (r.get("what") or {}).get("engine", "").startswith("HF transformers generate") else '<span class="na">–</span>'
            h.append(f"<tr><td><span>{esc(r['bench'].replace('homelab-', ''))}</span>{describe_html(r)}</td><td class=left>{esc(r.get('rate_metric'))}</td>"
                     f"<td>{fmt(r.get('rate_mean'), 2 if (r.get('rate_mean') or 0) < 10 else 1)}</td>"
                     f"<td>{fmt((r.get('rate_mean') or None) and 1 / r['rate_mean'], 1) if 'generation' in str(r.get('rate_metric')) or 'items' in str(r.get('rate_metric')) else fmt(None)}</td>"
                     f"<td class=left>{accuracy}</td>"
                     f"<td>{fmt(r.get('disk_gb'))}</td><td>{fmt(r.get('mem_peak_gib'))}</td><td>{fmt(r.get('power_mean_w'))}</td>"
                     f"<td>{fmt(r.get('temp_peak_c'), 0)}</td><td>{fmt((r.get('energy_j') or None) and r['energy_j'] / 1000)}</td>"
                     f"<td class={'ok' if r['status'] == 'ok' else 'bad'}>{esc(r['status'])}</td></tr>")
        h.append("</table></div><div class=note>HF transformers rows are plain PyTorch generate() baselines: no continuous "
                 "batching, no paged KV cache. Read them next to the vLLM benchmark of the same weights (named in the accuracy "
                 "column) for what a serving engine adds on this machine. Their accuracy is that benchmark's: accuracy belongs to "
                 "the weights, not to the engine running them.</div>")

    tables, h[tables_at:] = h[tables_at:], []
    gallery_html, page.gallery_files = gallery(run_dir, rows, out_dir)
    h.append(gallery_html)
    try:  # the use-case sections (new benchmarks): never the reason the page fails
        cases_html, case_files = use_cases(run_dir, rows, out_dir)
    except Exception as exc:  # noqa: BLE001
        cases_html, case_files = f"<p class=sub>The use-case sections could not be drawn ({esc(type(exc).__name__)}: {esc(str(exc)[:200])}).</p>", []
    if cases_html:
        if VEGA not in "".join(h) and "class=vega" in cases_html:  # the LLM charts load Vega; without them, load it here
            cases_html = VEGA + cases_html
        h.append(cases_html)
        page.gallery_files = page.gallery_files + case_files

    skipped = set()
    plan_path = os.path.join(run_dir, "report", "plan.json")
    if os.path.exists(plan_path):
        skipped = {b["bench"] for b in json.load(open(plan_path)) if b.get("skipped")}
    failed = defaultdict(list)
    for r in rows:
        bench = split(r["bench"])[0]
        if r["status"] in ("ok", "running") or bench in skipped or bench in quarantine:
            continue
        failed[model_key(r["bench"])[0]].append(r)
    h.extend(tables)  # LLM serving compare / footprint, pass definitions, other benchmarks: after every chart
    h.append("<h2>Failures</h2>")
    if failed:
        h.append("<ul>")
        for head, rs in sorted(failed.items()):
            kinds = sorted({r["status"] for r in rs})
            cause = server_error(run_dir, rs) or next((r["error"] for r in rs if r.get("error") and not _client_side(r["error"])), None) \
                or next((r["error"] for r in rs if r.get("error")), None)
            parts = sorted({model_key(r["bench"])[1] or "run" for r in rs})
            h.append(f"<li><code>{esc(head.replace('homelab-', ''))}</code>: <span class=bad>{esc(', '.join(kinds))}</span>"
                     f" <span class=sub>({esc(', '.join(parts))})</span>{' — ' + esc(cause[:300]) if cause else ''}</li>")
        h.append("</ul>")
    else:
        h.append("<p class=sub>No failure.</p>")
    evictions = 0
    if os.path.exists(os.path.join(run_dir, "memguard.log")):
        evictions = sum(1 for l in open(os.path.join(run_dir, "memguard.log")) if '"evict"' in l)
    h.append(f"<p class=sub>memguard: {len(kills)} benchmark(s) stopped before the machine ran out of memory, "
             f"{evictions} page-cache eviction(s) when free memory ran low.</p>")
    if quarantine:
        groups = defaultdict(list)
        for bench, e in sorted(quarantine.items()):
            groups[(e.get("kind"), e.get("reason"))].append(bench)
        h.append("<h2>Not run on this machine (quarantine)</h2><ul>")
        for (kind, reason), benches in sorted(groups.items(), key=lambda kv: (kv[0][0] != "hang", kv[1][0])):
            names = ", ".join(b.replace("homelab-", "") for b in benches)
            h.append(f"<li><span class={'bad' if kind == 'hang' else 'sub'}>{esc(kind)}</span> <code>{esc(names)}</code>: {esc(reason)}</li>")
        h.append("</ul><div class=note>hang: the machine froze or rebooted while it ran (power cycle needed); unsupported: "
                 "can't run here. <code>scripts/homelab/quarantine.py remove &lt;bench&gt;</code> lets one run again.</div>")

    if prog:
        h.append(f"<details{' open' if alive else ''}><summary><h2 style='display:inline'>Run plan</h2></summary><div class=wrap><table><tr><th>#</th><th>benchmark</th><th>results</th><th>time</th></tr>")
        for i, e in enumerate(prog["entries"], 1):
            if e.get("skipped"):
                h.append(f"<tr class=skip><td>{i}</td><td>{esc(e['bench'].replace('homelab-', ''))}</td>"
                         f"<td class=plan>skipped: {esc(e['skipped'])}</td><td></td></tr>")
                continue
            chips = "".join(
                f"<span class='chip {'ok' if st == 'ok' else 'bad' if st == 'failed' else 'run' if st == 'running' else ''}'"
                f" title='{esc(st)}'>{esc(label)}</span>" for label, st, _ in e["states"])
            h.append(f"<tr><td>{i}</td><td>{esc(e['bench'].replace('homelab-', ''))}</td><td class=plan><div class=chips>{chips}</div></td>"
                     f"<td>{fmt(e['minutes'], 0, suffix=' min')}</td></tr>")
        h.append("</table></div><div class=note>milabench's run order. Chips: one per result (client pass); "
                 "green passed, red failed, amber running, grey pending.</div></details>")

    h.append("<details><summary><h2 style='display:inline'>Every pass</h2></summary><div class=wrap><table><tr><th>pass</th><th>status</th><th>out tok/s</th><th>tok/s/user</th><th>req/s</th>"
             "<th>TTFT p50</th><th>TTFT p99</th><th>TPOT p50</th><th>TPOT p99</th><th>mem peak GiB</th><th>power W</th>"
             "<th>temp °C</th><th>energy kJ</th><th>tok/J</th><th>duration</th></tr>")
    for r in sorted(rows, key=lambda r: r.get("start") or 0):
        h.append(f"<tr><td>{esc(r['bench'].replace('homelab-', ''))}</td>"
                 f"<td class={'ok' if r['status'] == 'ok' else 'run' if r['status'] == 'running' else 'bad'}>{esc(r['status'])}</td>"
                 f"<td>{fmt(r.get('output_throughput'))}</td><td>{fmt(per_user(r))}</td><td>{fmt(r.get('request_throughput'), 2)}</td>"
                 f"<td>{fmt(r.get('ttft_ms_p50'), 0)}</td><td>{fmt(r.get('ttft_ms_p99'), 0)}</td><td>{fmt(r.get('tpot_ms_p50'))}</td>"
                 f"<td>{fmt(r.get('tpot_ms_p99'))}</td><td>{fmt(r.get('mem_peak_gib'))}</td><td>{fmt(r.get('power_mean_w'))}</td>"
                 f"<td>{fmt(r.get('temp_peak_c'), 0)}</td><td>{fmt((r.get('energy_j') or None) and r['energy_j'] / 1000)}</td>"
                 f"<td>{fmt(r.get('output_tokens_per_joule'), 2)}</td><td>{fmt((r.get('duration_s') or None) and r['duration_s'] / 60, 1, suffix=' min')}</td></tr>")
    h.append("</table></div></details></main>")
    return "\n".join(h)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--rows", default=None, help="report.json (default <run>/report/report.json)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    rows = json.load(open(args.rows or os.path.join(args.run_dir, "report", "report.json")))
    tmp = args.out + ".tmp"
    out_dir = os.path.dirname(os.path.abspath(args.out))
    with open(tmp, "w") as f:
        f.write(page(args.run_dir, rows, out_dir))
    os.replace(tmp, args.out)
    # the files the page links (gallery samples), for publishing it with them
    with open(os.path.splitext(args.out)[0] + ".files.json", "w") as f:
        json.dump(getattr(page, "gallery_files", []), f)
    print(f"{len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
