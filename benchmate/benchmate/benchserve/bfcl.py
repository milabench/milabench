"""``bfcl`` client passes: Berkeley Function Calling Leaderboard (bfcl-eval) against the running server, with
per-category accuracy and the latency of each call.

    passes:
      - name: bfcl
        bfcl:
          categories: [simple_python, multiple, parallel, parallel_multiple, irrelevance,
                       live_simple, live_multiple, live_irrelevance, multi_turn_base]   # default: OFFLINE below
          every: 1          # every k-th item of each category, in file order (1: all)
          threads: 16       # requests in flight (1: single-stream latency)
          thinking: [off]   # one run per value: off / on (chat_template_kwargs.enable_thinking); null: template default
          max_tokens: null  # max_tokens of every request (default: the server's, the rest of the context)

QualityBench's ``bfcl`` task (quality.py) scores 4 single-turn categories (simple_python, multiple, parallel,
parallel_multiple; 400 items) and reports one accuracy. This pass runs the same harness (bfcl-eval 2026.3.23 in
QualityBench's pinned venv, FC mode: the server's --tool-call-parser decodes the calls) on any offline category:
the relevance checks (irrelevance: no call is the right answer, i.e. hallucinated calls), the live (user-contributed)
sets and the multi-turn sets, whose tool APIs (file system, trading, travel, ...) are simulated in-process.
Not offline, refused: web_search (SerpAPI), memory (its vector backend downloads an embedding model).

Events: one ``eval_summary`` per category and thinking mode (accuracy, 95% Wilson interval, and the call
latency: p50 / p90 / mean seconds per request, as BFCL times them on the client; multi-turn categories also per
turn, every step of a turn summed), plus one for BFCL's overall groups (non_live / live / multi_turn / relevance).
Latency is measured at ``threads`` requests in flight: run a threads: 1 pass for the single-stream number.

Server requirements: --enable-auto-tool-choice and the model's --tool-call-parser (vLLM), --jinja (llama-server);
thinking on: the model's --reasoning-parser.
"""

from __future__ import annotations

import glob
import json
import math
import os
import statistics
import subprocess
import time

from benchmate.benchserve import quality, tools

NON_LIVE = ["simple_python", "simple_java", "simple_javascript", "multiple", "parallel", "parallel_multiple",
            "irrelevance"]
LIVE = ["live_simple", "live_multiple", "live_parallel", "live_parallel_multiple", "live_irrelevance",
        "live_relevance"]
MULTI_TURN = ["multi_turn_base", "multi_turn_miss_func", "multi_turn_miss_param", "multi_turn_long_context"]
OFFLINE = NON_LIVE + LIVE + MULTI_TURN
DEFAULT = ["simple_python", "multiple", "parallel", "parallel_multiple", "irrelevance",
           "live_simple", "live_multiple", "live_irrelevance", "multi_turn_base"]
SHIM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bfcl_shim.py")


def _venv_python() -> str:
    return os.path.join(quality._home(), "venvs", "bfcl", "bin", "python")


def _categories(spec: dict) -> list[str]:
    cats = list(spec.get("categories") or DEFAULT)
    bad = [c for c in cats if c not in OFFLINE]
    if bad:
        raise SystemExit(f"bfcl: categories {bad} are not offline-runnable (choose from {OFFLINE})")
    return cats


def prepare(spec: dict):
    """QualityBench and its pinned BFCL environment (bfcl-eval and its data: nothing is downloaded at run time)."""
    _categories(spec)
    tools.ensure("qualitybench")
    subprocess.run(quality._command("setup", "--tasks", "bfcl"), check=True, env=quality._env())


def _data_file(category: str) -> str:
    found = glob.glob(os.path.join(quality._home(), "venvs", "bfcl", "lib", "python*", "site-packages",
                                   "bfcl_eval", "data", f"BFCL_v4_{category}.json"))
    if not found:
        raise FileNotFoundError(f"bfcl: no data for {category} (run the benchmark's prepare first)")
    return found[0]


def pick_ids(categories: list[str], every: int) -> dict[str, list[str]]:
    ids = {}
    for cat in categories:
        with open(_data_file(cat)) as f:
            all_ids = [json.loads(line)["id"] for line in f if line.strip()]
        ids[cat] = all_ids[::max(1, int(every))]
    return ids


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def _flat(x) -> list[float]:
    if isinstance(x, (int, float)):
        return [float(x)]
    out = []
    for y in x or []:
        out += _flat(y)
    return out


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    return round(s[min(len(s) - 1, int(q * len(s)))], 4)


def latency_stats(results: list[dict]) -> dict:
    """Per-request latency (s) of BFCL's result entries; multi-turn entries hold [turn][step] lists."""
    calls, turns, out_tokens = [], [], []
    for r in results:
        lat = r.get("latency")
        calls += _flat(lat)
        if isinstance(lat, list):
            turns += [sum(_flat(t)) for t in lat]
        out_tokens += _flat(r.get("output_token_count"))
    stats = {"requests": len(calls), "p50_s": round(statistics.median(calls), 4) if calls else None, "p90_s": _pct(calls, 0.9),
             "mean_s": round(statistics.fmean(calls), 4) if calls else None,
             "output_tokens_mean": round(statistics.fmean(out_tokens), 1) if out_tokens else None}
    if turns:
        stats.update({"turns": len(turns), "turn_p50_s": _pct(turns, 0.5), "turn_p90_s": _pct(turns, 0.9)})
    return stats


def _load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def summarize(root: str, categories: list[str], *, name: str, thinking, push) -> list[dict]:
    out, acc = [], {}
    for cat in categories:
        scores = glob.glob(os.path.join(root, "score", "**", f"BFCL_v4_{cat}_score.json"), recursive=True)
        results = glob.glob(os.path.join(root, "result", "**", f"BFCL_v4_{cat}_result.json"), recursive=True)
        if not scores:
            print(f"[bfcl] {cat}: no score file (generation cut or failed)", flush=True)
            continue
        head = _load_jsonl(scores[0])[0]
        k, n = int(head.get("correct_count", 0)), int(head.get("total_count", 0))
        acc[cat] = (k, n)
        summary = {"name": name, "suite": "bfcl", "task": f"bfcl:{cat}", "thinking": thinking, "correct": k,
                   "samples": n, "scores": {"acc": (k / n) if n else 0.0}, "ci95": wilson(k, n),
                   "latency": latency_stats(_load_jsonl(results[0])) if results else {}, "results": root}
        print("[eval]", json.dumps(summary), flush=True)
        push(eval_summary=summary)
        out.append(summary)

    # BFCL's groups: unweighted mean of the categories present (the leaderboard weighs live by size; both kept)
    def mean(cats):
        vals = [acc[c][0] / acc[c][1] for c in cats if c in acc and acc[c][1]]
        return round(statistics.fmean(vals), 4) if vals else None

    def pooled(cats):
        k = sum(acc[c][0] for c in cats if c in acc)
        n = sum(acc[c][1] for c in cats if c in acc)
        return round(k / n, 4) if n else None

    groups = {
        "non_live": mean([c for c in NON_LIVE if c != "irrelevance"]),
        "live": pooled([c for c in LIVE if c not in ("live_irrelevance", "live_relevance")]),
        "multi_turn": mean(MULTI_TURN),
        "irrelevance": mean(["irrelevance", "live_irrelevance"]),
        "relevance": mean(["live_relevance"]),
    }
    groups = {k: v for k, v in groups.items() if v is not None}
    if groups:
        groups["overall"] = round(statistics.fmean(groups.values()), 4)
        summary = {"name": name, "suite": "bfcl", "task": "bfcl:overall", "thinking": thinking, "scores": groups,
                   "results": root}
        print("[eval]", json.dumps(summary), flush=True)
        push(eval_summary=summary)
        out.append(summary)
    return out


def _deadline_s() -> float | None:
    """Seconds left for generation (QualityBench's deadline rule: the benchmark's max_duration, minus a margin
    for scoring)."""
    at = quality.default_deadline(margin=600)
    if at is None:
        return None
    import datetime

    return max(datetime.datetime.fromisoformat(at).timestamp() - time.time(), 60)


def run(bench_argv, spec: dict, *, flag, output_dir: str, name: str, push) -> list[dict]:
    host = flag(bench_argv, "--host", "127.0.0.1")
    port = flag(bench_argv, "--port", "8000")
    model = flag(bench_argv, "--served-model-name") or flag(bench_argv, "--model")
    categories = _categories(spec)
    ids = pick_ids(categories, spec.get("every", 1))
    py = _venv_python()
    if not os.path.exists(py):
        raise FileNotFoundError(f"bfcl: {py} missing: run the benchmark's prepare first")

    thinking = spec.get("thinking", ["off"])
    thinking = [thinking] if isinstance(thinking, (str, bool)) or thinking is None else list(thinking)
    thinking = [("on" if m else "off") if isinstance(m, bool) else m for m in thinking]
    summaries = []
    for mode in thinking:
        root = os.path.join(output_dir, f"{name}.bfcl" + (f".thinking-{mode}" if mode else ""))
        os.makedirs(root, exist_ok=True)
        with open(os.path.join(root, "test_case_ids_to_generate.json"), "w") as f:
            json.dump(ids, f, indent=1)
        env = {**os.environ, "BFCL_PROJECT_ROOT": root, "OPENAI_BASE_URL": f"http://{host}:{port}/v1",
               "OPENAI_API_KEY": "local", "HF_HUB_OFFLINE": "1", "PYTHONUNBUFFERED": "1",
               "TOKENIZERS_PARALLELISM": "false"}
        env.pop("BFCL_THINKING", None)
        if mode:
            env["BFCL_THINKING"] = mode
        if spec.get("max_tokens"):
            env["BFCL_MAX_TOKENS"] = str(int(spec["max_tokens"]))
        gen = [py, SHIM, model, "generate", "--run-ids", "--temperature", "0",
               "--num-threads", str(int(spec.get("threads", 16)))]
        print("BFCL:", " ".join(gen), {k: v for k, v in env.items() if k.startswith("BFCL_")}, flush=True)
        t0 = time.time()
        try:
            # generation resumes: a rerun only asks for the items without a result
            subprocess.run(gen, env=env, timeout=_deadline_s())
        except subprocess.TimeoutExpired:
            print("[bfcl] deadline reached: scoring the items generated so far", flush=True)
        wall = time.time() - t0
        subprocess.run([py, SHIM, model, "evaluate", "--test-category", ",".join(categories), "--partial-eval"],
                       env=env)
        got = summarize(root, categories, name=name, thinking=mode, push=push)
        push(bfcl_pass={"name": name, "thinking": mode, "wall_s": round(wall, 1),
                        "items": sum(map(len, ids.values())), "threads": int(spec.get("threads", 16))})
        summaries += got
    return summaries
