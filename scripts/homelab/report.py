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
import time
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
                except (json.JSONDecodeError, AttributeError, TypeError):  # TypeError: set to null to turn it off
                    pass
    # single-pack benches: --model is an HF id, or ComfyUI's dest:repo:file (one or a list)
    model = (cfg.get("argv") or {}).get("--model")
    for spec in [model] if isinstance(model, str) else model or []:
        add(spec.split(":")[1] if isinstance(spec, str) and spec.count(":") >= 2 else spec)
    return ids


QUANT_TAGS = {"fp8": "FP8", "nvfp4": "NVFP4", "mxfp4": "MXFP4", "awq": "AWQ 4-bit", "w8a8": "INT8 W8A8",
              "bf16": "BF16", "gguf": "GGUF"}


def describe(cfg: dict) -> dict:
    """What a benchmark runs, in words: engine, precision, workload, and what its rate counts."""
    definition = os.path.basename(cfg.get("definition", ""))
    tags = set(cfg.get("tags") or [])
    argv = cfg.get("argv") or {}
    server = (cfg.get("server") or {}).get("argv") or {}
    client = (cfg.get("client") or {}).get("argv") or {}
    out = {}
    if definition == "serving":
        out["engine"] = {"vllm": "vLLM", "sglang": "SGLang", "atom": "ATOM"}.get((cfg.get("server") or {}).get("backend", "vllm"), "vLLM")
        precision = [QUANT_TAGS[t] for t in sorted(tags) if t in QUANT_TAGS] or [str(server.get("--dtype", "BF16")).upper()]
        if server.get("--quantization") == "fp8":
            precision = ["FP8 (quantized at load)"]
        spec = server.get("--speculative-config")
        out["precision"] = ", ".join(precision)
        out["workload"] = ("real-text prompts" if client.get("--real-text") else "random prompts") + \
            (f", speculative decoding ({json.loads(spec).get('method') or 'draft model'})" if spec else "") + \
            f", context {server.get('--max-model-len', '?')}"
        out["metric"] = "output tokens/s, all streams"
    elif definition == "llamacpp":
        out["engine"] = f"llama.cpp ({server.get('--backend', '?')})"
        out["precision"] = "GGUF " + os.path.basename(str(server.get("--file", ""))).split("-00001")[0].rsplit("-", 1)[-1].replace(".gguf", "")
        out["workload"] = "QualityBench" if server.get("--mode") == "quality" else f"{server.get('--parallel', '?')} slots, real-text prompts"
        out["metric"] = "output tokens/s, all streams"
    elif definition == "inference":
        mode = argv.get("--mode")
        out["engine"] = "HF transformers generate()" if mode == "chat" else f"HF transformers ({mode})"
        out["precision"] = {"chat": "BF16", "whisper": "FP16", "flux": "BF16"}.get(mode, "?")
        out["workload"] = f"{argv.get('--dataset', '?')}, batch {str(argv.get('--batch_size', '?')).replace('auto_batch', 'auto ')}"
        out["metric"] = {"chat": "tokens/s, prompt + generated", "whisper": "audio samples/s"}.get(mode, "items/s")
    elif definition == "comfyui":
        out["engine"] = "ComfyUI"
        out["workload"] = os.path.basename(str(argv.get("--workflow", ""))).replace(".json", "")
        out["metric"] = "generations/s (images, clips or songs)"
    elif definition == "omni":
        out["engine"] = "vLLM-Omni"
        out["precision"] = "FP8 (quantized at load)" if server.get("--quantization") == "fp8" else "BF16"
        out["workload"] = f"{client.get('--task', '?')}, {client.get('--width', '?')}x{client.get('--height', '?')}, {client.get('--steps', '?')} steps"
        out["metric"] = "generations/s"
    elif definition == "unsloth":
        out["engine"] = "unsloth (LoRA fine-tune)"
        out["metric"] = "items/s"
    elif definition == "llm-finetune":
        qlora = argv.get("--load_in_4bit") or "qlora" in tags
        out["engine"] = "unsloth + TRL (" + ("QLoRA, 4-bit base" if qlora else "LoRA") + ")"
        out["workload"] = f"sequences of {argv.get('--seq_length', '?')} tokens, LoRA rank {argv.get('--lora_r', '?')}"
        out["metric"] = "training tokens/s"
    elif definition in ("kohya", "aitoolkit", "simpletuner"):
        out["engine"] = {"kohya": "kohya sd-scripts", "aitoolkit": "ai-toolkit", "simpletuner": "SimpleTuner"}[definition] + " (LoRA)"
        out["precision"] = str(argv.get("--mixed_precision", "?")).upper()
        out["workload"] = f"{argv.get('--steps', '?')} steps, batch {argv.get('--batch_size', '?')}, {argv.get('--resolution', '?')} px"
        out["metric"] = "images/s"
    elif definition in ("retrieval", "docparse", "vlm"):
        out["engine"] = "vLLM" + (" (pooling)" if definition == "retrieval" else "")
        precision = [QUANT_TAGS[t] for t in sorted(tags) if t in QUANT_TAGS]
        if server.get("--quantization") == "fp8":
            precision = ["FP8 (quantized at load)"]
        if precision:
            out["precision"] = ", ".join(precision)
    elif definition == "docparse_pipeline":
        out["engine"] = "classic OCR pipeline (no LLM)"
    elif definition == "tts":
        out["engine"] = "vLLM-Omni"
        out["metric"] = "seconds of audio per second"
    elif definition == "comfyui_edit":
        out["engine"] = "ComfyUI"
        out["workload"] = "image editing, " + os.path.basename(str(argv.get("--workflow", ""))).replace(".json", "")
        out["metric"] = "edits/s"
    elif definition == "omni_edit":
        out["engine"] = "vLLM-Omni"
        out["workload"] = "image editing"
        out["metric"] = "edits/s"
    return out


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
    "kv_cache_gib": re.compile(r"(?:Available KV cache memory: |reserved )([\d.]+) Gi?B(?: memory for KV Cache)?"),
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
                # a crash leaves a half-written last line (NUL bytes): skip what does not parse
                if str(row.get("time_ms", "")).isdigit():
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


TTS_KEYS = ("rtf_x", "ttfa_s", "ttft_s", "audio_seconds", "gen_seconds", "step_audio_x_realtime", "wer", "wer_words")
EVAL_KEYS = ("suite", "task", "thinking", "mode", "tier", "samples", "correct", "errors", "scores", "ci95", "timing",
             "latency", "concurrency", "bm25", "index_docs_per_s", "rerank_pairs_per_s", "corpus_docs", "scorer_error")
FT_KEYS = ("model", "qlora", "seq_length", "tokens_per_step", "lora_r", "trainable_params_m", "steps", "tokens",
           "seconds", "train_tokens_per_s", "step_time_s_p50", "step_time_s_p90", "weights_gib",
           "torch_peak_allocated_gib", "torch_peak_reserved_gib", "system_mem_peak_gib", "loss_start", "loss_end",
           "loss_drop", "loss_ok", "energy_j", "kwh_per_1m_tokens", "tokens_per_joule", "joules_per_token")


def read_bench(path: str, gpu_rows, mem_rows) -> dict | None:
    cfg = meta = None
    start = end = None
    rc = None
    rates, samples = [], []
    errors = []
    passes, evals, agents = [], [], []
    tts, steps, finetune = defaultdict(list), {}, None
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
        elif kind == "line" and isinstance(data, str) and re.search(r"(Error|error:|FAILED|Traceback)", data):
            errors.append(data.strip())
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
            if isinstance(data.get("finetune_summary"), dict):
                finetune = data["finetune_summary"]
            for key in TTS_KEYS:  # text-to-speech: one event per request, one per step, the WER at the end
                if isinstance(data.get(key), (int, float)):
                    tts[key].append(data[key])
            for key in ("steps_per_second", "samples_per_second", "num_steps"):  # diffusion fine-tuning
                if isinstance(data.get(key), (int, float)):
                    steps[key] = data[key]
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
    if rc is None and time.time() - os.path.getmtime(path) < 900:
        row["status"] = "running"
    if errors and rc not in (0, None):
        row["error"] = next((e for e in reversed(errors) if not e.startswith(("File ", "Traceback"))), errors[-1])[:300]
    row["disk_gb"] = round(sum(d for m in row["models"] if (d := disk_gb(m))), 2) or None
    row["what"] = describe(cfg)
    if row["what"].get("metric") and not passes:
        row["rate_metric"] = row["what"]["metric"]

    # throughput
    if passes:
        p = passes[-1]
        row.update({k: p.get(k) for k in ("output_throughput", "total_throughput", "request_throughput",
                                          "ttft_ms_p50", "ttft_ms_p99", "tpot_ms_p50", "tpot_ms_p99", "itl_ms_p50",
                                          "e2el_ms_p50", "completed", "failed", "mean_input_len", "mean_output_len")})
        # the use-case passes' own fields (pooling: docs/s; OCR: pages/min), only when the pass has them
        row.update({k: p[k] for k in ("docs_per_s", "docs_per_request", "e2el_ms_p90", "e2el_ms_p99", "kind",
                                      "pages_per_min", "concurrency", "truncated") if p.get(k) is not None})
        if p.get("duration_s") is not None:
            row["pass_duration_s"] = p["duration_s"]
        row["window"] = (p.get("start_time"), p.get("end_time"))
    elif rates:
        values = [v for _, v, _ in rates]
        row.setdefault("rate_metric", rates[0][0] if rates[0][0] != "rate" else (rates[0][2] or "rate"))
        row["rate_mean"] = statistics.fmean(values)
        row["rate_median"] = statistics.median(values)
    # the use cases' accuracy passes (BFCL extended, retrieval, VQA, OCR, RULER): every summary, compact
    kept = [{k: ev[k] for k in EVAL_KEYS if ev.get(k) is not None} for ev in evals
            if ev.get("suite") not in ("qualitybench",) and isinstance(ev.get("scores"), dict)]
    if kept:
        row["evals"] = kept
    # text-to-speech (the ComfyUI / omni packs push gen_seconds too: only a pass with rtf_x or a WER is speech)
    if not (tts.get("rtf_x") or tts.get("wer")):
        tts = {}
    if tts.get("rtf_x"):
        row["tts.rtf_x"] = statistics.median(tts["rtf_x"])
        row["tts.requests"] = len(tts["rtf_x"])
    if tts.get("ttfa_s"):
        ttfa = sorted(tts["ttfa_s"])
        row["tts.ttfa_s_p50"] = statistics.median(ttfa)
        row["tts.ttfa_s_p90"] = ttfa[min(len(ttfa) - 1, int(round(0.9 * (len(ttfa) - 1))))]
    if tts.get("ttft_s"):
        row["tts.ttft_s_p50"] = statistics.median(tts["ttft_s"])
    if tts.get("audio_seconds"):
        row["tts.audio_seconds"] = round(sum(tts["audio_seconds"]), 2)
    if tts.get("gen_seconds"):
        row["tts.gen_seconds"] = round(sum(tts["gen_seconds"]), 2)
    if tts.get("step_audio_x_realtime"):
        row["tts.x_realtime"] = statistics.fmean(tts["step_audio_x_realtime"])
    if tts.get("wer"):
        row["tts.wer"] = tts["wer"][-1]
        row["tts.wer_words"] = (tts.get("wer_words") or [None])[-1]
    # fine-tuning: LLM (finetune_summary) and diffusion (steps/s, images/s)
    if finetune:
        row.update({f"ft.{k}": finetune.get(k) for k in FT_KEYS if finetune.get(k) is not None})
        if finetune.get("window_start") and finetune.get("window_end"):
            row["window"] = (finetune["window_start"], finetune["window_end"])  # the measured steps only
    if steps:
        row.update({"ft.steps_per_s": steps.get("steps_per_second"), "ft.images_per_s": steps.get("samples_per_second"),
                    "ft.steps": steps.get("num_steps")})
    # accuracy
    for ev in evals:
        task = ev.get("task")
        if task == "logprobs":
            for key, value in (ev.get("scores") or {}).items():
                if key.startswith("ppl") and isinstance(value, (int, float)):
                    row[key.replace("ppl-", "ppl.")] = value
            continue
        if ev.get("suite") == "bfcl":  # per category: in row["evals"] (an acc.bfcl.* key would read as QualityBench's bfcl task)
            continue
        # the mode the run used (its results file is <bench>.qualitybench.thinking-<mode>.json): QualityBench files
        # a single-mode run under "off" whatever the mode (gpt-oss, scored thinking on, came back as "off")
        mode = re.search(r"\.thinking-(on|off)\.json$", str(ev.get("results") or ""))
        thinking = mode.group(1) if mode else ev.get("thinking")
        bench_name = str(row.get("bench", ""))
        if bench_name.endswith((".lcbfull", "-lcbfull")):  # the full tier (lcb-full.yaml; llama.cpp benches have no pass)
            task = f"{task}_full"
        # the quality pass says its tier: LiveCodeBench at the full tier is the 175 problems, wherever it ran
        elif task == "lcb" and ev.get("tier") == "full":
            task = f"{task}_full"
        key = f"acc.{ev.get('suite', 'lmeval')}.{task}" + (f".think-{thinking}" if thinking else "")
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

    # a server that died before its clients ran (killed, failed to start) leaves only its own file: one failed
    # row for its benchmark, so the failure shows in the report
    for path in files:
        first = next(events(path), {})
        cfg = first.get("data") if first.get("event") == "config" else None
        if not cfg or "server" not in (cfg.get("tag") or []):
            continue
        bench = cfg["tag"][0]
        if any(r["bench"] == bench or r["bench"].startswith(bench + ".") for r in rows):
            continue
        start = end = rc = None
        errors = []
        for e in events(path):
            if e.get("event") == "start":
                start = e["data"].get("time")
            elif e.get("event") == "end":
                end, rc = e["data"].get("time"), e["data"].get("return_code")
            elif e.get("event") == "line" and isinstance(e.get("data"), str) and re.search(r"(Error|error:)", e["data"]):
                errors.append(e["data"].strip())
        if rc in (0, None) and time.time() - os.path.getmtime(path) < 900:
            continue  # still starting: its clients come next
        row = {"bench": bench, "file": os.path.basename(path), "models": model_ids(cfg), "what": describe(cfg),
               "tags": sorted(set(cfg.get("tags", []))), "start": start, "end": end, "return_code": rc,
               "status": "failed", "error": ("server: " + errors[-1][:280]) if errors else "server stopped before its clients ran"}
        if start and end and any(start <= k["time"] <= end + 5 for k in kills):
            row["status"] = "killed-oom"
            reason = next(k.get("reason") for k in kills if start <= k["time"] <= end + 5)
            row["error"] = f"memguard: {reason}"
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
                  "| bench | disk GB | out tok/s | tok/s/user | TTFT p50 ms | TPOT p50 ms | mem peak GiB | weights GiB | KV GiB | power W | energy kJ | tok/J |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in serving:
            lines.append("| " + " | ".join([r["bench"], fmt(r.get("disk_gb")), fmt(r.get("output_throughput")),
                                              fmt(r.get("tpot_ms_p50") and 1000 / r["tpot_ms_p50"]),
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
