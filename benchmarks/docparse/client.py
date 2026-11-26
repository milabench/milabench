"""Page -> text against the running server, with each model's own recipe (prompt, render, sampling, output).

The recipe is the bench's ``docparse`` config section (read from $MILABENCH_CONFIG):

    docparse:
      adapter: chat            # chat: one OpenAI chat request per page (the default)
                               # mineru | paddleocr_vl: the vendor's client drives the server (sdk_run.py,
                               #   docparse-sdk venv): MinerU's two-step layout + crops, PaddleOCR-VL's layout
                               #   model (CPU) + one request per region
      prompt: "..."            # the model card's page prompt ("" sends the image alone)
      prompt_from: {spec: "chandra-ocr==0.2.0", expr: "..."}   # or the prompt from a pinned package (prepare)
      system: "..."            # optional system message
      text_first: false        # the prompt before the image (default: image first)
      output: markdown         # how convert.py reads the reply (markdown, olmocr, dots_json, doctags, html, ...)
      render: {dpi: 200, long_edge: 1288, max_pixels: ...}      # pages.render
      max_tokens: 8192
      temperature: 0.0
      top_p: 1.0
      extra_body: {...}        # vLLM sampling extras (repetition_penalty, skip_special_tokens, vllm_xargs,
                               # chat_template_kwargs, mm_processor_kwargs, ...)

Client passes (``client.passes[].docparse``):

    {mode: speed, concurrency: 1|8, per_group: n}
                                           the fixed 56-page set (pages.PERF; per_group: its first n pages of
                                           each of the 5 groups, the same pages for every model), after ``concurrency`` warmup
                                           pages; one ``pass_summary`` (pages/min, page latency, output tok/s;
                                           its start/end window gives the pass' energy and peak memory)
    {mode: quality, tier: quick, tasks: [...], concurrency: 8}
                                           every page of each task, then its scorer (scoring.py, outside the
                                           timed window); one ``eval_summary`` per task (suite ``docparse``)

Pages are rendered (and base64-encoded) before the clock starts: the timed window is the model's work, as a
server sees it. Replies are kept raw (<runs>/<bench>.<pass>.docparse/<task>.answers.jsonl).
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import statistics
import subprocess
import tempfile
import time

import pages
import scoring
import toolenv

HERE = os.path.dirname(os.path.abspath(__file__))


def recipe() -> dict:
    cfg = json.loads(os.environ.get("MILABENCH_CONFIG") or "{}")
    return cfg.get("docparse") or {}


def prompt_of(rec: dict) -> str:
    if "prompt_from" in rec:
        path = prompt_file(rec)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path}: run the benchmark's prepare first (prompt_from)")
        return open(path).read()
    return rec.get("prompt", "")


def prompt_file(rec: dict) -> str:
    import hashlib

    src = rec["prompt_from"]
    key = hashlib.sha256(json.dumps(src, sort_keys=True).encode()).hexdigest()[:16]
    return os.path.join(pages.root(), "prompts", f"{src['module']}.{key}.txt")


def resolve_prompt(rec: dict) -> None:
    """prepare: ``prompt_from: {spec, module, expr}``: ``expr`` evaluated in ``module``'s namespace, in the
    docparse-sdk venv (where toolenv installs the pinned ``spec``)."""
    if "prompt_from" not in rec:
        return
    src = rec["prompt_from"]
    code = ("import importlib, sys; m = importlib.import_module(sys.argv[1]); "
            "sys.stdout.write(eval(sys.argv[2], vars(m)))")
    text = subprocess.run([toolenv.python("docparse-sdk"), "-c", code, src["module"], src["expr"]],
                          check=True, capture_output=True, text=True).stdout
    os.makedirs(os.path.dirname(prompt_file(rec)), exist_ok=True)
    with open(prompt_file(rec), "w") as f:
        f.write(text)


def _body(rec: dict, model: str, image_b64: str, prompt: str) -> dict:
    image = {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
    content = [image]
    if prompt:
        content = [{"type": "text", "text": prompt}, image] if rec.get("text_first") else [image, {"type": "text", "text": prompt}]
    messages = [{"role": "user", "content": content}]
    if rec.get("system"):
        messages.insert(0, {"role": "system", "content": rec["system"]})
    body = {"model": model, "messages": messages, "max_tokens": int(rec.get("max_tokens", 8192)),
            "temperature": float(rec.get("temperature", 0.0)), "top_p": float(rec.get("top_p", 1.0)),
            "stream": True, "stream_options": {"include_usage": True}}
    body.update(rec.get("extra_body") or {})
    return body


async def _ask(session, url: str, body: dict) -> dict:
    start = time.perf_counter()
    first, content, usage, error, finish = None, [], {}, None, None
    try:
        async with session.post(url, json=body) as resp:
            if resp.status != 200:
                error = f"HTTP {resp.status}: {(await resp.text())[:300]}"
            else:
                async for raw in resp.content:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    chunk = json.loads(payload)
                    usage = chunk.get("usage") or usage
                    for choice in chunk.get("choices") or []:
                        delta = choice.get("delta") or {}
                        text = delta.get("content")
                        if text and first is None:
                            first = time.perf_counter()
                        if text:
                            content.append(text)
                        finish = choice.get("finish_reason") or finish
    except Exception as exc:  # noqa: BLE001 - one failed page is one empty page, the pass goes on
        error = f"{type(exc).__name__}: {exc}"
    end = time.perf_counter()
    return {"reply": "".join(content), "error": error, "finish_reason": finish,
            "ttft_s": (first - start) if first else None, "e2el_s": end - start,
            "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens")}


async def _chat_run(base_url: str, model: str, rec: dict, images: list[str], concurrency: int, warmup: int):
    """(results, wall seconds, start epoch, end epoch): the warmup requests first, unmeasured."""
    import aiohttp

    prompt = prompt_of(rec)
    url = f"{base_url}/v1/chat/completions"
    semaphore = asyncio.Semaphore(concurrency)
    timeout = aiohttp.ClientTimeout(total=float(rec.get("request_timeout", 1800)))
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async def one(image):
            async with semaphore:
                return await _ask(session, url, _body(rec, model, image, prompt))

        if warmup:
            await asyncio.gather(*(one(images[i % len(images)]) for i in range(warmup)))
        t0, start = time.time(), time.perf_counter()
        results = await asyncio.gather(*(one(image) for image in images))
        wall = time.perf_counter() - start
    return results, wall, t0, t0 + wall


def _sdk_run(base_url: str, model: str, rec: dict, imgs: list, concurrency: int, warmup: int, out_dir: str):
    """The vendor client in the docparse-sdk venv; it times itself (sdk_run.py)."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for i, image in enumerate(imgs):
            path = os.path.join(tmp, f"{i:05d}.png")
            image.save(path)
            paths.append(path)
        out = os.path.join(tmp, "out.jsonl")
        proc = subprocess.run([toolenv.python("docparse-sdk"), "-I", os.path.join(HERE, "sdk_run.py"),
                               "--adapter", rec["adapter"], "--base-url", base_url, "--model", model,
                               "--concurrency", str(concurrency), "--warmup", str(warmup), "--out", out,
                               "--options", json.dumps(rec.get("sdk_options") or {}), *paths],
                              capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"sdk_run {rec['adapter']} failed: {proc.stderr[-2000:]}")
        summary = json.loads(proc.stdout.strip().splitlines()[-1])
        with open(out) as f:
            results = [json.loads(line) for line in f]
    return results, summary["wall_s"], summary["start_time"], summary["end_time"]


def answer(base_url: str, model: str, rec: dict, items: list[dict], concurrency: int, warmup: int, out_dir: str):
    images = [pages.render(item, rec.get("render")) for item in items]
    if rec.get("adapter", "chat") == "chat":
        encoded = [base64.b64encode(pages.png_bytes(img)).decode() for img in images]
        return asyncio.run(_chat_run(base_url, model, rec, encoded, concurrency, warmup))
    return _sdk_run(base_url, model, rec, images, concurrency, warmup, out_dir)


def _pct(values, q):
    values = sorted(v for v in values if v is not None)
    return values[min(len(values) - 1, int(round(q * (len(values) - 1))))] if values else None


def save(path: str, items: list[dict], results: list[dict], rec: dict) -> None:
    with open(path, "w") as f:
        for item, res in zip(items, results):
            f.write(json.dumps({"id": item["id"], "task": item["task"], "output": rec.get("output", "markdown"),
                                **res}, ensure_ascii=False) + "\n")


def timing(items, results, wall) -> dict:
    ok = [r for r in results if not r.get("error")]
    tokens = [r["completion_tokens"] for r in ok if r.get("completion_tokens")]
    return {"pages": len(items), "completed": len(ok), "failed": len(results) - len(ok), "duration_s": wall,
            "pages_per_min": 60 * len(ok) / wall if wall else None,
            "request_throughput": len(ok) / wall if wall else 0.0,
            "output_throughput": sum(tokens) / wall if wall and tokens else 0.0,
            "mean_output_len": statistics.fmean(tokens) if tokens else 0.0,
            "e2el_ms_p50": 1e3 * _pct([r["e2el_s"] for r in ok], 0.5) if ok else None,
            "e2el_ms_p90": 1e3 * _pct([r["e2el_s"] for r in ok], 0.9) if ok else None,
            "ttft_ms_p50": 1e3 * _pct([r.get("ttft_s") for r in ok], 0.5) if any(r.get("ttft_s") for r in ok) else None,
            "truncated": sum(1 for r in ok if r.get("finish_reason") == "length")}


def run(base_url: str, model: str, spec: dict, *, output_dir: str, name: str, pass_name: str, push) -> None:
    rec = recipe()
    os.makedirs(output_dir, exist_ok=True)
    concurrency = int(spec.get("concurrency", 8))
    if spec.get("mode", "quality") == "speed":
        items = pages.load("perf")
        if spec.get("per_group"):  # the first n pages of each group (scans, tables, formulas, ...): a shorter pass
            kept, seen = [], {}
            for item in items:
                seen[item["perf_group"]] = seen.get(item["perf_group"], 0) + 1
                if seen[item["perf_group"]] <= int(spec["per_group"]):
                    kept.append(item)
            items = kept
        results, wall, t0, t1 = answer(base_url, model, rec, items, concurrency, concurrency, output_dir)
        save(os.path.join(output_dir, "perf.answers.jsonl"), items, results, rec)
        summary = {"name": pass_name, "start_time": t0, "end_time": t1, "concurrency": concurrency,
                   **timing(items, results, wall)}
        groups = {}
        for item, res in zip(items, results):
            groups.setdefault(item.get("perf_group", "other"), []).append(res["e2el_s"])
        summary["e2el_s_mean_by_group"] = {g: statistics.fmean(v) for g, v in groups.items()}
        print("[pass]", json.dumps(summary), flush=True)
        push(pass_summary=summary)
        push(rate=summary["pages_per_min"] or 0.0, units="pages/min", task="train")
        if summary["completed"] == 0:
            raise RuntimeError(f"docparse speed pass: every page failed ({results[0].get('error')})")
        return
    tier = spec.get("tier", "quick")
    for task in spec.get("tasks") or pages.TASKS:
        items = pages.load(task, tier)
        if spec.get("limit"):
            items = items[: int(spec["limit"])]
        print(f"[docparse] {task} {tier}: {len(items)} pages, {concurrency} in flight", flush=True)
        results, wall, _, _ = answer(base_url, model, rec, items, concurrency, 0, output_dir)
        answers = os.path.join(output_dir, f"{task}.{tier}.answers.jsonl")
        save(answers, items, results, rec)
        t = timing(items, results, wall)
        try:
            scores = scoring.score(task, items, answers, tier, output_dir)
            error = None
        except Exception as exc:  # noqa: BLE001 - a scorer failure is reported, the other tasks still run
            scores, error = {"acc": None}, f"{type(exc).__name__}: {exc}"
        summary = {"name": name, "suite": "docparse", "task": task, "tier": tier, "samples": len(items),
                   "errors": t["failed"], "scores": {k: v for k, v in scores.items() if k not in ("ci95",)},
                   "ci95": scores.get("ci95") or [], "concurrency": concurrency, "timing": t,
                   "results": answers, "scorer_error": error}
        print("[eval]", json.dumps(summary), flush=True)
        push(eval_summary=summary)
        if t["completed"] == 0:
            raise RuntimeError(f"docparse {task}: every page failed ({results[0].get('error')})")
