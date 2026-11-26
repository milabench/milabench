"""Image understanding accuracy and latency on pinned slices of standard VQA sets.

Tasks (each a pinned revision of its Hugging Face dataset, items picked deterministically):

    docvqa    lmms-lab/DocVQA, validation    document pages: ANLS (the DocVQA metric, threshold 0.5)
    chartqa   HuggingFaceM4/ChartQA, test    charts: relaxed accuracy (numbers within 5%, else exact)
    ocrbench  echo840/OCRBench, test         OCR (text recognition, scene text, KIE, formulas): the official
                                             containment match (score / 1000 -> acc)
    mmmu      MMMU/MMMU, validation          college-level multi-discipline questions, single-image items
                                             only: option letter (multiple choice) or short answer

Tiers: ``quick`` = every k-th item of the task's ordered list (~100 per task: docvqa from its first validation
shard only, 115 MB), ``full`` = every item (docvqa's six validation shards, 1.1 GB).

Data layout ($MILABENCH_DIR_DATA/vlm): <task>/<tier>.jsonl (one item per line: id, image, prompt, answers, ...)
and <task>/images/<id>.<ext> (the original image bytes, no resizing: the model's own processor sizes them).
Two request sets for vLLM's serving benchmark (``--dataset-name custom_image``, the throughput passes):

    perf-doc.jsonl        docvqa quick pages + their question (short answers: prefill-bound)
    perf-describe.jsonl   chartqa / ocrbench / mmmu quick images, "Describe this image in detail." (decode-bound)

The client (``run``): OpenAI chat completions, streaming, greedy, N requests in flight; per request the time to
first token (reasoning or content), the total time and the server's prompt token count (text + image tokens).
One ``eval_summary`` event per task (suite ``vqa``: accuracy, 95% CI, latency and prefill figures), the
answers in <runs>/<bench>.vqa/<task>.jsonl.
"""

from __future__ import annotations

import ast
import asyncio
import base64
import json
import math
import os
import re
import statistics
import time

# ---------------------------------------------------------------------------
# Pinned sources
# ---------------------------------------------------------------------------

SOURCES = {
    "docvqa": {
        "repo": "lmms-lab/DocVQA",
        "revision": "539088ef8a8ada01ac8e2e6d4e372586748a265e",
        "files": {"quick": ["DocVQA/validation-00000-of-00006.parquet"],
                  "full": [f"DocVQA/validation-0000{i}-of-00006.parquet" for i in range(6)]},
        "quick": 100,
    },
    "chartqa": {
        "repo": "HuggingFaceM4/ChartQA",
        "revision": "b605b6e08b57faf4359aeb2fe6a3ca595f99b6c5",
        "files": {"quick": ["data/test-00000-of-00001-e2cd0b7a0f9eb20d.parquet"]},
        "quick": 100,
    },
    "ocrbench": {
        "repo": "echo840/OCRBench",
        "revision": "92a54bd1384387c178d5a07140a2d85e0a3d12e1",
        "files": {"quick": ["data/test-00000-of-00001.parquet"]},
        "quick": 100,
    },
    "mmmu": {
        "repo": "MMMU/MMMU",
        "revision": "876ce5cb130f7f7e290ce4d9984357737d4db5cf",
        # the 30 subjects' validation files, listed at that revision
        "files": {"quick": "*/validation-*.parquet"},
        "quick": 90,
    },
}

TASKS = list(SOURCES)

SHORT = "Answer the question using a single word or phrase."
MC = "Answer with the option's letter from the given choices directly."
DESCRIBE = "Describe this image in detail."


def data_root() -> str:
    return os.path.join(os.environ.get("MILABENCH_DIR_DATA", "."), "vlm")


def data_path(path: str) -> str:
    """A path relative to the vlm data folder (``vlm/perf-doc.jsonl`` or ``perf-doc.jsonl``) made absolute."""
    if os.path.isabs(path):
        return path
    return os.path.join(data_root(), path.removeprefix("vlm/"))


# ---------------------------------------------------------------------------
# Building the slices (prepare: the runs are offline)
# ---------------------------------------------------------------------------


def _files(task: str, tier: str) -> list[str]:
    src = SOURCES[task]
    files = src["files"].get(tier, src["files"]["quick"])
    if isinstance(files, str):
        import fnmatch

        from huggingface_hub import HfApi

        listing = HfApi().list_repo_files(src["repo"], repo_type="dataset", revision=src["revision"])
        files = sorted(f for f in listing if fnmatch.fnmatch(f, files))
    return files


def _rows(task: str, tier: str):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    src = SOURCES[task]
    for name in _files(task, tier):
        path = hf_hub_download(src["repo"], name, repo_type="dataset", revision=src["revision"])
        table = pq.read_table(path)
        for row in table.to_pylist():
            yield row


def _ext(data: bytes) -> str:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:3] == b"\xff\xd8\xff":
        return "jpg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    if data[:3] == b"GIF":
        return "gif"
    return "png"


def _listify(value):
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, (list, tuple)):
                return [str(v) for v in parsed]
        except (ValueError, SyntaxError):
            pass
        return [value]
    return [str(v) for v in value or []]


def _item(task: str, index: int, row: dict) -> dict | None:
    """The item of a dataset row: id, image bytes, prompt, answers (None: not used, e.g. multi-image MMMU)."""
    if task == "docvqa":
        return {"id": f"docvqa-{row['questionId']}", "image": row["image"]["bytes"],
                "prompt": f"{row['question']}\n{SHORT}", "answers": _listify(row["answers"]),
                "kind": ",".join(_listify(row.get("question_types")))}
    if task == "chartqa":
        return {"id": f"chartqa-{index}", "image": row["image"]["bytes"],
                "prompt": f"{row['query']}\n{SHORT}", "answers": _listify(row["label"]),
                "kind": "human" if str(row.get("human_or_machine")) == "0" else "augmented"}
    if task == "ocrbench":
        return {"id": f"ocrbench-{index}", "image": row["image"]["bytes"], "prompt": row["question"],
                "answers": _listify(row["answer"]), "kind": row["question_type"], "source": row["dataset"]}
    if task == "mmmu":
        if row.get("image_2") or not row.get("image_1"):
            return None  # single-image questions only (the servers allow one image per request)
        question = re.sub(r"<image \d+>", "", row["question"]).strip()
        options = _listify(row["options"])
        if row["question_type"] == "multiple-choice" and options:
            letters = "ABCDEFGHIJ"
            listed = "\n".join(f"{letters[i]}. {opt}" for i, opt in enumerate(options))
            prompt, kind = f"{question}\n{listed}\n{MC}", "multiple-choice"
        else:
            prompt, kind = f"{question}\n{SHORT}", "open"
        return {"id": row["id"], "image": row["image_1"]["bytes"], "prompt": prompt,
                "answers": _listify(row["answer"]), "options": options, "kind": kind,
                "subject": row["id"].split("_", 1)[1].rsplit("_", 1)[0]}
    raise ValueError(task)


def _pick(items: list[dict], task: str, tier: str) -> list[dict]:
    if tier == "full":
        return items
    n = SOURCES[task]["quick"]
    if task == "mmmu":
        # the same number per subject: the first items of each
        per = max(1, n // 30)
        out, seen = [], {}
        for item in items:
            if seen.get(item["subject"], 0) < per:
                seen[item["subject"]] = seen.get(item["subject"], 0) + 1
                out.append(item)
        return out
    step = max(1, len(items) // n)
    return items[::step][:n]


def build(task: str, tier: str = "quick") -> str:
    """Write <task>/<tier>.jsonl and its images (once); returns the jsonl path."""
    folder = os.path.join(data_root(), task)
    out = os.path.join(folder, f"{tier}.jsonl")
    if os.path.exists(out):
        return out
    os.makedirs(os.path.join(folder, "images"), exist_ok=True)
    items = [it for i, row in enumerate(_rows(task, tier)) if (it := _item(task, i, row)) is not None]
    picked = _pick(items, task, tier)
    tmp = out + ".tmp"
    with open(tmp, "w") as f:
        for item in picked:
            data = item.pop("image")
            image = os.path.join(folder, "images", f"{item['id']}.{_ext(data)}")
            if not os.path.exists(image):
                with open(image, "wb") as img:
                    img.write(data)
            item["image"] = image
            f.write(json.dumps(item) + "\n")
    os.replace(tmp, out)
    print(f"[vqa] {task} {tier}: {len(picked)} items -> {out}", flush=True)
    return out


def load(task: str, tier: str) -> list[dict]:
    with open(os.path.join(data_root(), task, f"{tier}.jsonl")) as f:
        return [json.loads(line) for line in f if line.strip()]


def build_perf_sets() -> None:
    """The request sets of the throughput passes (vLLM ``custom_image``), from the quick slices."""
    root = data_root()
    sets = {
        "perf-doc.jsonl": [(it["prompt"], it["image"]) for it in load("docvqa", "quick")],
        "perf-describe.jsonl": [(DESCRIBE, it["image"]) for task in ("chartqa", "ocrbench", "mmmu")
                                for it in load(task, "quick")],
    }
    for name, rows in sets.items():
        with open(os.path.join(root, name), "w") as f:
            for prompt, image in rows:
                f.write(json.dumps({"prompt": prompt, "image_files": [image]}) + "\n")
        print(f"[vqa] {name}: {len(rows)} requests", flush=True)


def prepare(specs: list[dict], perf: bool = True) -> None:
    """Every slice the passes read (and the quick ones the throughput sets are built from)."""
    for task in TASKS:
        build(task, "quick")
    for spec in specs:
        tier = spec.get("tier", "quick")
        for task in spec.get("tasks") or TASKS:
            build(task, tier)
    if perf:
        build_perf_sets()


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _clean(text: str) -> str:
    """The answer part of a reply: no reasoning block, no "Answer:" lead, no trailing period."""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    text = text.strip()
    m = re.search(r"(?:final answer|answer)\s*[:：]\s*(.+)", text, re.I | re.S)
    if m:
        text = m.group(1).strip()
    text = text.strip().strip("*").strip()
    return text.rstrip(".").strip()


def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def anls(pred: str, answers: list[str], threshold: float = 0.5) -> float:
    p = " ".join(_clean(pred).lower().split())
    best = 0.0
    for answer in answers:
        a = " ".join(answer.lower().split())
        if not a and not p:
            return 1.0
        nld = _levenshtein(p, a) / max(len(p), len(a), 1)
        best = max(best, 1 - nld if nld < threshold else 0.0)
    return best


def _number(text: str):
    t = text.strip().replace(",", "").replace("$", "").rstrip("%").strip()
    try:
        return float(t)
    except ValueError:
        return None


def relaxed(pred: str, answers: list[str], tolerance: float = 0.05) -> float:
    p = _clean(pred)
    for answer in answers:
        pn, an = _number(p), _number(answer)
        if an is not None and pn is not None:
            if (an == 0 and pn == 0) or (an != 0 and abs(pn - an) / abs(an) <= tolerance):
                return 1.0
        elif p.lower() == answer.strip().lower():
            return 1.0
    return 0.0


def ocrbench(pred: str, answers: list[str], source: str) -> float:
    """OCRBench's official match: the answer is contained in the reply (formulas: spaces removed)."""
    reply = pred.rsplit("</think>", 1)[-1]
    for answer in answers:
        if source == "HME100k":
            if answer.strip().replace("\n", " ").replace(" ", "") in reply.strip().replace("\n", " ").replace(" ", ""):
                return 1.0
        elif answer.lower().strip().replace("\n", " ") in reply.lower().strip().replace("\n", " "):
            return 1.0
    return 0.0


def mmmu(pred: str, item: dict) -> float:
    reply = _clean(pred)
    gold = item["answers"][0] if item["answers"] else ""
    if item["kind"] == "multiple-choice":
        letters = "ABCDEFGHIJ"[: len(item["options"])]
        m = (re.match(rf"^\(?([{letters}])\)?(?:[.:)\s]|$)", reply)
             or re.search(rf"(?:answer|option) is\s*\(?([{letters}])\)?\b", reply, re.I)
             or re.search(rf"\b([{letters}])\b", reply))
        if m:
            return float(m.group(1) == gold)
        # the option's text instead of its letter
        for letter, option in zip(letters, item["options"]):
            if option and option.strip().lower() == reply.lower():
                return float(letter == gold)
        return 0.0
    for answer in item["answers"]:
        pn, an = _number(reply), _number(answer)
        if pn is not None and an is not None and math.isclose(pn, an, rel_tol=1e-2, abs_tol=1e-6):
            return 1.0
        if answer.strip().lower() and answer.strip().lower() in reply.lower():
            return 1.0
    return 0.0


def score(task: str, item: dict, reply: str) -> float:
    if task == "docvqa":
        return anls(reply, item["answers"])
    if task == "chartqa":
        return relaxed(reply, item["answers"])
    if task == "ocrbench":
        return ocrbench(reply, item["answers"], item.get("source", ""))
    if task == "mmmu":
        return mmmu(reply, item)
    raise ValueError(task)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


def _data_url(path: str) -> str:
    ext = os.path.splitext(path)[1].lstrip(".").lower()
    mime = {"jpg": "jpeg"}.get(ext, ext)
    with open(path, "rb") as f:
        return f"data:image/{mime};base64,{base64.b64encode(f.read()).decode()}"


async def _ask(session, url: str, model: str, item: dict, spec: dict) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": _data_url(item["image"])}},
            {"type": "text", "text": item["prompt"]},
        ]}],
        "max_tokens": int(spec.get("max_tokens", 256)),
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if spec.get("chat_template_kwargs"):
        body["chat_template_kwargs"] = spec["chat_template_kwargs"]
    body.update(spec.get("extra_body") or {})
    start = time.perf_counter()
    first, content, reasoning, usage, error = None, [], [], {}, None
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
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    for choice in chunk.get("choices") or []:
                        delta = choice.get("delta") or {}
                        text = delta.get("content")
                        think = delta.get("reasoning_content") or delta.get("reasoning")
                        if (text or think) and first is None:
                            first = time.perf_counter()
                        if text:
                            content.append(text)
                        if think:
                            reasoning.append(think)
    except Exception as exc:  # noqa: BLE001 - one failed request is one wrong answer, the pass goes on
        error = f"{type(exc).__name__}: {exc}"
    end = time.perf_counter()
    return {
        "reply": "".join(content), "reasoning_chars": len("".join(reasoning)), "error": error,
        "ttft_s": (first - start) if first else None, "e2el_s": end - start,
        "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
    }


async def _run_task(base_url, model, items, spec, out_path):
    import aiohttp

    concurrency = int(spec.get("concurrency", 8))
    semaphore = asyncio.Semaphore(concurrency)
    timeout = aiohttp.ClientTimeout(total=float(spec.get("request_timeout", 1800)))
    results = [None] * len(items)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async def one(i, item):
            async with semaphore:
                results[i] = await _ask(session, f"{base_url}/v1/chat/completions", model, item, spec)

        start = time.perf_counter()
        await asyncio.gather(*(one(i, item) for i, item in enumerate(items)))
        wall = time.perf_counter() - start
    with open(out_path, "w") as f:
        for item, res in zip(items, results):
            f.write(json.dumps({"id": item["id"], "kind": item.get("kind"), "answers": item["answers"], **res}) + "\n")
    return results, wall


def _pct(values, q):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    return values[min(len(values) - 1, int(round(q * (len(values) - 1))))]


def _ci95(scores: list[float]) -> list[float]:
    n = len(scores)
    if n == 0:
        return []
    mean = sum(scores) / n
    if all(s in (0.0, 1.0) for s in scores):  # Wilson interval of a proportion
        z = 1.96
        centre = (mean + z * z / (2 * n)) / (1 + z * z / n)
        half = z * math.sqrt(mean * (1 - mean) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        return [max(0.0, centre - half), min(1.0, centre + half)]
    sd = statistics.pstdev(scores) if n > 1 else 0.0
    return [max(0.0, mean - 1.96 * sd / math.sqrt(n)), min(1.0, mean + 1.96 * sd / math.sqrt(n))]


def summarize(task: str, items: list[dict], results: list[dict], wall: float, *, name: str, spec: dict,
              path: str) -> dict:
    scores = [score(task, it, r["reply"]) if not r["error"] else 0.0 for it, r in zip(items, results)]
    ok = [r for r in results if not r["error"]]
    ttft = [r["ttft_s"] for r in ok if r["ttft_s"]]
    prompt = [r["prompt_tokens"] for r in ok if r["prompt_tokens"]]
    # prefill rate of one request: its prompt (text + image tokens) over its time to first token; at concurrency
    # 1 this is the image's encode + prefill cost, under load it includes queueing
    prefill = [r["prompt_tokens"] / r["ttft_s"] for r in ok if r["prompt_tokens"] and r["ttft_s"]]
    acc = sum(scores) / len(scores) if scores else None
    summary = {
        "name": name, "suite": "vqa", "task": task, "tier": spec.get("tier", "quick"),
        "samples": len(items), "errors": len(results) - len(ok),
        "correct": sum(1 for s in scores if s >= 0.5),
        "scores": {"acc": acc, {"docvqa": "anls", "chartqa": "relaxed_acc", "ocrbench": "ocrbench_score",
                                 "mmmu": "acc_mc_open"}[task]: acc},
        "ci95": _ci95(scores),
        "concurrency": int(spec.get("concurrency", 8)),
        "timing": {
            "wall_s": wall, "images_per_s": len(ok) / wall if wall else None,
            "ttft_ms_p50": 1e3 * _pct(ttft, 0.5) if ttft else None,
            "ttft_ms_p90": 1e3 * _pct(ttft, 0.9) if ttft else None,
            "e2el_ms_p50": 1e3 * _pct([r["e2el_s"] for r in ok], 0.5) if ok else None,
            "prompt_tokens_mean": statistics.fmean(prompt) if prompt else None,
            "prefill_tok_per_s_p50": _pct(prefill, 0.5),
            "completion_tokens_mean": statistics.fmean([r["completion_tokens"] for r in ok if r["completion_tokens"]] or [0]),
            "reasoning_replies": sum(1 for r in ok if r["reasoning_chars"]),
        },
        "results": path,
    }
    if task == "ocrbench":
        summary["scores"]["ocrbench_points"] = round(1000 * acc) if acc is not None and spec.get("tier") == "full" else None
    return summary


def run(bench_argv, spec: dict, *, flag, output_dir: str, name: str, push) -> list[dict]:
    """Every task of the pass against the running server; one ``eval_summary`` per task."""
    host = flag(bench_argv, "--host", "127.0.0.1")
    port = flag(bench_argv, "--port", "8000")
    model = flag(bench_argv, "--served-model-name") or flag(bench_argv, "--model")
    base_url = f"http://{host}:{port}"
    tier = spec.get("tier", "quick")
    os.makedirs(output_dir, exist_ok=True)
    summaries = []
    for task in spec.get("tasks") or TASKS:
        items = load(task, tier)
        if spec.get("limit"):
            items = items[: int(spec["limit"])]
        out_path = os.path.join(output_dir, f"{task}.{tier}.jsonl")
        print(f"[vqa] {task} {tier}: {len(items)} items, {spec.get('concurrency', 8)} in flight", flush=True)
        results, wall = asyncio.run(_run_task(base_url, model, items, spec, out_path))
        summary = summarize(task, items, results, wall, name=name, spec=spec, path=out_path)
        print("[eval]", json.dumps(summary), flush=True)
        push(eval_summary=summary)
        summaries.append(summary)
        if summary["errors"] == len(items):
            raise RuntimeError(f"vqa {task}: every request failed ({results[0]['error']})")
    return summaries
