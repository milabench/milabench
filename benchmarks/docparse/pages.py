"""Document pages: the pinned benchmark slices, the fixed speed set and the page renderer.

Sources (each a pinned revision of a Hugging Face dataset; prepare downloads only the files of the slice):

    olmocr        allenai/olmOCR-bench             1,403 single-page PDFs, 7,010 unit tests (text present /
                                                   absent, reading order, table cells, math rendered by KaTeX)
                                                   in 7 categories: arxiv_math, headers_footers, long_tiny_text,
                                                   multi_column, old_scans, old_scans_math, tables.
                                                   Score: test pass rate (the official olmocr.bench scorer)
    omnidocbench  opendatalab/OmniDocBench (v1.6)  1,651 page images, 9 document types, en / zh / mixed.
                                                   Score: the official end-to-end evaluator (text edit distance,
                                                   table TEDS, formula edit distance, reading order)
    opb           datalab-to/omni_parse_bench      2,937 pages, 16k unit tests tagged by language and rendering.
                                                   Two slices: ``fr`` (every page with a French test: CERFA
                                                   forms, French handwriting from fr-bench-pdf2md, multi-column)
                                                   and ``hw`` (handwriting: letters, manuscripts, old scans).
                                                   Score: test pass rate (the omni-parse-bench scorer)

Tiers: ``quick`` (105 olmocr PDFs: 15 spread over each category; ~90 OmniDocBench pages: 10 spread over each
document type; opb: every French page (103) and every other handwriting page (86)), ``full`` (every page of olmocr and
OmniDocBench, every French and handwriting page of opb).

The speed set (``perf``): 56 fixed pages, spread over the quick slices: 12 each of olmocr old_scans (scans),
tables, arxiv_math (formulas) and multi_column, plus 8 French pages of opb's multilingual suite (filled CERFA
forms). Every engine reads the same files.

Data layout ($MILABENCH_DIR_DATA/docparse):

    <task>/<tier>/items.jsonl      one page per line: {id, task, path (relative to the data root), kind, meta}
    <task>/<tier>/...              the slice's own files, in the layout its scorer expects
    perf/items.jsonl               the speed set (paths into the slices above)
"""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import tempfile

SOURCES = {
    "olmocr": {"repo": "allenai/olmOCR-bench", "revision": "54a96a6fb6a2bd3b297e59869491db4d3625b711",
               "categories": ["arxiv_math", "headers_footers", "long_tiny_text", "multi_column", "old_scans",
                              "old_scans_math", "table_tests"],
               "quick": 15},  # PDFs per category
    "omnidocbench": {"repo": "opendatalab/OmniDocBench", "revision": "aa1ee96d106dbe53d0ae59474d75c6e6d9b53fec",
                     "quick": 10},  # pages per document type (data_source)
    "opb": {"repo": "datalab-to/omni_parse_bench", "revision": "d70182b9f901d1660a27e60bf6be6d6e6ea83013"},
}

# olmOCR-bench names its table category file table_tests.jsonl, its PDF folder tables/
OLMOCR_PDF_DIR = {"table_tests": "tables"}

TASKS = ("olmocr", "omnidocbench", "opb_fr", "opb_hw")
PERF = [("olmocr", "old_scans", 12), ("olmocr", "table_tests", 12), ("olmocr", "arxiv_math", 12),
        ("olmocr", "multi_column", 12), ("opb", "multilingual", 8)]


def root() -> str:
    base = os.environ.get("MILABENCH_DIR_DATA") or os.path.join(os.path.expanduser("~"), ".cache", "milabench")
    return os.path.join(base, "docparse")


def task_dir(task: str, tier: str) -> str:
    return os.path.join(root(), task, tier)


def load(task: str, tier: str = "quick") -> list[dict]:
    """The pages of a task (``perf`` for the speed set)."""
    path = os.path.join(root(), "perf", "items.jsonl") if task == "perf" else os.path.join(task_dir(task, tier), "items.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path}: run the benchmark's prepare first")
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def file_of(item: dict) -> str:
    return os.path.join(root(), item["path"])


# ---------------------------------------------------------------------------
# Rendering: one page image per (page, render settings), made before any timing starts
# ---------------------------------------------------------------------------


def render(item: dict, settings: dict | None = None):
    """The page as a PIL RGB image.

    settings: ``dpi`` (PDF render resolution, default 200), ``long_edge`` (scale so the long edge is at most this),
    ``max_pixels`` (scale down to at most this many pixels). A PDF is rendered by poppler's pdftoppm (the
    renderer olmOCR uses); an image file is read as it is, then the same limits apply."""
    from PIL import Image

    settings = settings or {}
    path = file_of(item)
    if path.lower().endswith(".pdf"):
        dpi = int(settings.get("dpi", 200))
        if not shutil.which("pdftoppm"):
            raise RuntimeError("pdftoppm not found: install poppler-utils (the page renderer)")
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "page")
            subprocess.run(["pdftoppm", "-r", str(dpi), "-png", "-f", "1", "-l", "1", "-singlefile", path, out],
                           check=True, capture_output=True)
            image = Image.open(out + ".png").convert("RGB")
    else:
        image = Image.open(path).convert("RGB")
    w, h = image.size
    scale = 1.0
    if settings.get("long_edge"):
        scale = min(scale, int(settings["long_edge"]) / max(w, h))
    if settings.get("max_pixels"):
        scale = min(scale, (int(settings["max_pixels"]) / (w * h)) ** 0.5)
    if scale < 1.0:
        image = image.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    return image


def png_bytes(image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Prepare: download the slices (network), write items.jsonl
# ---------------------------------------------------------------------------


def _download(repo: str, revision: str, filename: str, local_dir: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, filename, repo_type="dataset", revision=revision, local_dir=local_dir)


def _every(items: list, n: int) -> list:
    """n items spread evenly over a sorted list (all of them when n is None or larger)."""
    if n is None or len(items) <= n:
        return list(items)
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


def _write_items(path: str, items: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def prepare_olmocr(tier: str) -> list[dict]:
    src = SOURCES["olmocr"]
    out = task_dir("olmocr", tier)
    marker = os.path.join(out, "items.jsonl")
    if os.path.exists(marker):
        return load("olmocr", tier)
    cache = os.path.join(root(), ".hf", "olmocr")
    items = []
    for cat in src["categories"]:
        jsonl = _download(src["repo"], src["revision"], f"bench_data/{cat}.jsonl", cache)
        with open(jsonl) as f:
            tests = [json.loads(line) for line in f if line.strip()]
        pdfs = _every(sorted({t["pdf"] for t in tests}), src["quick"] if tier == "quick" else None)
        keep = set(pdfs)
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, f"{cat}.jsonl"), "w") as f:
            for t in tests:
                if t["pdf"] in keep:
                    f.write(json.dumps(t, ensure_ascii=False) + "\n")
        for pdf in pdfs:
            local = _download(src["repo"], src["revision"], f"bench_data/pdfs/{pdf}", cache)
            dest = os.path.join(out, "pdfs", pdf)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copyfile(local, dest)
            items.append({"id": f"olmocr/{pdf[:-4]}", "task": "olmocr", "kind": "pdf",
                          "path": os.path.relpath(dest, root()), "meta": {"category": cat, "pdf": pdf}})
    _write_items(marker, items)
    return items


def prepare_omnidocbench(tier: str) -> list[dict]:
    src = SOURCES["omnidocbench"]
    out = task_dir("omnidocbench", tier)
    marker = os.path.join(out, "items.jsonl")
    if os.path.exists(marker):
        return load("omnidocbench", tier)
    cache = os.path.join(root(), ".hf", "omnidocbench")
    gt = json.load(open(_download(src["repo"], src["revision"], "OmniDocBench.json", cache)))
    by_type: dict[str, list[dict]] = {}
    for page in gt:
        info = page["page_info"]
        by_type.setdefault(info.get("page_attribute", {}).get("data_source", "other"), []).append(page)
    chosen = []
    for kind in sorted(by_type):
        pages = sorted(by_type[kind], key=lambda p: p["page_info"]["image_path"])
        chosen += _every(pages, src["quick"] if tier == "quick" else None)
    os.makedirs(os.path.join(out, "images"), exist_ok=True)
    with open(os.path.join(out, "OmniDocBench.json"), "w") as f:
        json.dump(chosen, f, ensure_ascii=False)
    items = []
    for page in chosen:
        name = os.path.basename(page["page_info"]["image_path"])
        local = _download(src["repo"], src["revision"], f"images/{name}", cache)
        dest = os.path.join(out, "images", name)
        shutil.copyfile(local, dest)
        attr = page["page_info"].get("page_attribute", {})
        items.append({"id": f"omnidocbench/{os.path.splitext(name)[0]}", "task": "omnidocbench", "kind": "image",
                      "path": os.path.relpath(dest, root()),
                      "meta": {"data_source": attr.get("data_source"), "language": attr.get("language")}})
    _write_items(marker, items)
    return items


def prepare_opb(tier: str) -> dict[str, list[dict]]:
    """Both opb slices (fr, hw) from the published per-test results table (it carries each test's tags)."""
    import pandas as pd

    src = SOURCES["opb"]
    done = {t: os.path.join(task_dir(f"opb_{t}", tier), "items.jsonl") for t in ("fr", "hw")}
    if all(os.path.exists(p) for p in done.values()):
        return {t: load(f"opb_{t}", tier) for t in done}
    cache = os.path.join(root(), ".hf", "opb")
    manifest = pd.read_parquet(_download(src["repo"], src["revision"], "manifest.parquet", cache))
    results = pd.read_parquet(_download(src["repo"], src["revision"], "results.parquet", cache),
                              columns=["system", "page_id", "suite", "tags"])
    tests = results[results.system == results.system.iloc[0]]
    has = lambda tag: set(tests[tests.tags.apply(lambda t: tag in list(t))].page_id)  # noqa: E731
    fr = sorted(has("fra"))
    hw = sorted(has("handwriting") - set(fr))
    slices = {"fr": fr, "hw": hw[::2] if tier == "quick" else hw}
    rows = manifest.set_index("page_id")
    out_items = {}
    for name, page_ids in slices.items():
        out = task_dir(f"opb_{name}", tier)
        items = []
        for pid in page_ids:
            row = rows.loc[pid]
            for rel in (row.doc_path, row.gt_path):
                local = _download(src["repo"], src["revision"], rel, cache)
                dest = os.path.join(out, rel)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.copyfile(local, dest)
            items.append({"id": f"opb/{pid}", "task": f"opb_{name}", "kind": "pdf" if row.doc_path.endswith(".pdf") else "image",
                          "path": os.path.relpath(os.path.join(out, row.doc_path), root()),
                          "meta": {"page_id": pid, "suite": row.suite,
                                   "gold": os.path.relpath(os.path.join(out, row.gt_path), root())}})
        _write_items(done[name], items)
        out_items[name] = items
    return out_items


def prepare_perf() -> list[dict]:
    path = os.path.join(root(), "perf", "items.jsonl")
    if os.path.exists(path):
        return load("perf")
    olm = prepare_olmocr("quick")
    opb = prepare_opb("quick")["fr"]
    items = []
    for source, group, n in PERF:
        pool = ([i for i in olm if i["meta"]["category"] == group] if source == "olmocr"
                else [i for i in opb if i["meta"]["suite"] == group])
        for item in _every(sorted(pool, key=lambda i: i["id"]), n):
            items.append({**item, "perf_group": "scans" if group == "old_scans" else
                          {"table_tests": "tables", "arxiv_math": "formulas", "multi_column": "multi_column",
                           "multilingual": "french_forms"}[group]})
    _write_items(path, items)
    return items


def prepare(tiers=("quick",), tasks=TASKS) -> None:
    for tier in tiers:
        if "olmocr" in tasks:
            prepare_olmocr(tier)
        if "omnidocbench" in tasks:
            prepare_omnidocbench(tier)
        if any(t.startswith("opb") for t in tasks):
            prepare_opb(tier)
    prepare_perf()
    shutil.rmtree(os.path.join(root(), ".hf"), ignore_errors=True)  # the slices hold copies: drop the HF download cache
