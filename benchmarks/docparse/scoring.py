"""Scores a pass's raw answers: convert to Markdown, then each task's own official scorer, in its tool venv.

Called by the client (serving or pipeline) after the pages of a task are answered, outside any timed window.

    olmocr        olmocr.bench.benchmark on the slice (its jsonl files + PDFs, the pages as
                  <candidate>/<pdf>_pg1_repeat1.md): the average of the per-category pass rates (the published
                  olmOCR-bench score, baseline tests included) and its 95% CI
    omnidocbench  the official evaluator (end2end, quick_match, no CDM): text edit distance, table TEDS, formula
                  edit distance, reading-order edit distance; ``overall`` = ((1 - text_edit) * 100 + table_TEDS +
                  (1 - formula_edit) * 100) / 3 (the v1.5 overall with formula edit distance in place of CDM)
    opb_fr        omni-parse-bench unit tests tagged ``fra`` (score_opb.py)
    opb_hw        omni-parse-bench unit tests tagged ``handwriting``

Every result is returned as {"acc": 0..1 main score, ...details}.
"""

from __future__ import annotations

import glob
import json
import os
import re
import shutil
import subprocess
import tempfile

import pages
import toolenv

HERE = os.path.dirname(os.path.abspath(__file__))


def convert(answers_path: str, md_dir: str) -> dict:
    py = toolenv.python("docparse-score")
    proc = subprocess.run([py, "-I", os.path.join(HERE, "convert.py"), answers_path, md_dir],
                          check=True, capture_output=True, text=True)
    if proc.stderr:
        print(proc.stderr[-4000:], flush=True)
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _md(md_dir: str, item: dict) -> str:
    return os.path.join(md_dir, item["id"].replace("/", "__") + ".md")


def score_olmocr(items: list[dict], md_dir: str, tier: str, out_dir: str) -> dict:
    src = pages.task_dir("olmocr", tier)
    with tempfile.TemporaryDirectory() as tmp:
        os.symlink(os.path.join(src, "pdfs"), os.path.join(tmp, "pdfs"))
        for jsonl in glob.glob(os.path.join(src, "*.jsonl")):
            if os.path.basename(jsonl) != "items.jsonl":
                os.symlink(jsonl, os.path.join(tmp, os.path.basename(jsonl)))
        for item in items:
            dest = os.path.join(tmp, "candidate", item["meta"]["pdf"][:-4] + "_pg1_repeat1.md")
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copyfile(_md(md_dir, item), dest) if os.path.exists(_md(md_dir, item)) else open(dest, "w").close()
        proc = subprocess.run([toolenv.python("docparse-score"), "-m", "olmocr.bench.benchmark", "--dir", tmp,
                               "--candidate", "candidate", "--bootstrap_samples", "1000"],
                              capture_output=True, text=True)
    with open(os.path.join(out_dir, "olmocr_bench.log"), "w") as f:
        f.write(proc.stdout + "\n" + proc.stderr)
    m = re.search(r"candidate\s*: Average Score: ([\d.]+)% ± ([\d.]+)%", proc.stdout)
    if not m:
        raise RuntimeError(f"olmOCR-bench scorer: no score (exit {proc.returncode}): {proc.stderr[-1000:]}")
    score, half = float(m.group(1)) / 100, float(m.group(2)) / 100
    per_file = {name.strip(): float(rate) / 100
                for name, rate in re.findall(r"^\s+(\S+\.jsonl|baseline)\s*: ([\d.]+)%", proc.stdout, re.M)}
    per_type = {name: float(rate) / 100
                for name, rate in re.findall(r"^\s{4}(\w+)\s*: ([\d.]+)% average pass rate", proc.stdout, re.M)}
    return {"acc": score, "olmocr_pass_rate": score, "ci95": [score - half, score + half],
            "by_category": per_file, "by_test_type": per_type}


def _num(node):
    """The headline number of an OmniDocBench metric entry ({"ALL_page_avg": x} / {"all": x} / x)."""
    if isinstance(node, (int, float)):
        return float(node)
    if isinstance(node, dict):
        for key in ("ALL_page_avg", "all", "ALL"):
            if key in node and isinstance(node[key], (int, float)):
                return float(node[key])
        for value in node.values():
            if isinstance(value, (int, float)):
                return float(value)
    return None


def score_omnidocbench(items: list[dict], md_dir: str, tier: str, out_dir: str) -> dict:
    import yaml

    src = toolenv.source_dir("omnidocbench")
    pred = os.path.join(out_dir, "omnidocbench_pred")
    os.makedirs(pred, exist_ok=True)
    for item in items:
        name = os.path.splitext(os.path.basename(item["path"]))[0] + ".md"
        shutil.copyfile(_md(md_dir, item), os.path.join(pred, name)) if os.path.exists(_md(md_dir, item)) \
            else open(os.path.join(pred, name), "w").close()
    workers = max(1, min(13, (os.cpu_count() or 4) // 2))
    config = {"end2end_eval": {
        "metrics": {"text_block": {"metric": ["Edit_dist"]},
                    "display_formula": {"metric": ["Edit_dist"]},
                    "table": {"metric": ["TEDS", "Edit_dist"], "teds_workers": workers},
                    "reading_order": {"metric": ["Edit_dist"]}},
        "dataset": {"dataset_name": "end2end_dataset",
                    "ground_truth": {"data_path": os.path.join(pages.task_dir("omnidocbench", tier), "OmniDocBench.json")},
                    "prediction": {"data_path": pred},
                    "match_method": "quick_match", "match_workers": workers}}}
    cfg_path = os.path.join(out_dir, "omnidocbench.yaml")
    with open(cfg_path, "w") as f:
        yaml.safe_dump(config, f)
    env = {**os.environ, **toolenv.omnidocbench_env()}
    proc = subprocess.run([toolenv.python("omnidocbench"), "pdf_validation.py", "--config", cfg_path],
                          cwd=src, capture_output=True, text=True, env=env)
    with open(os.path.join(out_dir, "omnidocbench.log"), "w") as f:
        f.write(proc.stdout + "\n" + proc.stderr)
    found = sorted(glob.glob(os.path.join(src, "result", f"{os.path.basename(pred)}*metric_result.json")),
                   key=os.path.getmtime)
    if proc.returncode != 0 or not found:
        raise RuntimeError(f"OmniDocBench evaluator failed (exit {proc.returncode}): {proc.stderr[-1000:]}")
    result = json.load(open(found[-1]))
    shutil.copyfile(found[-1], os.path.join(out_dir, "omnidocbench_metric_result.json"))
    get = lambda block, metric: _num(((result.get(block) or {}).get("all") or {}).get(metric))  # noqa: E731
    text, table, formula, order = (get("text_block", "Edit_dist"), get("table", "TEDS"),
                                   get("display_formula", "Edit_dist"), get("reading_order", "Edit_dist"))
    if table is not None and table <= 1.0:
        table *= 100  # TEDS as a percentage, as the leaderboard prints it
    parts = [x for x in ((1 - text) * 100 if text is not None else None, table,
                         (1 - formula) * 100 if formula is not None else None) if x is not None]
    overall = sum(parts) / len(parts) if parts else None
    return {"acc": overall / 100 if overall is not None else None, "overall_nocdm": overall,
            "text_edit": text, "table_teds": table, "formula_edit": formula, "reading_order_edit": order}


def score_opb(items: list[dict], md_dir: str, tier: str, out_dir: str, task: str) -> dict:
    tag = {"opb_fr": "fra", "opb_hw": "handwriting"}[task]
    out = os.path.join(out_dir, f"{task}_tests.json")
    items_path = os.path.join(pages.task_dir(task, tier), "items.jsonl")
    proc = subprocess.run([toolenv.python("docparse-score"), "-I", os.path.join(HERE, "score_opb.py"), items_path,
                           md_dir, pages.root(), tag, out], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"omni-parse-bench scorer failed: {proc.stderr[-1000:]}")
    r = json.load(open(out))
    n, k = r["tagged_tests"], r["tagged_passed"]
    return {"acc": r["pass_rate"], f"{tag}_pass_rate": r["pass_rate"], "tests": n, "passed": k,
            "ci95": _wilson(k, n), "all_html_pass_rate": r["all_pass_rate"]}


def _wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if not n:
        return []
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
    return [max(0.0, centre - half), min(1.0, centre + half)]


def score(task: str, items: list[dict], answers_path: str, tier: str, out_dir: str) -> dict:
    md_dir = os.path.join(out_dir, f"{task}_md")
    conv = convert(answers_path, md_dir)
    if task == "olmocr":
        result = score_olmocr(items, md_dir, tier, out_dir)
    elif task == "omnidocbench":
        result = score_omnidocbench(items, md_dir, tier, out_dir)
    else:
        result = score_opb(items, md_dir, tier, out_dir, task)
    result["convert_failed"] = conv.get("converted_failed", 0)
    return result
