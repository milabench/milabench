#!/usr/bin/env python
"""The docparse page sets and scorers (benchmarks/docparse: pages.py, toolenv.py), the engine's models (tessdata_best
for tesseract, RapidOCR's PP-OCRv6 ONNX files, docling's layout and table models): network, once."""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(HERE, "..", "docparse"))

import pages  # noqa: E402
import pipelines  # noqa: E402
import toolenv  # noqa: E402


def main():
    cfg = json.loads(os.environ.get("MILABENCH_CONFIG") or "{}")
    rec = cfg.get("docparse") or {}
    passes = [p["docparse"] or {} for p in (cfg.get("client") or {}).get("passes") or [] if "docparse" in p]
    quality = [s for s in passes if s.get("mode", "quality") == "quality"]
    tiers = sorted({s.get("tier", "quick") for s in quality}) or ["quick"]
    tasks = sorted({t for s in quality for t in (s.get("tasks") or pages.TASKS)}) or list(pages.TASKS)
    pages.prepare(tiers=tiers, tasks=tasks)
    if quality:
        toolenv.ensure("docparse-score")
        if "omnidocbench" in tasks:
            toolenv.ensure("omnidocbench")
    if rec.get("engine") == "tesseract" or rec.get("ocr") == "tesseract":
        pipelines.prepare_tessdata()
    if rec.get("engine"):
        pipelines.warm(rec["engine"], rec)


if __name__ == "__main__":
    main()
