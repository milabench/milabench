#!/usr/bin/env python
"""benchmarks/serving's prepare (the model weights), then the docparse data and tools (network, once):

* the pinned page slices of the passes' tasks and tiers, and the fixed speed set (pages.py)
* the scorer venvs (docparse-score, omnidocbench) and, when the bench uses a vendor SDK or a packaged prompt,
  docparse-sdk (toolenv.py); the packaged prompt itself (client.resolve_prompt)
"""

import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import client  # noqa: E402
import pages  # noqa: E402
import toolenv  # noqa: E402


def _serving_prepare():
    spec = importlib.util.spec_from_file_location("serving_prepare", os.path.join(HERE, "..", "serving", "prepare.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_docparse(passes: list[dict], rec: dict) -> None:
    specs = [p["docparse"] or {} for p in passes if "docparse" in p]
    if not specs:
        return
    quality = [s for s in specs if s.get("mode", "quality") == "quality"]
    tiers = sorted({s.get("tier", "quick") for s in quality}) or ["quick"]
    tasks = sorted({t for s in quality for t in (s.get("tasks") or pages.TASKS)}) or list(pages.TASKS)
    pages.prepare(tiers=tiers, tasks=tasks)
    if quality:
        toolenv.ensure("docparse-score")
        if "omnidocbench" in tasks:
            toolenv.ensure("omnidocbench")
    if rec.get("adapter", "chat") != "chat" or "prompt_from" in rec:
        toolenv.ensure("docparse-sdk")
        client.resolve_prompt(rec)


def main():
    serving = _serving_prepare()
    prepare_docparse(serving.client_passes(), client.recipe())
    serving.main()


if __name__ == "__main__":
    main()
