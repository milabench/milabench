#!/usr/bin/env python
"""benchmarks/serving's prepare (the model, its drafter, the text corpus, the other passes' tools), then the
pinned VQA slices and the image request sets of the throughput passes (vqa.py)."""

import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import vqa  # noqa: E402


def _serving_prepare():
    spec = importlib.util.spec_from_file_location(
        "serving_prepare", os.path.join(HERE, "..", "serving", "prepare.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    serving = _serving_prepare()
    specs = [spec["vqa"] or {} for spec in serving.client_passes() if "vqa" in spec]
    vqa.prepare(specs)
    serving.main()


if __name__ == "__main__":
    main()
