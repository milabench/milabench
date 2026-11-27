#!/usr/bin/env python
"""ComfyUI image-editing benchmark: image in + instruction in -> image out.

The ComfyUI pack's driver (../comfyui/main.py: server, model download, timing, observer, kept outputs) with
one change: generation i edits edits.json's edits[i % len] (an input image and an English instruction) with
seed benchmate.prompts.generation_seed(i). Every model gets the same inputs, instructions and seeds in the
same order, so the outputs each bench keeps (<bench>.outputs/, prompts.json: file -> instruction) compare
edit by edit, like the text-to-image table.

Workflow placeholders, on top of the ComfyUI pack's (__SEED__, __NAME__ from --set):
  "__IMAGE__"          -> the edit's input image, staged in ComfyUI's input/ folder (LoadImage)
  "...__PROMPT__..."   -> the edit's instruction

Metrics are the ComfyUI pack's: gen_seconds (one edit, from ComfyUI's execution_start to execution_success:
image load, text/vision encoding, VAE encode, sampling, decode) and img_per_s (edits per second).
"""
import importlib.util
import json
import os
import shutil
import sys
import time
import uuid
from argparse import ArgumentParser

HERE = os.path.dirname(os.path.abspath(__file__))
EDITS = json.load(open(os.path.join(HERE, "edits.json")))["edits"]
STAGED_PREFIX = "milabench_edit_"


def load_comfy_driver():
    spec = importlib.util.spec_from_file_location("comfyui_driver", os.path.join(HERE, "..", "comfyui", "main.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


comfy = load_comfy_driver()


def stage_inputs():
    """Copy the input images into ComfyUI's input/ folder (LoadImage reads from there)."""
    dest = os.path.join(comfy.COMFY_ROOT, "input")
    os.makedirs(dest, exist_ok=True)
    for name in sorted({e["image"] for e in EDITS}):
        target = os.path.join(dest, STAGED_PREFIX + name)
        src = os.path.join(HERE, "inputs", name)
        if not os.path.exists(target) or os.path.getmtime(target) < os.path.getmtime(src):
            shutil.copy2(src, target)


def edit(index):
    return EDITS[index % len(EDITS)]


def submit_edit(server, workflow, sets):
    """comfyui/main.py's submit_prompt, with an input image and an instruction per generation."""
    from benchmate.prompts import generation_seed

    index = comfy._generation["index"]
    comfy._generation["index"] += 1
    item = edit(index)
    seed, instruction = generation_seed(index), item["instruction"]
    graph = comfy.patch(json.load(open(workflow)), {**sets, "__IMAGE__": STAGED_PREFIX + item["image"]},
                        seed, instruction)
    resp = comfy.http_json(
        f"http://127.0.0.1:{server.port}/prompt",
        {"prompt": graph, "client_id": str(uuid.uuid4())},
    )
    return resp["prompt_id"], (time.perf_counter(), instruction)


def main(argv):
    parser = ArgumentParser()
    parser.add_argument("--keep", type=int, default=int(os.environ.get("MILABENCH_KEEP_OUTPUTS", len(EDITS))),
                        help="outputs kept for the report (default: one per edit)")
    args, _ = parser.parse_known_args(argv)
    comfy.KEEP_OUTPUTS = args.keep
    comfy.submit_prompt = submit_edit  # run_batch / run_prompt look it up at call time
    stage_inputs()
    return comfy.main(argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
