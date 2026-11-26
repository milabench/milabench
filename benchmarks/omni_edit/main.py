#!/usr/bin/env python
"""vLLM-Omni image-editing client: POST /v1/images/edits (multipart: the input image, the instruction, steps,
guidance, seed), one edit at a time by default; the server is milabench's (backend vllm-omni, see ../omni).

Edit i uses ../comfyui_edit/edits.json's edits[i % len] with seed benchmate.prompts.generation_seed(i): the same
inputs, instructions and seeds as the ComfyUI edit benches, so the two engines compare edit by edit. Metrics are
the omni / ComfyUI packs' (gen_seconds per edit, img_per_s = edits per second); the first --keep outputs go to
<bench>.outputs/ with prompts.json (file -> instruction).
"""
import base64
import importlib.util
import itertools
import json
import os
import sys
import threading
import time
import urllib.request
import uuid
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
EDIT_DIR = os.path.join(HERE, "..", "comfyui_edit")
EDITS = json.load(open(os.path.join(EDIT_DIR, "edits.json")))["edits"]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


omni = _load("omni_client", os.path.join(HERE, "..", "omni", "main.py"))  # prepare_voir


def post_edit(url, fields, image_path, timeout):
    boundary = uuid.uuid4().hex
    parts = [f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
             for k, v in fields.items() if v is not None]
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="{os.path.basename(image_path)}"'
                 f'\r\nContent-Type: image/png\r\n\r\n'.encode() + open(image_path, "rb").read() + b"\r\n")
    body = b"".join(parts) + f"--{boundary}--\r\n".encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


_state = {"index": 0, "kept": 0, "lock": threading.Lock()}


def keep_output(args, instruction, body):
    run_dir = os.environ.get("MILABENCH_DIR_RUN")
    if not run_dir:
        return
    with _state["lock"]:
        if _state["kept"] >= args.keep:
            return
        _state["kept"] += 1
        index = _state["kept"]
    item = (json.loads(body).get("data") or [{}])[0]
    if not item.get("b64_json"):
        return
    try:
        name = json.loads(os.environ.get("MILABENCH_CONFIG", "{}")).get("name") or "omni-edit"
    except ValueError:
        name = "omni-edit"
    dest = os.path.join(run_dir, f"{name}.outputs")
    os.makedirs(dest, exist_ok=True)
    filename = f"omni_edit_{index:03d}.png"
    with open(os.path.join(dest, filename), "wb") as f:
        f.write(base64.b64decode(item["b64_json"]))
    with _state["lock"]:
        index_file = os.path.join(dest, "prompts.json")
        prompts = json.load(open(index_file)) if os.path.exists(index_file) else {}
        prompts[filename] = instruction
        json.dump(prompts, open(index_file, "w"), indent=1)


def request(args):
    from benchmate.prompts import generation_seed

    with _state["lock"]:
        index = _state["index"]
        _state["index"] += 1
    item = EDITS[index % len(EDITS)]
    fields = {
        "prompt": item["instruction"], "n": 1, "size": f"{args.width}x{args.height}", "response_format": "b64_json",
        "num_inference_steps": args.steps, "seed": generation_seed(index),
        "guidance_scale": args.guidance, "true_cfg_scale": args.true_cfg, "negative_prompt": args.negative_prompt,
    }
    start = time.perf_counter()
    body = post_edit(f"http://{args.host}:{args.port}/v1/images/edits", fields,
                     os.path.join(EDIT_DIR, "inputs", item["image"]), args.timeout)
    elapsed = time.perf_counter() - start
    try:
        keep_output(args, item["instruction"], body)
    except (OSError, ValueError) as exc:  # a sample is nice to have, never worth failing the benchmark
        print(f"could not keep the output: {exc}", flush=True)
    return elapsed


def arguments(argv):
    parser = ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance", type=float, default=None, help="guidance_scale (embedded guidance / CFG)")
    parser.add_argument("--true-cfg", type=float, default=None, help="true_cfg_scale (Qwen-Image-Edit's CFG)")
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--keep", type=int, default=int(os.environ.get("MILABENCH_KEEP_OUTPUTS", len(EDITS))))
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--ready-check-timeout-sec", type=float, default=None)  # the server side's
    args, _ = parser.parse_known_args(argv)
    return args


def main(argv=None):
    args = arguments(sys.argv[1:] if argv is None else argv)
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        step = lambda: [job.result() for job in [pool.submit(request, args) for _ in range(args.concurrency)]]
        for _ in range(args.warmup):
            for dur in step():
                print(f"warmup edit: {dur:.2f}s", flush=True)
        observer, monitor = omni.prepare_voir()
        dataset = observer.loader(itertools.repeat(None), custom_step=False)
        with monitor():
            for i, _ in enumerate(dataset):
                t0 = time.perf_counter()
                durs = step()
                wall = time.perf_counter() - t0
                for dur in durs:
                    observer.record_metric(gen_seconds=dur, img_per_s=1.0 / dur, batch_size=1)
                if len(durs) > 1:
                    observer.record_metric(step_images=len(durs), step_wall_seconds=wall,
                                           step_imgs_per_s=len(durs) / wall, concurrency=args.concurrency,
                                           batch_size=len(durs))
                if args.max_steps and i + 1 >= args.max_steps:
                    break
    return 0


if __name__ == "__main__":
    sys.exit(main())
