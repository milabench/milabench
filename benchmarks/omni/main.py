#!/usr/bin/env python
"""vLLM-Omni diffusion client: milabench runs ``vllm serve <model> --omni`` (milabench.commands.serving,
backend vllm-omni); this script sends generation requests and pushes their timings to the observer.

One step = ``--concurrency`` requests in flight; every generation is timed on its own. The metrics are
the ComfyUI pack's (``gen_seconds``, ``img_per_s`` = generations / s, images or clips), so the two engines
compare on the same workflow settings (model, resolution, steps, guidance).

Tasks:
  t2i  POST /v1/images/generations (JSON)
  t2v  POST /v1/videos/sync (multipart, returns the video bytes)
"""
import base64
import itertools
import json
import threading
import os
import random
import sys
import time
import urllib.request
import uuid
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor


# prompts: benchmate.prompts (shared with the ComfyUI image workflows)


def post_json(url, payload, timeout):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def post_multipart(url, fields, timeout):
    boundary = uuid.uuid4().hex
    body = b"".join(
        f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode()
        for key, value in fields.items()
    ) + f"--{boundary}--\r\n".encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


KEEP_OUTPUTS = int(os.environ.get("MILABENCH_KEEP_OUTPUTS", "4"))
_kept = {"count": 0, "lock": threading.Lock()}


def keep_output(args, prompt, body):
    """The first few generations next to the run's results, <bench>.outputs/, with their prompts
    (prompts.json): the report shows what the model made, not only how fast."""
    run_dir = os.environ.get("MILABENCH_DIR_RUN")
    if not run_dir:
        return
    with _kept["lock"]:
        if _kept["count"] >= KEEP_OUTPUTS:
            return
        _kept["count"] += 1
        index = _kept["count"]
    try:
        name = json.loads(os.environ.get("MILABENCH_CONFIG", "{}")).get("name") or "omni"
    except ValueError:
        name = "omni"
    dest = os.path.join(run_dir, f"{name}.outputs")
    os.makedirs(dest, exist_ok=True)
    if args.task == "t2i":  # OpenAI images API: {"data": [{"b64_json": ...}]}
        item = (json.loads(body).get("data") or [{}])[0]
        if not item.get("b64_json"):
            return
        data, filename = base64.b64decode(item["b64_json"]), f"omni_{index:03d}.png"
    else:  # /v1/videos/sync answers the MP4 itself
        data, filename = body, f"omni_{index:03d}.mp4"
    with open(os.path.join(dest, filename), "wb") as f:
        f.write(data)
    with _kept["lock"]:
        index_file = os.path.join(dest, "prompts.json")
        prompts = json.load(open(index_file)) if os.path.exists(index_file) else {}
        prompts[filename] = prompt
        json.dump(prompts, open(index_file, "w"), indent=1)


def request(args, prompt, seed):
    """One generation; returns its wall time (s) and the size of the response."""
    base = f"http://{args.host}:{args.port}"
    params = {"num_inference_steps": args.steps, "seed": seed}
    if args.guidance is not None:
        params["guidance_scale"] = args.guidance
    if args.negative_prompt:
        params["negative_prompt"] = args.negative_prompt

    start = time.perf_counter()
    if args.task == "t2i":
        out = post_json(f"{base}/v1/images/generations", {
            "prompt": prompt, "size": f"{args.width}x{args.height}", "n": 1, **params,
        }, args.timeout)
    else:
        out = post_multipart(f"{base}/v1/videos/sync", {
            "prompt": prompt, "width": args.width, "height": args.height,
            "num_frames": args.num_frames, "fps": args.fps, **params,
        }, args.timeout)
    elapsed = time.perf_counter() - start
    try:
        keep_output(args, prompt, out)
    except (OSError, ValueError) as exc:  # a sample is nice to have, never worth failing the benchmark
        print(f"could not keep the output: {exc}", flush=True)
    return elapsed, len(out)


_generation = {"index": 0, "lock": threading.Lock()}


def next_generation():
    """Generation i: the shared prompt i and seed i (benchmate.prompts), the same in every image model and engine,
    so the outputs each benchmark keeps (its first ones) compare prompt by prompt."""
    from benchmate.prompts import generation_seed, image_prompt

    with _generation["lock"]:
        index = _generation["index"]
        _generation["index"] += 1
    return image_prompt(index), generation_seed(index)


def run_step(pool, args, count):
    jobs = [pool.submit(request, args, *next_generation()) for _ in range(count)]
    return [job.result() for job in jobs]


def prepare_voir():
    import torchcompat.core as accelerator
    from benchmate.observer import BenchObserver
    from benchmate.monitor import bench_monitor
    from benchmate.toggles import get_observation_count

    observer = BenchObserver(
        accelerator.Event,
        earlystop=get_observation_count(30),
        batch_size_fn=lambda x: 1,
        raise_stop_program=False,
        stdout=True,
    )
    return observer, bench_monitor


def arguments(argv):
    parser = ArgumentParser()
    parser.add_argument("--task", choices=["t2i", "t2v"], default="t2i")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance", type=float, default=None,
                        help="guidance_scale (distilled guidance for FLUX, CFG for the others)")
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--concurrency", type=int, default=1, help="generations in flight per step")
    parser.add_argument("--warmup", type=int, default=1, help="untimed generations first (compile, graphs)")
    parser.add_argument("--max-steps", type=int, default=0, help="stop after N steps (0 = voir decides)")
    parser.add_argument("--timeout", type=float, default=3600)
    # the server side's, ignored here
    parser.add_argument("--ready-check-timeout-sec", type=float, default=None)
    args, _ = parser.parse_known_args(argv)
    return args


def main(argv=None):
    args = arguments(sys.argv[1:] if argv is None else argv)
    label = "video" if args.task == "t2v" else "image"

    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        for _ in range(args.warmup):
            for dur, _ in run_step(pool, args, args.concurrency):
                print(f"warmup {label}: {dur:.2f}s", flush=True)

        observer, monitor = prepare_voir()
        dataset = observer.loader(itertools.repeat(None), custom_step=False)
        with monitor():
            for i, _ in enumerate(dataset):
                t0 = time.perf_counter()
                results = run_step(pool, args, args.concurrency)
                wall = time.perf_counter() - t0
                for dur, _ in results:
                    observer.record_metric(gen_seconds=dur, img_per_s=1.0 / dur, batch_size=1, task=args.task)
                if len(results) > 1:
                    observer.record_metric(
                        step_images=len(results),
                        step_wall_seconds=wall,
                        step_imgs_per_s=len(results) / wall,
                        concurrency=args.concurrency,
                        batch_size=len(results),
                    )
                if args.max_steps and i + 1 >= args.max_steps:
                    break
    return 0


if __name__ == "__main__":
    sys.exit(main())
