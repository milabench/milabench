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
import itertools
import json
import os
import random
import sys
import time
import urllib.request
import uuid
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor


PROMPTS = [
    "a red fox in a snowy forest at golden hour, photorealistic",
    "a lighthouse on a rocky coast during a storm, dramatic clouds, oil painting",
    "a bowl of ramen on a wooden table, steam rising, close-up food photography",
    "an astronaut riding a horse across a desert at sunset, cinematic",
    "a cozy reading nook with a cat sleeping on a pile of books, soft window light",
    "a futuristic city street at night with neon signs reflected in puddles",
    "a hummingbird drinking from a red flower, macro photograph, shallow depth of field",
    "a mountain lake at sunrise with mist over the water, wide angle landscape",
]


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
    return time.perf_counter() - start, len(out)


def run_step(pool, args, count):
    jobs = [pool.submit(request, args, random.choice(PROMPTS), random.randrange(2**31)) for _ in range(count)]
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
