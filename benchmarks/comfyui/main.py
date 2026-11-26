#!/usr/bin/env python
"""ComfyUI headless benchmark driver.

Boots a pinned ComfyUI checkout (isolated venv with torch xpu), submits an
API-format workflow repeatedly over its HTTP API, and pushes per-image
generation times to the milabench observer.

Workflow JSONs contain placeholder strings patched at submit time:
  "__SEED__"  -> the generation's seed (benchmate.prompts.generation_seed: fixed per generation index)
  "...__PROMPT__..." -> the generation's shared prompt (benchmate.prompts.image_prompt), in image workflows
  "__NAME__"  -> value from --set NAME=value (e.g. checkpoint filenames)

--input REPO:FILE (or a local path) links an input file (a LoadImage image) into COMFY_ROOT/input under its
basename: --set INPUT=<basename> names it in the workflow.
"""
import itertools
import json
import shutil
import os
import random
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from argparse import ArgumentParser


COMFY_ROOT = os.environ.get("MILABENCH_COMFY_ROOT", "/srv/ai-cold/scratch/comfyui")
COMFY_PYTHON = os.environ.get(
    "MILABENCH_COMFY_PYTHON", "/srv/ai-cold/scratch/comfyui-xpu-venv/bin/python"
)


def http_json(url, data=None, timeout=60):
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode() if data is not None else None,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def link_model(dest_dir, cached_path, filename):
    target = os.path.join(dest_dir, os.path.basename(filename))
    os.makedirs(os.path.dirname(target), exist_ok=True)
    if os.path.islink(target) and os.readlink(target) != cached_path:
        # a dangling link (the cache entry it pointed to is gone), or one to another repo's file of the same name
        # (Fibo 2.0's and FLUX 3 Klein's qwen3vl_4b_bf16, Flux-2-Dev's and Klein's flux2-vae differ): point it at
        # the file this bench asked for, not whichever bench linked the name first
        if os.path.exists(target):
            print(f"model relinked: {target} -> {cached_path} (was {os.readlink(target)})", flush=True)
        os.unlink(target)
    if not os.path.exists(target):
        os.symlink(cached_path, target)
    return target


def fetch_models(models):
    from huggingface_hub import hf_hub_download

    if len(models) == 1 and "," in models[0]:
        models = models[0].split(",")

    paths = []
    for spec in models:
        dest, repo, filename = spec.split(":", 2)
        cached = hf_hub_download(repo_id=repo, filename=filename)
        paths.append(link_model(os.path.join(COMFY_ROOT, "models", dest), cached, filename))
        print(f"model ready: {paths[-1]}", flush=True)
    return paths


def fetch_inputs(inputs):
    """--input REPO:FILE (hf_hub_download) or a local path, linked into COMFY_ROOT/input under its basename."""
    paths = []
    for spec in inputs:
        for item in [s for s in spec.split(",") if s]:
            if os.path.exists(item):
                source = os.path.abspath(item)
            else:
                repo, sep, filename = item.partition(":")
                if not sep:
                    raise SystemExit(f"--input {item!r}: neither a local file nor REPO:FILE")
                from huggingface_hub import hf_hub_download

                source = hf_hub_download(repo_id=repo, filename=filename)
            paths.append(link_model(os.path.join(COMFY_ROOT, "input"), source, source))
            print(f"input ready: {paths[-1]}", flush=True)
    return paths


def patch(node, sets, seed, prompt=None):
    if isinstance(node, dict):
        return {k: patch(v, sets, seed, prompt) for k, v in node.items()}
    if isinstance(node, list):
        return [patch(v, sets, seed, prompt) for v in node]
    if node == "__SEED__":
        return seed
    if isinstance(node, str) and node in sets:
        return sets[node]
    if isinstance(node, str) and "__PROMPT__" in node and prompt is not None:
        return node.replace("__PROMPT__", prompt)
    return node


# generations so far: generation i uses the shared prompt i and seed i (benchmate.prompts), the same in every
# image model, so their kept outputs compare prompt by prompt
_generation = {"index": 0}


class ComfyServer:
    def __init__(self, port, device, log_path):
        self.port = port
        env = os.environ.copy()
        env["XPU_VISIBLE_DEVICES"] = str(device)
        env.setdefault("COMFYUSB_DISABLE_HARDWARE", "")
        self.log = open(log_path, "w")
        self.proc = subprocess.Popen(
            [
                COMFY_PYTHON, "main.py",
                "--disable-auto-launch",
                "--listen", "127.0.0.1",
                "--port", str(port),
            ],
            cwd=COMFY_ROOT,
            env=env,
            stdout=self.log,
            stderr=subprocess.STDOUT,
        )

    def wait_ready(self, timeout=900):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"comfyui server exited early, see {self.log.name}")
            try:
                http_json(f"http://127.0.0.1:{self.port}/system_stats", timeout=5)
                return
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                time.sleep(2)
        raise RuntimeError("comfyui server did not become ready in time")

    def shutdown(self):
        self.proc.terminate()
        try:
            self.proc.wait(30)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.log.close()


def submit_prompt(server, workflow, sets):
    from benchmate.prompts import generation_seed, image_prompt

    index = _generation["index"]
    _generation["index"] += 1
    seed, prompt = generation_seed(index), image_prompt(index)
    text = open(workflow).read()
    graph = patch(json.loads(text), sets, seed, prompt)
    resp = http_json(
        f"http://127.0.0.1:{server.port}/prompt",
        {"prompt": graph, "client_id": str(uuid.uuid4())},
    )
    # the prompt only when the workflow takes one (__PROMPT__): audio and video workflows keep their own
    return resp["prompt_id"], (time.perf_counter(), prompt if "__PROMPT__" in text else None)


def run_prompt(server, workflow, sets, timeout=3600):
    return run_batch(server, workflow, sets, 1, timeout)[0]


def keep_outputs(entry, limit, prompt=None):
    """Copy a generation's files (images, videos, audio) next to the run's results, <bench>.outputs/, the first
    `limit` of them: the report shows what the model made, not only how fast."""
    run_dir = os.environ.get("MILABENCH_DIR_RUN")
    if not run_dir or limit <= 0:
        return
    try:
        name = json.loads(os.environ.get("MILABENCH_CONFIG", "{}")).get("name") or "comfyui"
    except ValueError:
        name = "comfyui"
    dest = os.path.join(run_dir, f"{name}.outputs")
    os.makedirs(dest, exist_ok=True)
    kept = len([f for f in os.listdir(dest) if f != "prompts.json"])
    for node in (entry.get("outputs") or {}).values():
        for kind in ("images", "gifs", "videos", "audio"):
            for item in node.get(kind) or []:
                if kept >= limit or item.get("type", "output") != "output":
                    return
                src = os.path.join(COMFY_ROOT, "output", item.get("subfolder", ""), item["filename"])
                if os.path.isfile(src):
                    shutil.copy2(src, os.path.join(dest, os.path.basename(src)))
                    kept += 1
                    if prompt:  # which prompt made which file, for the report's side-by-side view
                        index_file = os.path.join(dest, "prompts.json")
                        prompts = json.load(open(index_file)) if os.path.exists(index_file) else {}
                        prompts[os.path.basename(src)] = prompt
                        json.dump(prompts, open(index_file, "w"), indent=1)


KEEP_OUTPUTS = int(os.environ.get("MILABENCH_KEEP_OUTPUTS", "4"))


def run_batch(server, workflow, sets, count, timeout=3600):
    """Submit `count` generations; return per-image durations measured from
    ComfyUI's own execution_start/execution_success timestamps (falls back to
    submit->completion wall time)."""
    pending = dict(submit_prompt(server, workflow, sets) for _ in range(count))
    results = {}
    deadline = time.perf_counter() + timeout
    while pending and time.perf_counter() < deadline:
        hist = http_json(f"http://127.0.0.1:{server.port}/history", timeout=30)
        for prompt_id in list(pending):
            if prompt_id not in hist:
                continue
            entry = hist.pop(prompt_id)
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError(f"workflow execution failed: {json.dumps(status)[:2000]}")
            started, prompt = pending[prompt_id]
            elapsed = time.perf_counter() - started
            if "execution_start" in status and "execution_success" in status:
                elapsed = (status["execution_success"] - status["execution_start"]) / 1000.0
            results[prompt_id] = elapsed
            del pending[prompt_id]
            try:
                keep_outputs(entry, KEEP_OUTPUTS, prompt)
            except OSError as exc:  # a sample is nice to have, never worth failing the benchmark
                print(f"could not keep outputs: {exc}", flush=True)
        if pending and server.proc.poll() is not None:
            raise RuntimeError("comfyui server died during generation")
        time.sleep(0.25)
    if pending:
        raise RuntimeError("workflow generation timed out")
    return [results[p] for p in results]


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


def main(argv=None):
    parser = ArgumentParser()
    parser.add_argument("--workflow", required=True, help="API-format workflow JSON")
    parser.add_argument(
        "--model", action="append", default=[],
        help="DEST:REPO:FILE downloaded+symlinked into the comfy models dir",
    )
    parser.add_argument(
        "--input", action="append", default=[], dest="inputs",
        help="REPO:FILE (or a local path) symlinked into the comfy input dir under its basename",
    )
    parser.add_argument("--set", action="append", default=[], dest="sets",
                        help="placeholder replacement NAME=VALUE")
    parser.add_argument("--device", default=os.environ.get("MILABENCH_GPU_ID", "0"))
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--concurrency", type=int, default=1,
        help="images in flight per measured step (server queue depth)",
    )
    parser.add_argument(
        "--max-steps", type=int, default=0,
        help="stop after N images (0 = let voir/earlystop decide)",
    )
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument(
        "--no-observer", action="store_true", help="skip voir/observer (bring-up)"
    )
    args, _ = parser.parse_known_args(argv)

    fetch_models(args.model)
    fetch_inputs(args.inputs)
    kvs = [kv for entry in args.sets for kv in entry.split(",") if kv]
    sets = {f"__{kv.split('=', 1)[0]}__": kv.split("=", 1)[1] for kv in kvs}

    log_path = os.path.join(
        tempfile.gettempdir(), f"comfyui-server-{args.port or os.getpid()}.log"
    )
    server = ComfyServer(args.port or free_port(), args.device, log_path)
    try:
        server.wait_ready()
        for _ in range(args.warmup):
            dur = run_prompt(server, args.workflow, sets)
            print(f"warmup gen: {dur:.2f}s", flush=True)

        if args.no_observer:
            while True:
                dur = run_prompt(server, args.workflow, sets)
                print(f"gen: {dur:.2f}s ({1/dur:.4f} img/s)", flush=True)

        observer, monitor = prepare_voir()
        dataset = observer.loader(itertools.repeat(None), custom_step=False)
        with monitor():
            for i, _ in enumerate(dataset):
                t0 = time.perf_counter()
                durs = run_batch(server, args.workflow, sets, args.concurrency)
                wall = time.perf_counter() - t0
                for dur in durs:
                    observer.record_metric(
                        gen_seconds=dur,
                        img_per_s=1.0 / dur,
                        batch_size=1,
                        workflow=os.path.basename(args.workflow),
                    )
                if len(durs) > 1:
                    observer.record_metric(
                        step_images=len(durs),
                        step_wall_seconds=wall,
                        step_imgs_per_s=len(durs) / wall,
                        concurrency=args.concurrency,
                        batch_size=len(durs),
                    )
                if args.max_steps and i + 1 >= args.max_steps:
                    break
    finally:
        server.shutdown()
    return 0


if __name__ == "__main__":
    main(sys.argv[1:])
