#!/usr/bin/env python
"""llama.cpp serving benchmark.

Default `host` mode: boots llama-server (OpenAI-compatible API) with the
selected ggml backend (sycl | vulkan) and drives it with the shared
benchmate bench client (vllm.benchmarks.serve in-process, random ISL/OSL,
saturating request rate, bounded concurrency) — the exact client used by the
vLLM packs, so the numbers are directly comparable and report as tok/s.

`micro` mode runs llama-bench (pure prompt/generate token throughput).

Anything after `--` in the pack argv is forwarded verbatim to
`vllm bench serve`; unknown args before `--` are forwarded to llama-server.
"""
import itertools
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from argparse import ArgumentParser

from benchmate.benchserve import InferenceServerError, run_benchmark_watched, set_metric_sink


SETVARS = os.environ.get("ONEAPI_SETVARS", "/opt/intel/oneapi/setvars.sh")
DEFAULT_ROOT = os.environ.get(
    "MILABENCH_LLAMACPP_ROOT", "/srv/ai-cold/scratch/llama.cpp"
)


def find_binary(root, backend, name):
    for candidate in (
        os.path.join(root, f"build-{backend}", "bin", name),
        os.path.join(root, f"build-{backend}", name),
    ):
        if os.path.exists(candidate):
            return candidate
    raise RuntimeError(f"could not find {name} for backend '{backend}' under {root}")


def server_env(backend, device):
    env = os.environ.copy()
    # Respect a per-job selector already injected by the runner (D0/D1 spread).
    if backend == "sycl":
        env.setdefault("ONEAPI_DEVICE_SELECTOR", f"level_zero:{device}")
        if os.environ.get("MILABENCH_GPU_ID"):
            env["ONEAPI_DEVICE_SELECTOR"] = f"level_zero:{device}"
    else:
        env.setdefault("GGML_VK_VISIBLE_DEVICES", str(device))
        if os.environ.get("MILABENCH_GPU_ID"):
            env["GGML_VK_VISIBLE_DEVICES"] = str(device)
    return env


def wrap_sycl(cmd):
    if os.environ.get("MILABENCH_LLAMACPP_NO_SETVARS"):
        return cmd
    return ["bash", "-c", f'source "{SETVARS}" --force >/dev/null 2>&1; exec "$@"', "--", *cmd]


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class LlamaServer:
    def __init__(self, args, model_path, port, extra_argv):
        binary = find_binary(args.llama_root, args.backend, "llama-server")
        cmd = [
            binary,
            "--model", model_path,
            "--alias", "model",
            "--host", "127.0.0.1",
            "--port", str(port),
            *extra_argv,
        ]
        if args.backend == "sycl":
            cmd = wrap_sycl(cmd)
        env = server_env(args.backend, args.device)
        self.port = port
        self.log_path = os.path.join(tempfile.gettempdir(), f"llama-server-{port}.log")
        self.log = open(self.log_path, "w")
        print("SERVER:", " ".join(cmd), flush=True)
        self.proc = subprocess.Popen(cmd, env=env, stdout=self.log, stderr=subprocess.STDOUT)

    def wait_ready(self, timeout=1200):
        url = f"http://127.0.0.1:{self.port}/v1/models"
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited early, see {self.log_path}")
            try:
                with urllib.request.urlopen(url, timeout=5) as resp:
                    if resp.status == 200:
                        return
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
                pass
            time.sleep(2)
        raise RuntimeError("llama-server did not become ready in time")

    def check(self):
        rc = self.proc.poll()
        if rc is not None:
            raise InferenceServerError(
                f"llama-server exited early (code {rc}), see {self.log_path}"
            )

    def shutdown(self):
        self.proc.terminate()
        try:
            self.proc.wait(30)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.log.close()


_ROW = re.compile(r"^\|")


def parse_bench_output(text):
    pp = tg = None
    for line in text.splitlines():
        if not _ROW.match(line):
            continue
        cols = [c.strip() for c in line.strip().strip("|").split("|")]
        value = None
        for cell in reversed(cols):
            m = re.match(r"^([0-9]+(?:\.[0-9]+)?)(?:\s*(?:±|\+/-).*)?$", cell)
            if m:
                value = float(m.group(1))
                break
        if value is None:
            continue
        for cell in cols:
            if re.fullmatch(r"pp\d+", cell):
                pp = value
            elif re.fullmatch(r"tg\d+", cell):
                tg = value
    return pp, tg


def run_micro_once(args, model_path, device):
    bench = find_binary(args.llama_root, args.backend, "llama-bench")
    cmd = [bench, "-m", model_path, "-p", str(args.pp), "-n", str(args.tg),
           "-r", str(args.repetitions)]
    env = server_env(args.backend, device)
    if args.backend == "sycl":
        cmd = wrap_sycl(cmd)
    print("RUN:", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout + proc.stderr)
        raise RuntimeError(f"llama-bench failed with code {proc.returncode}")
    pp, tg = parse_bench_output(proc.stdout)
    if pp is None or tg is None:
        sys.stderr.write(proc.stdout + proc.stderr)
        raise RuntimeError("could not parse llama-bench output")
    return pp, tg


def prepare_voir():
    import torchcompat.core as accelerator
    from benchmate.observer import BenchObserver
    from benchmate.monitor import bench_monitor
    from benchmate.toggles import get_observation_count

    observer = BenchObserver(
        accelerator.Event,
        earlystop=get_observation_count(120),
        batch_size_fn=lambda x: 1,
        raise_stop_program=False,
        stdout=True,
    )
    set_metric_sink(observer.record_metric)
    return observer, bench_monitor


def split_args(argv, my_args):
    sep = len(argv)
    client = []
    if "--" in argv:
        sep = argv.index("--")
        client = argv[sep + 1:]
    server_extra = [a for a in argv[:sep] if a not in my_args]
    return server_extra, client


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = ArgumentParser()
    parser.add_argument("--repo", required=True, help="HF repo id of the GGUF")
    parser.add_argument("--file", required=True, help="GGUF filename in the repo")
    parser.add_argument("--model", default=None,
                        help="HF id used by the bench client for the tokenizer")
    parser.add_argument("--backend", default="sycl", choices=["sycl", "vulkan"])
    parser.add_argument("--mode", default="host", choices=["host", "micro"])
    parser.add_argument("--device", default=os.environ.get("MILABENCH_GPU_ID", "0"))
    parser.add_argument("--llama-root", default=DEFAULT_ROOT)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0)
    # micro mode only
    parser.add_argument("--pp", type=int, default=512)
    parser.add_argument("--tg", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=5)
    args, rest = parser.parse_known_args(argv)
    server_extra, client_argv = split_args(rest, [])

    if not args.model:
        args.model = args.repo

    from huggingface_hub import hf_hub_download

    model_path = hf_hub_download(repo_id=args.repo, filename=args.file)

    observer, monitor = prepare_voir()
    dataset = observer.loader(itertools.repeat(None), custom_step=False)

    if args.mode == "micro":
        with monitor():
            for i, _ in enumerate(dataset):
                pp, tg = run_micro_once(args, model_path, args.device)
                observer.record_metric(
                    pp_tps=pp, tg_tps=tg,
                    backend=args.backend, model=args.file, batch_size=1,
                )
                if args.max_steps and i + 1 >= args.max_steps:
                    break
        return 0

    port = args.port or free_port()
    server = LlamaServer(args, model_path, port, server_extra)
    try:
        server.wait_ready()
        bench_argv = [
            "--base-url", f"http://127.0.0.1:{port}",
            "--endpoint", "/v1/completions",
            "--backend", "openai",
            "--model", args.model,
            "--served-model-name", "model",
            "--dataset-name", "random",
            "--request-rate", "inf",
            "--ready-check-timeout-sec", "1200",
            *client_argv,
        ]
        with monitor():
            run_benchmark_watched(server, bench_argv)
    finally:
        server.shutdown()
    return 0


if __name__ == "__main__":
    main(sys.argv[1:])
