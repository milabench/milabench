"""``quality`` client passes: QualityBench (launch80/QualityBench) against the running server.

    passes:
      - name: quality
        quality:
          tier: quick              # quick (every tenth item) | full
          thinking: [off]          # one run per value: off / on (thinking models need one of them)
          tasks: [gsm8k, ifeval, mmlu_pro, humaneval, mbpp, bfcl, lcb, logprobs, needle]   # default suite
          thinking_max_tokens: 32768
          max_tokens: 24576        # optional: a floor on every reply's max_tokens, thinking off too
          preflight_retry: true    # optional: see qb_launch.py
          reply_window: 3600       # time-to-solution window, s per reply (default 3600): a reply not back by then
                                   # scores as wrong, is not retried, and is counted (caps.<task>.timeouts)
          concurrency: 16          # optional: QualityBench's --concurrency (requests in flight per harness)

The same pinned items, greedy decoding (seed 608) and scoring as QualityBench's published results, so runs
compare item by item: ``qualitybench compare <ref>.json <run>.json`` on the files this pass writes.
One ``eval_summary`` event per task and thinking mode (accuracy and its 95% Wilson interval), one for the
logprobs block (perplexity), next to the server's throughput passes.

Server requirements (QualityBench): ``logprobs`` needs vLLM's prompt_logprobs (``--max-logprobs 64``) and
no speculative decoding; ``bfcl`` needs the model's tool-call parser; the code tasks a container runtime
(docker or podman, see tools.py).
"""

from __future__ import annotations

import json
import os
import subprocess

from benchmate.benchserve import tools

SUITE = ["gsm8k", "ifeval", "mmlu_pro", "humaneval", "mbpp", "bfcl", "lcb", "logprobs", "needle"]


# evalplus sandbox (HumanEval+ / MBPP+ scoring). QualityBench pins ganler/evalplus by digest, an amd64-only
# image: on arm64 (DGX Spark) it fails with "exec format error". The same image, built locally for the
# machine's architecture from evalplus' own Dockerfile (python:3.11-slim, git, the evalplus version
# QualityBench locks).
EVALPLUS_VERSION = "0.3.1"
EVALPLUS_LOCAL_IMAGE = f"localhost/qualitybench-evalplus:{EVALPLUS_VERSION}-{{arch}}"
EVALPLUS_CONTAINERFILE = f"""FROM docker.io/library/python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir evalplus=={EVALPLUS_VERSION}
WORKDIR /app
CMD ["bash"]
"""


def evalplus_image() -> str | None:
    """The evalplus image to use instead of QualityBench's, or None to keep its own (x86_64)."""
    import platform

    if override := os.environ.get("MILABENCH_QB_EVALPLUS_IMAGE"):
        return override
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        return None
    return EVALPLUS_LOCAL_IMAGE.format(arch=machine)


def ensure_evalplus_image(image: str):
    runtime = tools.container_runtime()
    if runtime is None:
        print("[quality] no container runtime: HumanEval+ / MBPP+ will fail", flush=True)
        return
    if subprocess.run([runtime, "image", "inspect", image], capture_output=True).returncode == 0:
        return
    print(f"[quality] building {image} for this architecture", flush=True)
    subprocess.run([runtime, "build", "-t", image, "-f", "-", "."], input=EVALPLUS_CONTAINERFILE.encode(),
                   check=True, cwd=tools.root())


# QualityBench runs through qb_launch.py (in its own environment): it applies milabench's adjustments, each from
# an environment variable (see that file)
_LAUNCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qb_launch.py")


def _command(*argv) -> list[str]:
    return [tools.python("qualitybench"), _LAUNCH, *argv]


def _home() -> str:
    return os.path.join(tools.root(), "qualitybench-home")


def _env() -> dict:
    # QualityBench keeps its harness envs, datasets and timings here
    # QUALITYBENCH_CONTAINER: the code tasks' runtime (docker / podman), the one the other passes use
    env = {**os.environ, **tools.container_env(), "QUALITYBENCH_HOME": _home(), "QUALITYBENCH_CONFIRM": "never"}
    if image := evalplus_image():
        env["QB_EVALPLUS_IMAGE"] = image
    return env


def _tasks(spec: dict) -> list[str]:
    return list(spec.get("tasks") or SUITE)


def _server_section() -> tuple[str, dict]:
    """The benchmark's server backend and argv (shared section merged with the arch variant)."""
    try:
        cfg = json.loads(os.environ.get("MILABENCH_CONFIG", "{}"))
    except json.JSONDecodeError:
        return "vllm", {}
    arch = cfg.get("system", {}).get("arch", "cuda")
    shared = cfg.get("server") or {}
    variant = ((cfg.get("variants") or {}).get(arch) or {}).get("server") or {}
    argv = {**(shared.get("argv") or {}), **(variant.get("argv") or {})}
    backend = variant.get("backend") or shared.get("backend")
    if backend is None:
        backend = "llamacpp" if "llamacpp" in str(cfg.get("definition", "")) else "vllm"
    return backend, argv


def supported_tasks(tasks: list[str], backend: str, server_argv: dict) -> tuple[list[str], dict]:
    """Drop the tasks this server cannot run as configured; returns (tasks, {task: why})."""
    dropped = {}
    if "logprobs" in tasks:
        if backend not in ("vllm", "vllm-omni"):
            dropped["logprobs"] = f"needs vLLM's prompt_logprobs ({backend} server)"
        elif server_argv.get("--speculative-config") or server_argv.get("--speculative-algorithm"):
            dropped["logprobs"] = "prompt logprobs are not exact with speculative decoding"
        elif int(server_argv.get("--max-logprobs") or 0) < 64:
            dropped["logprobs"] = "server needs --max-logprobs 64"
    if "bfcl" in tasks and backend in ("vllm", "vllm-omni") and not server_argv.get("--tool-call-parser"):
        dropped["bfcl"] = "scores the server's tool-call parser: none configured (--tool-call-parser)"
    elif "bfcl" in tasks and backend in ("vllm", "vllm-omni") and not server_argv.get("--enable-auto-tool-choice"):
        dropped["bfcl"] = "vLLM refuses tool_choice auto without --enable-auto-tool-choice"
    if "needle" in tasks:
        context = server_argv.get("--max-model-len") or server_argv.get("--context-length")
        if context is not None and int(context) < 32768:
            dropped["needle"] = f"needs a 32k context (server: {context})"
    return [t for t in tasks if t not in dropped], dropped


def prepare(spec: dict):
    """Install QualityBench, its harness environments, datasets and the LiveCodeBench image."""
    tools.ensure("qualitybench")
    tasks = _tasks(spec)
    if (image := evalplus_image()) and {"humaneval", "mbpp"} & set(tasks):
        ensure_evalplus_image(image)
    subprocess.run(_command("setup", "--tasks", ",".join(tasks)), check=True, env=_env())


def default_deadline(margin: float = 900) -> str | None:
    """QualityBench's --deadline from the benchmark's max_duration (milabench kills the process then): it starts
    no task after the deadline and writes the results it has, cut tasks marked partial. Without it a slow run (a
    model that always reasons) loses every finished task when it is killed. The margin covers the task running
    at the deadline."""
    import datetime

    from benchmate.benchserve import _milabench_config

    import time

    limit = _milabench_config().get("max_duration")
    if not limit:
        return None
    now = time.time()
    # the limit runs from the benchmark's start (server and earlier passes included), not this pass's
    try:
        end = float(os.environ.get("MILABENCH_STARTED_AT", "")) + float(limit)
    except ValueError:
        end = now + float(limit)
    left = max(end - now, 0)
    at = now + max(left - margin, 0.5 * left)
    return datetime.datetime.fromtimestamp(at, datetime.timezone.utc).replace(microsecond=0).isoformat()


def run(bench_argv, spec: dict, *, flag, output_dir: str, name: str, push) -> list[dict]:
    host = flag(bench_argv, "--host", "127.0.0.1")
    port = flag(bench_argv, "--port", "8000")
    model = flag(bench_argv, "--served-model-name") or flag(bench_argv, "--model")
    tokenizer = flag(bench_argv, "--tokenizer") or flag(bench_argv, "--model")
    os.makedirs(output_dir, exist_ok=True)
    tasks, dropped = supported_tasks(_tasks(spec), *_server_section())
    for task, why in dropped.items():
        print(f"[quality] skipping {task}: {why}", flush=True)

    thinking = spec.get("thinking", ["off"])
    thinking = [thinking] if isinstance(thinking, (str, bool)) or thinking is None else list(thinking)
    # YAML reads a bare on / off as a boolean
    thinking = [("on" if m else "off") if isinstance(m, bool) else m for m in thinking]
    summaries = []
    deadline = spec.get("deadline") or default_deadline()
    for mode in thinking:
        out = os.path.join(output_dir, f"{name}.qualitybench" + (f".thinking-{mode}" if mode else "") + ".json")
        cmd = _command("run", "--endpoint", f"http://{host}:{port}/v1", "--model", model, "--tokenizer", tokenizer)
        cmd += [
               "--tasks", ",".join(tasks), "--out", out, "--yes", "--note", f"milabench={name}",
               # no time-estimate probe: it sends real requests (~30 min with a reasoning model) for a number
               # nobody reads inside a benchmark
               "--no-probe"]
        if spec.get("tier", "quick") == "quick":
            cmd.append("--quick")
        if mode:
            cmd += ["--thinking", mode]
            if mode == "on":
                cmd += ["--thinking-max-tokens", str(spec.get("thinking_max_tokens", 32768))]
        if deadline:
            cmd += ["--deadline", str(deadline)]
        if spec.get("concurrency"):
            cmd += ["--concurrency", str(int(spec["concurrency"]))]
        env = _env()
        if spec.get("max_tokens"):
            env["QB_MAX_TOKENS_FLOOR"] = str(int(spec["max_tokens"]))
        if spec.get("preflight_retry"):
            env["QB_PREFLIGHT_RETRY"] = "1"
        env["QB_REPLY_WINDOW"] = str(int(spec.get("reply_window") or 3600))
        print("QUALITY:", " ".join(cmd), {k: v for k, v in env.items() if k.startswith("QB_")}, flush=True)
        # exit 0: done; other codes: some tasks failed, the results file still holds the others
        proc = subprocess.run(cmd, env=env)
        if not os.path.exists(out):
            raise RuntimeError(f"qualitybench wrote no results (exit {proc.returncode})")
        summaries += summarize(json.load(open(out)), name=name, thinking=mode, path=out, push=push)
    return summaries


def summarize(results: dict, *, name: str, thinking, path: str | None, push) -> list[dict]:
    """eval_summary events of a QualityBench results file."""
    out = []
    # QualityBench files a thinking-on run under "off" when the model always reasons (gpt-oss): a run asked for
    # "on" whose file holds no "on" arm is that run
    mislabeled = thinking == "on" and not any(task.get("on") for task in results.get("tasks", []))
    for task in results.get("tasks", []):
        # a run has one arm, "off" / "on", or both when the file merges two runs
        for arm in ("off", "on"):
            if not (score := task.get(arm)):
                continue
            summary = {
                "name": name, "suite": "qualitybench", "task": task["id"], "label": task.get("label"),
                "thinking": "on" if mislabeled else arm, "correct": score["k"], "samples": score["n"],
                "scores": {"acc": score["acc"] / 100.0},
                "ci95": [c / 100.0 for c in score.get("ci", [])],
                "results": path,
            }
            print("[eval]", json.dumps(summary), flush=True)
            push(eval_summary=summary)
            out.append(summary)
    if logprobs := results.get("logprobs"):
        # one perplexity per text set: {"ppl-wiki": {"label": "wikitext-2", "ppl": 8.5, ...}, "ppl-c4": ..., ...}
        scores = {k: v for k, v in logprobs.items() if isinstance(v, (int, float))}
        scores.update({key: block["ppl"] for key, block in logprobs.items()
                       if isinstance(block, dict) and isinstance(block.get("ppl"), (int, float))})
        summary = {"name": name, "suite": "qualitybench", "task": "logprobs", "thinking": thinking,
                   "scores": scores, "results": path}
        print("[eval]", json.dumps(summary), flush=True)
        push(eval_summary=summary)
        out.append(summary)
    if needle := results.get("needle"):
        # {"summary": {"32768": 0.9, "131072": 0.7}, "items": [{"length", "depth", "correct", ...}, ...]}
        scores = {}
        for length, acc in (needle.get("summary") or {}).items():
            try:
                scores[f"acc_{int(length) // 1024}k"] = float(acc)
            except (TypeError, ValueError):
                continue
        items = needle.get("items") or []
        summary = {"name": name, "suite": "qualitybench-needle", "task": "needle_by_length", "thinking": thinking,
                   "correct": sum(1 for item in items if item.get("correct")), "samples": len(items),
                   "scores": scores, "results": path}
        print("[eval]", json.dumps(summary), flush=True)
        push(eval_summary=summary)
        out.append(summary)
    for caveat in results.get("caveats") or []:
        print("[quality caveat]", caveat, flush=True)
    return out
