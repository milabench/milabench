"""``quality`` client passes: QualityBench (launch80/QualityBench) against the running server.

    passes:
      - name: quality
        quality:
          tier: quick              # quick (every tenth item) | full
          thinking: [off]          # one run per value: off / on (thinking models need one of them)
          tasks: [gsm8k, ifeval, mmlu_pro, humaneval, mbpp, bfcl, lcb, logprobs, needle]   # default suite
          thinking_max_tokens: 32768

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


def _home() -> str:
    return os.path.join(tools.root(), "qualitybench-home")


def _env() -> dict:
    # QualityBench keeps its harness envs, datasets and timings here
    # QUALITYBENCH_CONTAINER: the code tasks' runtime (docker / podman), the one the other passes use
    return {**os.environ, **tools.container_env(), "QUALITYBENCH_HOME": _home(), "QUALITYBENCH_CONFIRM": "never"}


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
    return variant.get("backend") or shared.get("backend", "vllm"), argv


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
    if "needle" in tasks:
        context = server_argv.get("--max-model-len") or server_argv.get("--context-length")
        if context is not None and int(context) < 32768:
            dropped["needle"] = f"needs a 32k context (server: {context})"
    return [t for t in tasks if t not in dropped], dropped


def prepare(spec: dict):
    """Install QualityBench, its harness environments, datasets and the LiveCodeBench image."""
    exe = tools.ensure("qualitybench")
    subprocess.run([exe, "setup", "--tasks", ",".join(_tasks(spec))], check=True, env=_env())


def run(bench_argv, spec: dict, *, flag, output_dir: str, name: str, push) -> list[dict]:
    exe = tools.executable("qualitybench")
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
    for mode in thinking:
        out = os.path.join(output_dir, f"{name}.qualitybench" + (f".thinking-{mode}" if mode else "") + ".json")
        cmd = [exe, "run", "--endpoint", f"http://{host}:{port}/v1", "--model", model, "--tokenizer", tokenizer,
               "--tasks", ",".join(tasks), "--out", out, "--yes", "--note", f"milabench={name}"]
        if spec.get("tier", "quick") == "quick":
            cmd.append("--quick")
        if mode:
            cmd += ["--thinking", mode]
            if mode == "on":
                cmd += ["--thinking-max-tokens", str(spec.get("thinking_max_tokens", 32768))]
        if deadline := spec.get("deadline"):
            cmd += ["--deadline", str(deadline)]
        print("QUALITY:", " ".join(cmd), flush=True)
        # exit 0: done; other codes: some tasks failed, the results file still holds the others
        proc = subprocess.run(cmd, env=_env())
        if not os.path.exists(out):
            raise RuntimeError(f"qualitybench wrote no results (exit {proc.returncode})")
        summaries += summarize(json.load(open(out)), name=name, thinking=mode, path=out, push=push)
    return summaries


def summarize(results: dict, *, name: str, thinking, path: str | None, push) -> list[dict]:
    """eval_summary events of a QualityBench results file."""
    out = []
    for task in results.get("tasks", []):
        # a run has one arm, "off" / "on", or both when the file merges two runs
        for arm in ("off", "on"):
            if not (score := task.get(arm)):
                continue
            summary = {
                "name": name, "suite": "qualitybench", "task": task["id"], "label": task.get("label"),
                "thinking": arm, "correct": score["k"], "samples": score["n"],
                "scores": {"acc": score["acc"] / 100.0},
                "ci95": [c / 100.0 for c in score.get("ci", [])],
                "results": path,
            }
            print("[eval]", json.dumps(summary), flush=True)
            push(eval_summary=summary)
            out.append(summary)
    if logprobs := results.get("logprobs"):
        summary = {"name": name, "suite": "qualitybench", "task": "logprobs", "thinking": thinking,
                   "scores": {k: v for k, v in logprobs.items() if isinstance(v, (int, float))}, "results": path}
        print("[eval]", json.dumps(summary), flush=True)
        push(eval_summary=summary)
        out.append(summary)
    for caveat in results.get("caveats") or []:
        print("[quality caveat]", caveat, flush=True)
    return out
