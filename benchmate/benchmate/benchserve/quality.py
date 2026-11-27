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

Per task: ``tasks`` can also map each task to its own tier, or to its own options (tier, max_tokens,
thinking_max_tokens, reply_window, concurrency, preflight_retry), over the pass-level ones:

          tasks: {gsm8k: quick, ifeval: quick, lcb: {tier: full, max_tokens: 32768}}

Tasks with the same options run in one QualityBench call; each distinct set is its own call on the same server
(results ``<name>.qualitybench.<tier>.thinking-<mode>.json`` when there are several, the usual name otherwise).
Every eval_summary carries its ``tier``.

``quality_toggle:`` narrows a run from the command line without editing the config, e.g.
``--override '*.quality_toggle.tier=quick'`` (run.sh: QUALITY=probe): ``tier`` forces every task's tier,
``tasks`` and ``thinking`` keep only the listed ones (a mode the bench does not run is not added).

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


# options a task can set for itself (``tasks: {lcb: {tier: full, max_tokens: 32768}}``)
TASK_OPTIONS = ("tier", "max_tokens", "thinking_max_tokens", "reply_window", "concurrency", "preflight_retry")


def _as_list(value) -> list:
    return [value] if isinstance(value, (str, bool)) or value is None else list(value)


def _modes(value) -> list[str]:
    # YAML reads a bare on / off as a boolean
    return [("on" if m else "off") if isinstance(m, bool) else m for m in _as_list(value)]


def _task_options(tasks) -> dict[str, dict]:
    """{task: its own options}: a list gives none, a mapping a tier (str) or a dict of options per task."""
    if not isinstance(tasks, dict):
        return {t: {} for t in (tasks or SUITE)}
    out = {}
    for task, opts in tasks.items():
        if opts is None:
            opts = {}
        elif isinstance(opts, str):
            opts = {"tier": opts}
        unknown = set(opts) - set(TASK_OPTIONS)
        if unknown:
            raise ValueError(f"quality task {task}: unknown options {sorted(unknown)} (known: {TASK_OPTIONS})")
        out[task] = dict(opts)
    return out


def effective_spec(spec: dict, toggle: dict | None = None) -> tuple[dict, list[str]]:
    """The pass spec with the ``quality_toggle:`` applied. Returns ({task: options}, thinking modes); every task's
    options are complete (the pass-level ones filled in)."""
    toggle = toggle or {}
    defaults = {k: spec[k] for k in TASK_OPTIONS if k in spec}
    resolved = {t: {**defaults, **opts} for t, opts in _task_options(spec.get("tasks")).items()}
    for opts in resolved.values():
        opts.setdefault("tier", "quick")
    thinking = _modes(spec.get("thinking", ["off"]))
    if toggle.get("tier"):
        tier = "quick" if toggle["tier"] == "probe" else toggle["tier"]
        for opts in resolved.values():
            opts["tier"] = tier
    if toggle.get("tasks"):
        keep = set(_as_list(toggle["tasks"]))
        resolved = {t: o for t, o in resolved.items() if t in keep}
    if toggle.get("thinking"):
        keep = set(_modes(toggle["thinking"]))
        thinking = [m for m in thinking if m in keep]
    return resolved, thinking


def task_groups(resolved: dict[str, dict]) -> list[tuple[dict, list[str]]]:
    """Tasks with identical options, in suite order: one QualityBench call each."""
    groups: dict[tuple, tuple[dict, list[str]]] = {}
    for task, opts in resolved.items():
        key = tuple(sorted((k, json.dumps(v)) for k, v in opts.items()))
        groups.setdefault(key, (opts, []))[1].append(task)
    return list(groups.values())


def group_tags(groups) -> list[str]:
    """The results file tag of each group: its tier, numbered when two groups share one."""
    tiers = [opts["tier"] for opts, _ in groups]
    return [t if tiers.count(t) == 1 else f"{t}{tiers[:i + 1].count(t)}" for i, t in enumerate(tiers)]


def _bench_config() -> dict:
    try:
        return json.loads(os.environ.get("MILABENCH_CONFIG", "{}")) or {}
    except json.JSONDecodeError:
        return {}


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
    tasks = list(_task_options(spec.get("tasks")))
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
    resolved, thinking = effective_spec(spec, _bench_config().get("quality_toggle"))
    tasks, dropped = supported_tasks(list(resolved), *_server_section())
    for task, why in dropped.items():
        print(f"[quality] skipping {task}: {why}", flush=True)
    resolved = {t: o for t, o in resolved.items() if t in tasks}
    if not resolved or not thinking:
        print(f"[quality] nothing to run (tasks {list(resolved)}, thinking {thinking})", flush=True)
        return []

    groups = task_groups(resolved)
    tags = group_tags(groups) if len(groups) > 1 else [None]
    summaries = []
    deadline = spec.get("deadline") or default_deadline()
    for mode in thinking:
        for (opts, group), tag in zip(groups, tags):
            out = os.path.join(output_dir, f"{name}.qualitybench" + (f".{tag}" if tag else "")
                               + (f".thinking-{mode}" if mode else "") + ".json")
            cmd = _command("run", "--endpoint", f"http://{host}:{port}/v1", "--model", model, "--tokenizer", tokenizer)
            cmd += [
                   "--tasks", ",".join(group), "--out", out, "--yes", "--note", f"milabench={name}",
                   # no time-estimate probe: it sends real requests (~30 min with a reasoning model) for a number
                   # nobody reads inside a benchmark
                   "--no-probe"]
            if opts["tier"] == "quick":
                cmd.append("--quick")
            if mode:
                cmd += ["--thinking", mode]
                if mode == "on":
                    cmd += ["--thinking-max-tokens", str(opts.get("thinking_max_tokens", 32768))]
            if deadline:
                cmd += ["--deadline", str(deadline)]
            if opts.get("concurrency"):
                cmd += ["--concurrency", str(int(opts["concurrency"]))]
            env = _env()
            if opts.get("max_tokens"):
                env["QB_MAX_TOKENS_FLOOR"] = str(int(opts["max_tokens"]))
            if opts.get("preflight_retry"):
                env["QB_PREFLIGHT_RETRY"] = "1"
            env["QB_REPLY_WINDOW"] = str(int(opts.get("reply_window") or 3600))
            print("QUALITY:", " ".join(cmd), {k: v for k, v in env.items() if k.startswith("QB_")}, flush=True)
            # exit 0: done; other codes: some tasks failed, the results file still holds the others
            proc = subprocess.run(cmd, env=env)
            if not os.path.exists(out):
                # one group's failure (a crashed harness) does not cost the others
                print(f"[quality] qualitybench wrote no results for {group} (exit {proc.returncode})", flush=True)
                if len(groups) == 1:
                    raise RuntimeError(f"qualitybench wrote no results (exit {proc.returncode})")
                continue
            summaries += summarize(json.load(open(out)), name=name, thinking=mode, path=out, push=push,
                                   tier=opts["tier"])
    if not summaries:
        raise RuntimeError("qualitybench wrote no results")
    return summaries


def summarize(results: dict, *, name: str, thinking, path: str | None, push, tier: str | None = None) -> list[dict]:
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
            if tier:
                summary["tier"] = tier
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
