"""DRAFT (2026-11-16, not yet run against a real server) -- ``terminalbench`` client passes: Terminal-Bench 2.0
(89 terminal tasks in containers) solved by an agent harness against the running server, run and scored by Harbor.

    passes:
      - name: terminalbench
        terminalbench:
          tasks: full                 # full: all 89 | smoke: SMOKE (3 easy tasks) | [task, ...]
          harness: terminus-2         # any `harbor run -a` agent, or module.path:Class
          harness_kwargs: {}          # extra --ak key=value over the defaults below
          harness_env: {}             # extra --ae KEY=VALUE (agents installed in the task container)
          attempts: 1                 # harbor -k; the leaderboard uses 5
          workers: 4                  # trials in parallel (harbor -n)
          timeout_multiplier: 1.0     # agent timeouts; 1.0 = leaderboard rules
          verifier_timeout_multiplier: 1.0
          thinking: null              # null: server default | true | false (chat_template_kwargs.enable_thinking)
          temperature: 0.6
          max_tokens: 16384           # per reply (Terminus-2 sends none otherwise)
          context: 65536              # = the server's --max-model-len
          images: ghcr.io/milabench/terminal-bench   # null: upstream docker.io/alexgshaw (amd64 only)
          pull: false                 # prepare pulls every task image (~100 GB); false: on first use
          runner: null                # null: this machine (the client) | {host, user, key, port, workdir}
          endpoint: null              # the server as the runner reaches it; default http://<--host>:<--port>/v1

Images: github.com/milabench/container rebuilds every task natively for arm64 and publishes
<images>/<task>:<tag>, a multi-arch image (upstream amd64 + its arm64), so the tasks run natively on aarch64
(DGX Spark) and x86_64 alike. prepare() points each task's docker_image there.

Harness: Terminus-2 (Harbor's reference agent, the leaderboard's) runs here and drives a tmux session in the task
container; it reaches the server directly. Agents installed in the container (mini-swe-agent, ...) get the
endpoint through OPENAI_BASE_URL and must be able to reach it from inside the container.

Runner: Harbor and the task containers run on the client's machine (the milabench main node) by default. With
``runner.host`` they run on that host over ssh (Harbor installed there by prepare); the job folder is copied back.
The server must then listen on an address that host reaches (vLLM --host 0.0.0.0), named by ``endpoint``.

Reports, besides the solve rate: agent timeouts, context overflows, errors per type, pass@k over the attempts, the
rate without the tasks at risk on arm64, agent hours. One ``agent_summary`` event, and
``<name>.terminalbench/trials.jsonl`` (one line per trial) for paired comparisons across runs.
Needs podman on the runner (Harbor --env podman; its API socket is served for the pass if needed).
"""

from __future__ import annotations

import glob
import json
import os
import platform
import re
import shlex
import subprocess
import threading
import time
from collections import Counter

from benchmate.benchserve import tools

# TODO when wired in: move to tools.TOOLS
tools.TOOLS.setdefault("harbor", {"spec": "harbor==0.26.0", "bin": "harbor"})

DATASET = "terminal-bench@2.0"
IMAGES = "ghcr.io/milabench/terminal-bench"
SMOKE = ["fix-git", "overfull-hbox", "cobol-modernization"]
# rebuilt for arm64 but at risk there (x86 guests under TCG, very slow builds or huge images): run and reported
ARM_RISK = {"caffe-cifar-10", "install-windows-3.11", "qemu-startup", "qemu-alpine-ssh",
            "custom-memory-heap-crash", "mteb-leaderboard", "mteb-retrieve"}
# harnesses that talk to the server through LiteLLM on this side (api_base / model_info / llm_call_kwargs)
LITELLM_HARNESSES = {"terminus-2"}


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    from benchmate.benchserve.swebench import wilson as _wilson

    return _wilson(k, n, z)


# --- where Harbor runs --------------------------------------------------------------------------------------

class Runner:
    """The client's machine, or a host reached over ssh: commands, file copies, paths."""

    def __init__(self, spec: dict | None):
        self.spec = spec or {}
        self.host = self.spec.get("host")

    @property
    def remote(self) -> bool:
        return bool(self.host)

    @property
    def target(self) -> str:
        return f"{self.spec['user']}@{self.host}" if self.spec.get("user") else self.host

    def ssh(self) -> list[str]:
        cmd = ["ssh", "-oBatchMode=yes", "-oCheckHostIP=no", "-p", str(self.spec.get("port", 22))]
        if key := self.spec.get("key"):
            cmd += ["-i", os.path.expanduser(key)]
        return cmd

    def workdir(self) -> str:
        if self.remote:
            return self.spec.get("workdir", ".cache/milabench/terminalbench")  # relative: the remote home
        return os.path.join(tools.root(), "terminalbench-data")

    def harbor(self) -> str:
        return f"{self.workdir()}/harbor/bin/harbor" if self.remote else tools.executable("harbor")

    def python(self) -> str:
        return f"{self.workdir()}/harbor/bin/python" if self.remote else tools.python("harbor")

    def run(self, argv: list[str], env: dict | None = None, check: bool = True, stdin: bytes | None = None):
        if not self.remote:
            return subprocess.run(argv, env={**os.environ, **(env or {})}, check=check, input=stdin)
        line = " ".join([*(f"{k}={shlex.quote(str(v))}" for k, v in (env or {}).items()), shlex.join(argv)])
        return subprocess.run([*self.ssh(), self.target, line], check=check, input=stdin)

    def fetch(self, remote_dir: str, local_dir: str):
        os.makedirs(local_dir, exist_ok=True)
        subprocess.run(["rsync", "-a", "-e", shlex.join(self.ssh()), f"{self.target}:{remote_dir}/", local_dir + "/"],
                       check=True)


# --- tasks ---------------------------------------------------------------------------------------------------

_REWRITE = r'''
"""Point every task's docker_image at the rebuilt multi-arch images; list the tasks whose image is missing."""
import json, os, re, subprocess, sys
folder, registry, pull = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
missing = []
for name in sorted(os.listdir(folder)):
    toml = os.path.join(folder, name, "task.toml")
    if not os.path.exists(toml):
        continue
    text = open(toml).read()
    current = re.search(r'^docker_image\s*=\s*"([^"]+)"', text, re.M).group(1)
    image = f"{registry}/{name}:{current.rsplit(':', 1)[1]}"
    if pull:
        ok = subprocess.run(["podman", "pull", "-q", image], capture_output=True).returncode == 0
    else:
        ok = subprocess.run(["podman", "manifest", "inspect", image], capture_output=True).returncode == 0
    if not ok:
        missing.append(name)
        continue
    open(toml, "w").write(re.sub(r'^docker_image\s*=.*$', f'docker_image = "{image}"', text, flags=re.M))
json.dump(missing, open(os.path.join(folder, "missing_images.json"), "w"))
print(f"[terminalbench] {len(missing)} task images missing from {registry}: {missing}", flush=True)
'''


def tasks_dir(runner: Runner) -> str:
    return f"{runner.workdir()}/tasks/terminal-bench"


def selected(spec: dict, available: list[str] | None = None) -> list[str] | None:
    """Task names to run; None: all of them."""
    sel = spec.get("tasks", "full")
    if isinstance(sel, list):
        return sel
    if sel == "smoke":
        return SMOKE
    if sel == "full":
        return available
    raise ValueError(f"terminalbench.tasks: full | smoke | [task, ...], not {sel!r}")


def prepare(spec: dict):
    """Install Harbor, download the tasks and point them at the multi-arch images (on the runner)."""
    runner = Runner(spec.get("runner"))
    work = runner.workdir()
    if runner.remote:
        install = (f"mkdir -p {work} && cd {work} && (test -x harbor/bin/harbor || "
                   f"(uv venv -p {tools.PYTHON} harbor && uv pip install --python harbor/bin/python "
                   f"{shlex.quote(tools.TOOLS['harbor']['spec'])}))")
        runner.run(["sh", "-c", install])
    else:
        tools.ensure("harbor")
        os.makedirs(work, exist_ok=True)
    runner.run(["sh", "-c", f"test -d {tasks_dir(runner)} || {shlex.quote(runner.harbor())} download {DATASET} "
                            f"-o {work}/tasks"])
    if registry := spec.get("images", IMAGES):
        script = f"{work}/rewrite_images.py"
        if runner.remote:
            runner.run(["sh", "-c", f"cat > {script}"], stdin=_REWRITE.encode())
        else:
            with open(script, "w") as f:
                f.write(_REWRITE)
        runner.run([runner.python(), script, tasks_dir(runner), registry, "1" if spec.get("pull") else "0"])


# --- run -----------------------------------------------------------------------------------------------------

def harness_args(spec: dict, endpoint: str, served: str) -> tuple[list[str], dict]:
    """``harbor run`` arguments for the agent and the model, and the environment Harbor runs in."""
    harness = spec.get("harness", "terminus-2")
    env = {"HOSTED_VLLM_API_KEY": "local", "OPENAI_API_KEY": "local"}
    kwargs: dict = {}
    if harness in LITELLM_HARNESSES:
        max_tokens, context = int(spec.get("max_tokens", 16384)), int(spec.get("context", 65536))
        call: dict = {"max_tokens": max_tokens}
        if spec.get("thinking") is not None:
            call["extra_body"] = {"chat_template_kwargs": {"enable_thinking": bool(spec["thinking"])}}
        # hosted_vllm: LiteLLM needs the limits to summarize before the context overflows (no model_info: 1M)
        kwargs = {"api_base": endpoint, "temperature": spec.get("temperature", 0.6),
                  "model_info": {"max_input_tokens": context - max_tokens, "max_output_tokens": max_tokens,
                                 "input_cost_per_token": 0, "output_cost_per_token": 0},
                  "llm_call_kwargs": call}
        model = f"hosted_vllm/{served}"
    else:
        env.update({"OPENAI_BASE_URL": endpoint, "OPENAI_API_BASE": endpoint})
        model = f"openai/{served}"
    kwargs.update(spec.get("harness_kwargs") or {})
    env.update({k: str(v) for k, v in (spec.get("harness_env") or {}).items()})
    argv = ["-a", harness, "-m", model]
    for k, v in kwargs.items():
        argv += ["--ak", f"{k}={v if isinstance(v, str) else json.dumps(v)}"]
    if harness not in LITELLM_HARNESSES:
        for k, v in env.items():
            argv += ["--ae", f"{k}={v}"]
    return argv, env


TASK_IMAGE = "terminal-bench"  # every task image: ghcr.io/milabench/terminal-bench/<task> (and its local builds)


def _task_containers() -> set[str]:
    """IDs of the podman containers running a Terminal-Bench task image, up or exited."""
    try:
        out = subprocess.run(["podman", "ps", "-a", "--format", "{{.ID}} {{.Image}}"], capture_output=True,
                             text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return set()
    return {line.split()[0] for line in out.splitlines() if TASK_IMAGE in line}


def _remove_containers(ids: set[str]) -> None:
    if ids:
        subprocess.run(["podman", "rm", "-f", "-t", "5", *sorted(ids)], capture_output=True, timeout=300)


def run(bench_argv, spec: dict, *, flag, output_dir: str, name: str, push) -> dict:
    runner = Runner(spec.get("runner"))
    host = flag(bench_argv, "--host", "127.0.0.1")
    port = flag(bench_argv, "--port", "8000")
    served = flag(bench_argv, "--served-model-name") or flag(bench_argv, "--model")
    endpoint = spec.get("endpoint") or f"http://{host}:{port}/v1"
    if runner.remote and re.match(r"https?://(127\.|localhost|0\.0\.0\.0)", endpoint):
        raise ValueError("terminalbench: a remote runner needs `endpoint`, an address of the server it can reach")

    folder = tasks_dir(runner)
    missing: list[str] = []
    if not runner.remote:
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"{folder} missing: run the benchmark's prepare first")
        if os.path.exists(os.path.join(folder, "missing_images.json")):
            missing = json.load(open(os.path.join(folder, "missing_images.json")))
    names = selected(spec)

    job = name.replace("/", "_")
    jobs_dir = f"{runner.workdir()}/jobs" if runner.remote else output_dir
    h_argv, env = harness_args(spec, endpoint, served)
    cmd = [runner.harbor(), "run", "-p", folder, "--env", "podman", *h_argv,
           "-n", str(spec.get("workers", 4)), "-k", str(spec.get("attempts", 1)),
           "--agent-timeout-multiplier", str(spec.get("timeout_multiplier", 1.0)),
           "--verifier-timeout-multiplier", str(spec.get("verifier_timeout_multiplier", 1.0)),
           "-o", jobs_dir, "--job-name", job, "-y", "-q"]
    for task in names or []:
        if task not in missing:
            cmd += ["-i", task]
    print(f"HARBOR ({runner.host or 'local'}):", shlex.join(cmd), flush=True)
    start = time.time()
    if runner.remote:
        runner.run(cmd, env=env, check=False)
        runner.fetch(f"{jobs_dir}/{job}", os.path.join(output_dir, job))
    else:
        os.makedirs(output_dir, exist_ok=True)
        # Harbor's task containers outlive a cancelled job (its server killed, the job stopped): 4 kept building for
        # 10 h under the next models' servers and ran the machine out of memory (2026-11-24). Whatever task
        # containers this pass started are removed when it ends, however it ends (SIGTERM unwinds like Ctrl-C)
        before = _task_containers()
        import signal

        def on_term(signum, frame):
            raise SystemExit(128 + signum)

        previous = signal.signal(signal.SIGTERM, on_term) if threading.current_thread() is threading.main_thread() else None
        try:
            # docker compose (under podman compose) talks to podman's API socket: served for the pass if needed
            with tools.container_api("podman") as api:
                runner.run(cmd, env={**tools.container_env("podman"), **env, **api.env}, check=False)
        finally:
            left = _task_containers() - before
            if left:
                print(f"terminalbench: removing {len(left)} task container(s) left behind", flush=True)
            _remove_containers(left)
            if previous is not None:
                signal.signal(signal.SIGTERM, previous)
    hours = (time.time() - start) / 3600
    summary = summarize(spec, os.path.join(output_dir, job), name, hours, missing, runner.host or "local")
    print("[agent]", json.dumps(summary), flush=True)
    push(agent_summary=summary)
    return summary


def summarize(spec: dict, job_dir: str, name: str, hours: float, missing: list[str], runner: str) -> dict:
    trials = []
    for path in sorted(glob.glob(os.path.join(job_dir, "*", "result.json"))):
        t = json.load(open(path))
        rewards = (t.get("verifier_result") or {}).get("rewards") or {}
        agent = t.get("agent_result") or {}
        trials.append({"task": t["task_name"], "trial": t["trial_name"],
                       "reward": float(rewards.get("reward", 0.0)),
                       "exception": (t.get("exception_info") or {}).get("exception_type"),
                       "arm_risk": t["task_name"] in ARM_RISK,
                       "n_input_tokens": agent.get("n_input_tokens"), "n_output_tokens": agent.get("n_output_tokens")})
    out = os.path.join(os.path.dirname(job_dir), "trials.jsonl")
    with open(out, "w") as f:
        for r in trials:
            f.write(json.dumps(r) + "\n")

    n = len(trials)
    solved = sum(r["reward"] >= 1.0 for r in trials)
    errors = Counter(r["exception"] for r in trials if r["exception"])
    by_task: dict[str, list[bool]] = {}
    for r in trials:
        by_task.setdefault(r["task"], []).append(r["reward"] >= 1.0)
    safe = [r for r in trials if not r["arm_risk"]]
    return {
        "name": name, "suite": "terminal-bench-2.0", "harness": spec.get("harness", "terminus-2"),
        "arch": platform.machine(), "runner": runner, "samples": n, "tasks": len(by_task), "solved": solved,
        "scores": {"resolved": solved / n if n else 0.0}, "ci95": wilson(solved, n),
        "pass_at_k": sum(any(v) for v in by_task.values()) / len(by_task) if by_task else 0.0,
        "resolved_excl_arm_risk": sum(r["reward"] >= 1.0 for r in safe) / len(safe) if safe else 0.0,
        "agent_timeouts": errors.get("AgentTimeoutError", 0),
        "context_exceeded": errors.get("ContextLengthExceededError", 0) + errors.get("ContextWindowExceededError", 0),
        "errors": dict(errors), "skipped_missing_images": missing, "agent_hours": hours,
        "settings": {k: spec.get(k) for k in ("tasks", "attempts", "workers", "timeout_multiplier", "thinking",
                                              "temperature", "max_tokens", "context", "harness_kwargs", "images")},
        "trials": out,
    }
