"""``swebench`` client passes: a coding agent (mini-swe-agent) solving SWE-bench Verified tasks against the
running server, scored by the SWE-bench harness.

    passes:
      - name: swebench
        swebench:
          tier: quick              # quick: 50 tasks | full: 281 (data/swebench_verified_arm64.json)
          workers: 4               # agent sessions in parallel against the server
          step_limit: 250
          temperature: 0.0
          config: swebench.yaml    # mini-swe-agent config: tool calls (needs the server's tool-call parser);
                                   # swebench_backticks.yaml: commands in the text, no parser
          eval_workers: 4

The tasks are the SWE-bench Verified ones with a published arm64 evaluation image, so the same subset runs
natively on aarch64 (DGX Spark) and on x86_64. A coding agent fails in ways one-shot benchmarks miss (a
quantized model repeating the same command until it runs out of steps), so besides the solve rate this
pass reports how each session ended and the agent time per solved task:

    solved / n (95% Wilson), submitted, step_limit (LimitsExceeded), looped (the same command 3+ times in a
    row, or at most 2 distinct commands in the last 10 steps of a session that hit the step limit), errors,
    steps per session, agent_hours (wall time of the agent run), solved_per_hour

One ``agent_summary`` event, and ``<name>.swebench/instances.jsonl`` (one line per task) for paired
comparisons across runs. Needs a container runtime on the client's machine: docker or podman
(MILABENCH_CONTAINER picks one, see tools.py).
"""

from __future__ import annotations

import glob
import json
import math
import os
import platform
import subprocess
import time
from collections import Counter

from benchmate.benchserve import tools

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "swebench_verified_arm64.json")
SOURCE = "SWE-bench/SWE-bench_Verified"


def arch() -> str:
    return "arm64" if platform.machine().lower() in ("aarch64", "arm64") else "x86_64"


def instance_ids(tier: str) -> list[str]:
    return json.load(open(DATA))[tier]


def image(instance_id: str, arch_name: str) -> str:
    # fully qualified: podman refuses short names by default, docker accepts both
    key = instance_id.replace("__", "_1776_")
    return f"docker.io/swebench/sweb.eval.{arch_name}.{key}:latest".lower()


def dataset_dir(tier: str) -> str:
    return os.path.join(tools.root(), "swebench-data", f"verified-{tier}-{arch()}")


_BUILD = r'''
import json, sys
from datasets import load_dataset
source, out, arch, ids = sys.argv[1], sys.argv[2], sys.argv[3], set(json.loads(sys.argv[4]))
rows = [dict(r) for r in load_dataset(source, split="test") if r["instance_id"] in ids]
missing = ids - {r["instance_id"] for r in rows}
assert not missing, f"not in {source}: {sorted(missing)}"
with open(out, "w") as f:
    for r in sorted(rows, key=lambda r: r["instance_id"]):
        name = "docker.io/swebench/sweb.eval.%s.%s:latest" % (arch, r["instance_id"].replace("__", "_1776_"))
        # `image`: the SWE-bench harness; `image_name`: mini-swe-agent
        r["image"] = r["image_name"] = name.lower()
        f.write(json.dumps(r) + "\n")
print(f"{len(rows)} tasks -> {out}")
'''


def prepare(spec: dict):
    """Install the agent and the harness, write the task list for this arch, pull the task images."""
    tools.ensure("mini-swe-agent")
    tools.ensure("swebench")
    tier = spec.get("tier", "quick")
    ids = instance_ids(tier)
    folder = dataset_dir(tier)
    os.makedirs(folder, exist_ok=True)
    out = os.path.join(folder, "test.jsonl")
    if not os.path.exists(out):
        subprocess.run([tools.python("swebench"), "-c", _BUILD, SOURCE, out, arch(), json.dumps(ids)], check=True)
    runtime = tools.container_runtime()
    if runtime is None:
        print("[swebench] no container runtime here (docker / podman): the images are pulled on first use", flush=True)
        return
    for instance_id in ids:
        subprocess.run([runtime, "pull", "-q", image(instance_id, arch())], check=True)


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [max(0.0, centre - half), min(1.0, centre + half)]


def commands(trajectory: dict) -> list[str]:
    """The shell commands of a session, in order (tool calls or text actions alike)."""
    out = []
    for message in trajectory.get("messages", []):
        if message.get("role") != "assistant":
            continue
        for action in (message.get("extra") or {}).get("actions") or []:
            if command := action.get("command"):
                out.append(" ".join(command.split()))
    return out


def loop_stats(cmds: list[str], hit_limit: bool) -> dict:
    longest, run = 0, 0
    for i, c in enumerate(cmds):
        run = run + 1 if i and c == cmds[i - 1] else 1
        longest = max(longest, run)
    tail_distinct = len(set(cmds[-10:])) if len(cmds) >= 10 else None
    looped = longest >= 3 or (hit_limit and tail_distinct is not None and tail_distinct <= 2)
    return {"longest_repeat": longest, "tail_distinct": tail_distinct, "looped": looped}


def run(bench_argv, spec: dict, *, flag, output_dir: str, name: str, push) -> dict:
    tier = spec.get("tier", "quick")
    folder = dataset_dir(tier)
    if not os.path.exists(os.path.join(folder, "test.jsonl")):
        raise FileNotFoundError(f"{folder}/test.jsonl missing: run the benchmark's prepare first")

    host = flag(bench_argv, "--host", "127.0.0.1")
    port = flag(bench_argv, "--port", "8000")
    model = flag(bench_argv, "--served-model-name") or flag(bench_argv, "--model")
    workers = int(spec.get("workers", 4))
    trajs = os.path.join(output_dir, "trajectories")
    os.makedirs(trajs, exist_ok=True)

    agent = [
        tools.executable("mini-swe-agent"), "swebench",
        "--subset", folder, "--split", "test", "-o", trajs, "-w", str(workers),
        "-c", spec.get("config", "swebench.yaml"),
        "-c", f"model.model_name=hosted_vllm/{model}",
        "-c", f"model.model_kwargs.api_base=http://{host}:{port}/v1",
        "-c", f"model.model_kwargs.temperature={spec.get('temperature', 0.0)}",
        "-c", "model.cost_tracking=ignore_errors",
        "-c", f"agent.step_limit={spec.get('step_limit', 250)}",
        "-c", "agent.cost_limit=0",
    ]
    runtime = tools.container_runtime()
    if runtime is None:
        raise RuntimeError("SWE-bench needs a container runtime on this machine (docker or podman)")
    env = {**os.environ, **tools.container_env(runtime), "MSWEA_COST_TRACKING": "ignore_errors",
           "OPENAI_API_KEY": "local", "HOSTED_VLLM_API_KEY": "local"}
    print(f"AGENT ({runtime}):", " ".join(agent), flush=True)
    start = time.time()
    subprocess.run(agent, check=True, env=env)
    agent_seconds = time.time() - start

    preds = os.path.join(trajs, "preds.json")
    run_id = f"{name}".replace("/", "_")
    harness = [tools.executable("swebench"), "eval", os.path.join(folder, "test.jsonl"), "-p", preds,
               "-r", run_id, "-j", str(spec.get("eval_workers", workers))]
    print("SWEBENCH:", " ".join(harness), flush=True)
    # the harness drives the runtime through docker's API: podman serves it on its socket
    with tools.container_api(runtime) as api:
        subprocess.run(harness, check=True, cwd=output_dir, env={**env, **api.env})
    reports = sorted(glob.glob(os.path.join(output_dir, f"*.{run_id}.json")), key=os.path.getmtime)
    if not reports:
        raise RuntimeError(f"no SWE-bench report *.{run_id}.json in {output_dir}")
    report = json.load(open(reports[-1]))
    resolved = set(report.get("resolved_ids", []))

    per_instance = []
    for instance_id in instance_ids(tier):
        path = os.path.join(trajs, instance_id, f"{instance_id}.traj.json")
        trajectory = json.load(open(path)) if os.path.exists(path) else {}
        status = (trajectory.get("info") or {}).get("exit_status") or "missing"
        cmds = commands(trajectory)
        per_instance.append({
            "instance_id": instance_id, "resolved": instance_id in resolved, "exit_status": status,
            "steps": len(cmds), **loop_stats(cmds, status == "LimitsExceeded"),
        })
    with open(os.path.join(output_dir, "instances.jsonl"), "w") as f:
        for record in per_instance:
            f.write(json.dumps(record) + "\n")

    n = len(per_instance)
    solved = sum(r["resolved"] for r in per_instance)
    statuses = Counter(r["exit_status"] for r in per_instance)
    summary = {
        "name": name, "suite": "swebench-verified", "tier": tier, "arch": arch(), "samples": n,
        "solved": solved, "scores": {"resolved": solved / n if n else 0.0}, "ci95": wilson(solved, n),
        "submitted": statuses.get("Submitted", 0),
        "step_limit": statuses.get("LimitsExceeded", 0),
        "looped": sum(r["looped"] for r in per_instance),
        "errors": n - statuses.get("Submitted", 0) - statuses.get("LimitsExceeded", 0),
        "exit_statuses": dict(statuses),
        "mean_steps": sum(r["steps"] for r in per_instance) / n if n else 0.0,
        "agent_hours": agent_seconds / 3600,
        "solved_per_hour": solved / (agent_seconds / 3600) if agent_seconds else 0.0,
        "workers": workers, "step_limit_setting": int(spec.get("step_limit", 250)),
        "report": reports[-1],
    }
    print("[agent]", json.dumps(summary), flush=True)
    push(agent_summary=summary)
    return summary
