"""GLM-5.3 GRPO with slime (Megatron-LM trainer, SGLang rollouts, Ray), colocated on 32 nodes: one row per job, per step, per
node (host memory) and per visible engine decode line, from the run directory under ``data/raw/trillium/glm-rl/runs/``
(``analyze.py collect``: ``$SCRATCH/trillium/new/glm-rl/runs/``, GLM-RL-03 only).

Everything comes from the run's ``.0.data`` file: the command line (``start`` event), the node list (``config``), slime's
log lines (``line`` events: per-step ``perf N`` dicts of the rollout manager and of the trainer, ``rollout N`` and ``step N``
dicts, ``Timer update_weights end``, ``Memory-Usage`` snapshots, the trainer's micro-batch progress bar, SGLang's ``Decode
batch`` lines), the reward (``data`` events) and milabench's GPU monitor of the first node (``gpudata``).

Per step (slime's own timers, ``perf N``): ``rollout_s`` = ``perf/rollout_time``; ``train_s`` = ``perf/train_time``; ``sync_s`` =
``perf/update_weights_time``, the weight sync before the step's rollout; ``step_s`` = ``perf/step_time`` = ``train_wait_time`` +
``train_time``; ``rest_s`` = step - rollout - training - sync (the trainer's offload and onload, the engines' onload, data).
Step 0 is the warm-up. Training tokens/s = slime's ``actor_train_tok_per_s``, already global (prompt plus response tokens of
all samples over ``actor_train_time``). Log lines carry Trillium's local time (America/Toronto), as sacct.

Ray prints a line once and counts its repeats across the cluster ("[repeated Nx]"), so the ``Decode batch`` and
``Memory-Usage`` lines are a sample: one rank's line per repeat window, not every rank's. Ray names no address for actors on
its head node, the job's first node: those lines are that node's.

Stages: Megatron orders ranks TP, CP, EP/DP, then PP, so with 256 GPUs and PP8 ranks 32 s to 32 s + 31 are stage s. Engines: slime
takes the nodes for its engines in IP order, 8 per 64-GPU engine (``ray/placement_group.py``, ``engine_group.py``; README
``trillium-new`` 8f931f02), so a decode line's engine follows from its node's address.
"""

from __future__ import annotations

import ast
import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

# The 32-node row (REQUESTS.md GLM-RL-03). The 1-, 2-, 4- and 8-node entries are bring-up and tests: not read. GLM-RL-01
# (1265179) failed at engine load and was withdrawn; its directory is not collected.
SCALE_ENTRIES = ("slime-glm5-grpo",)
REQUESTS = {"1266065": "GLM-RL-03"}
LOG_TZ = ZoneInfo("America/Toronto")
SLOW_DECODE = 7.5  # tokens/s per sequence: the engines' decode lines fall at 7.6-10.1 or 5.8-6.1 (GLM-RL-03), nothing between

_TS = r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\]"
_PERF_ROLLOUT = re.compile(_TS + r" rollout_metrics\.py:\d+ - perf (\d+): (\{.*\})")
_PERF_TRAIN = re.compile(_TS + r" train_metric_utils\.py:\d+ - perf (\d+): (\{.*\})")
_ROLLOUT = re.compile(_TS + r" train_metric_utils\.py:\d+ - rollout (\d+): (\{.*\})")
_STEP = re.compile(_TS + r" model\.py:\d+ - step (\d+): (\{.*\})")
_SYNC = re.compile(_TS + r" timer\.py:\d+ - Timer update_weights end \(elapsed: ([\d.]+)s\)")
_ENGINE_UP = re.compile(_TS + r" Engine startup timings \(s\): load_weight=([\d.]+)")
_MEMORY = re.compile(r"\(\w+ pid=\d+(?:, ip=([\d.]+))?\).*?" + _TS + r" memory_utils\.py:\d+ - \[Rank (\d+)\] Memory-Usage ([\w ]+): (\{.*?\})")
_MICRO = re.compile(r"actor train:.*?\| *\d+/(\d+) \[")
_COMPILE = re.compile(r"TileLang completes to compile kernel(?:.*?repeated (\d+)x)?")
_DECODE = re.compile(r"\(SGLangEngine pid=\d+(?:, ip=([\d.]+))?\).*?\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) DP(\d+) TP\d+ EP\d+\] Decode batch, "
                     r"#running-req: (\d+), #token: (\d+), .*?gen throughput \(token/s\): ([\d.]+), #queue-req: (\d+)(?:.*?repeated (\d+)x)?")


def _local(ts: str) -> float:
    return datetime.fromisoformat(ts).replace(tzinfo=LOG_TZ).timestamp()


def _dict(text: str) -> dict:
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return {}


def _argv(command: list) -> dict:
    out, i = {}, 0
    while i < len(command):
        a = command[i]
        if isinstance(a, str) and a.startswith("--"):
            nxt = command[i + 1] if i + 1 < len(command) else None
            if isinstance(nxt, str) and not nxt.startswith("--"):
                out[a], i = nxt, i + 2
                continue
            out[a] = True
        i += 1
    return out


def read_run(path: Path) -> dict:
    """Facts of one run's .0.data file: argv, nodes, per-step dicts, syncs, memory snapshots, micro-batches, decode lines, GPU monitor."""
    d = {"argv": {}, "nodes": [], "rollout": {}, "train": {}, "rollout_stats": {}, "step": {}, "syncs": [], "engines_up": [],
         "memory": [], "micro": {}, "compiles": {}, "decode": [], "reward": [], "gpu": []}
    step = None  # the step whose rollout ended last: the trainer's progress bar that follows belongs to it
    with open(path) as f:
        for line in f:
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            ev, x = e.get("event"), e.get("data")
            if ev == "start" and isinstance(x, dict):
                d["argv"] = _argv(x.get("command") or [])
            elif ev == "config" and isinstance(x, dict):
                d["nodes"] = [(n["hostname"], n["ip"]) for n in (x.get("system") or {}).get("nodes", [])]
            elif ev == "data" and isinstance(x, dict):
                if "rollout_raw_reward" in x:
                    d["reward"].append(x["rollout_raw_reward"])
                elif "gpudata" in x:
                    d["gpu"].append((x["time"], max(g["memory"][0] for g in x["gpudata"].values()) / 1024))
            elif ev == "line" and isinstance(x, str):
                if "Decode batch" in x:
                    if (m := _DECODE.search(x)):
                        d["decode"].append({"time": _local(m[2]), "ip": m[1], "dp_rank": int(m[3]), "running": int(m[4]), "kv_tokens": int(m[5]),
                                            "gen_tok_s": float(m[6]), "queue": int(m[7]), "repeated": int(m[8] or 1)})
                elif (m := _PERF_ROLLOUT.search(x)):
                    step = int(m[2])
                    d["rollout"][step] = {"end": _local(m[1]), **_dict(m[3])}
                elif (m := _PERF_TRAIN.search(x)):
                    d["train"][int(m[2])] = {"end": _local(m[1]), **_dict(m[3])}
                elif (m := _ROLLOUT.search(x)):
                    d["rollout_stats"][int(m[2])] = _dict(m[3])
                elif (m := _STEP.search(x)):
                    d["step"][int(m[2])] = _dict(m[3])
                elif (m := _SYNC.search(x)):
                    d["syncs"].append((_local(m[1]), float(m[2])))
                elif (m := _ENGINE_UP.search(x)):
                    d["engines_up"].append(_local(m[1]))
                elif (m := _MEMORY.search(x)):
                    d["memory"].append({"ip": m[1], "time": _local(m[2]), "rank": int(m[3]), "phase": m[4].strip(), **_dict(m[5])})
                elif step is not None and (m := _MICRO.search(x)):
                    d["micro"][step] = max(d["micro"].get(step, 0), int(m[1]))
                elif step is not None and (m := _COMPILE.search(x)):
                    d["compiles"][step] = d["compiles"].get(step, 0) + 1 + int(m[1] or 0)  # the line and the repeats Ray folded into it
    return d


def _int(argv: dict, key: str, default=None):
    v = argv.get(key, default)
    return int(v) if v is not None and v is not True else default


def build_run(run_dir: Path, job_start: datetime | None) -> tuple[dict, list[dict], list[dict], list[dict]]:
    """(run, steps, hosts, decode) of one run directory <job name>.<job>.<entry>."""
    entry = run_dir.name.split(".")[-1]
    job = run_dir.name.split(".")[-2]
    d = read_run(run_dir / f"{entry}.0.data")
    a = d["argv"]
    gpus = _int(a, "--actor-num-nodes", 0) * _int(a, "--actor-num-gpus-per-node", 0)
    tp, pp, cp, ep = (_int(a, k, 1) for k in ("--tensor-model-parallel-size", "--pipeline-model-parallel-size", "--context-parallel-size",
                                               "--expert-model-parallel-size"))
    layers, first, last = (_int(a, k) for k in ("--num-layers", "--decoder-first-pipeline-num-layers", "--decoder-last-pipeline-num-layers"))
    engine_gpus = _int(a, "--rollout-num-gpus-per-engine")
    batch = _int(a, "--global-batch-size")
    run = {"entry": entry, "job": job, "request": REQUESTS.get(job, ""), "nodes": _int(a, "--actor-num-nodes"), "gpus": gpus,
           "hosts": ";".join(h for h, _ in d["nodes"]), "colocate": bool(a.get("--colocate")),
           "tp": tp, "pp": pp, "cp": cp, "ep": ep, "dp": gpus // (tp * pp * cp) if gpus else None, "layers": layers,
           "first_stage_layers": first, "last_stage_layers": last,
           "middle_stage_layers": (layers - first - last) / (pp - 2) if layers and first and last and pp > 2 else None,
           "engines": gpus // engine_gpus if gpus and engine_gpus else None, "engine_gpus": engine_gpus,
           "engine_ep": _int(a, "--sglang-ep-size"), "engine_dp": _int(a, "--sglang-dp-size"),
           "prompts": _int(a, "--rollout-batch-size"), "samples_per_prompt": _int(a, "--n-samples-per-prompt"), "samples": batch,
           "max_response": _int(a, "--rollout-max-response-len"), "max_tokens_per_gpu": _int(a, "--max-tokens-per-gpu"),
           "num_rollout": _int(a, "--num-rollout"), "reward_type": a.get("--rm-type"), "routing_replay": bool(a.get("--use-rollout-routing-replay")),
           "kl_loss": bool(a.get("--use-kl-loss")), "moe_backend": a.get("--sglang-moe-runner-backend"),
           "fp8_gemm_backend": a.get("--sglang-fp8-gemm-runner-backend"), "mem_fraction": a.get("--sglang-mem-fraction-static"),
           "chat_kwargs": a.get("--apply-chat-template-kwargs"), "steps_logged": len(d["train"]), "syncs": len(d["syncs"])}
    t0 = job_start.replace(tzinfo=LOG_TZ).timestamp() if job_start else None
    if t0 and d["syncs"]:
        run["startup_min"] = (d["syncs"][0][0] - t0) / 60  # to the end of the first weight sync
    if t0 and d["engines_up"]:
        run["engines_up_min"] = (max(d["engines_up"]) - t0) / 60  # the last engine startup line Ray printed
    covered = {round(tr.get("perf/update_weights_time", -1), 1) for tr in d["train"].values()}
    extra = [s for _, s in d["syncs"] if round(s, 1) not in covered]
    if len(d["syncs"]) > len(d["train"]) and extra:
        run["sync_after_last_s"] = d["syncs"][-1][1]  # slime's update after the last training step: in no step
    steps = []
    for k in sorted(d["train"]):
        tr, ro, rs, st = d["train"][k], d["rollout"].get(k, {}), d["rollout_stats"].get(k, {}), d["step"].get(k, {})
        rollout, train, sync, step = ro.get("perf/rollout_time"), tr.get("perf/train_time"), tr.get("perf/update_weights_time"), tr.get("perf/step_time")
        row = {"job": job, "step": k, "warmup": k == 0, "rollout_s": rollout, "train_s": train, "actor_train_s": tr.get("perf/actor_train_time"),
               "sync_s": sync, "offload_s": tr.get("perf/sleep_time"), "onload_s": tr.get("perf/wake_up_time"), "step_s": step,
               "train_tok_s": tr.get("perf/actor_train_tok_per_s"), "train_tflops_gpu": tr.get("perf/actor_train_tflops"),
               "rollout_tok_s_gpu": ro.get("perf/tokens_per_gpu_per_sec"),
               "resp_mean": ro.get("rollout/response_len/mean"), "resp_median": ro.get("rollout/response_len/median"),
               "resp_max": ro.get("rollout/response_len/max"), "resp_min": ro.get("rollout/response_len/min"),
               "sample_len_mean": rs.get("rollout/total_lengths"), "truncated_pct": 100 * rs["rollout/truncated"] if "rollout/truncated" in rs else None,
               "raw_reward": rs.get("rollout/raw_reward", d["reward"][k] if k < len(d["reward"]) else None),
               "latency_median_s": ro.get("perf/request/e2e_latency/median"), "latency_max_s": ro.get("perf/request/e2e_latency/max"),
               "decode_tok_s_median": ro.get("perf/decode/throughput/median"), "decode_tok_s_max": ro.get("perf/decode/throughput/max"),
               "grad_norm": st.get("train/grad_norm"), "microbatches": d["micro"].get(k), "compiles": d["compiles"].get(k, 0),
               "rollout_end": ro.get("end")}
        if None not in (rollout, train, sync, step):
            row["rest_s"] = step - rollout - train - sync
            row["rollout_pct"] = 100 * rollout / step
        if row["sample_len_mean"] and batch:
            row["train_tokens"] = row["sample_len_mean"] * batch
        if row["microbatches"] and pp:
            row["bubble_pct"] = 100 * (pp - 1) / (row["microbatches"] + pp - 1)  # 1F1B's idle share, (p - 1) / (m + p - 1): computed
        steps.append(row)
    # GPU memory of the first node (milabench monitors it only), the fullest GPU, over each rollout window
    gpu = pd.DataFrame(d["gpu"], columns=["time", "used_gib"])
    for row in steps:
        if row["rollout_end"] and row["rollout_s"] and len(gpu):
            w = gpu[gpu.time.between(row["rollout_end"] - row["rollout_s"], row["rollout_end"])]
            row["gpu_rollout_gib"] = w.used_gib.max() if len(w) else None
    # Host memory per node, slime's snapshots (psutil "used", GiB despite the "GB" label: total_GB is the GPU's 178.34 GiB)
    ip_host = {ip: h for h, ip in d["nodes"]}
    head = d["nodes"][0][1] if d["nodes"] else None  # Ray's head: the job's first node, whose actors' lines carry no address
    for x in d["memory"] + d["decode"]:
        x["ip"] = x["ip"] or head
    mem = pd.DataFrame(d["memory"])
    hosts = []
    if len(mem):
        per_stage = gpus // pp if gpus and pp else None
        mem["stage"] = mem["rank"] // per_stage if per_stage else None
        for ip, g in mem.groupby("ip"):
            hosts.append({"job": job, "host": ip_host.get(ip, ip), "ip": ip, "stage": int(g.stage.mode().iloc[0]) if per_stage else None,
                          "snapshots": len(g), "host_used_gib": g.host_used_GB.max(), "host_total_gib": g.host_total_GB.max(),
                          "gpu_used_gib": g.used_GB.max(), "gpu_total_gib": g.total_GB.max()})
    per_engine = engine_gpus // _int(a, "--actor-num-gpus-per-node", 8) if engine_gpus else None
    by_ip = sorted((ip for _, ip in d["nodes"]), key=lambda ip: tuple(int(p) for p in ip.split(".")))
    engine_of = {ip: i // per_engine for i, ip in enumerate(by_ip)} if per_engine else {}
    decode = []
    for row in steps:
        if not (row["rollout_end"] and row["rollout_s"]):
            continue
        start = row["rollout_end"] - row["rollout_s"]
        for x in d["decode"]:
            if start <= x["time"] <= row["rollout_end"]:
                decode.append({"job": job, "step": row["step"], "minute": (x["time"] - start) / 60, "host": ip_host.get(x["ip"], x["ip"]),
                               "engine": engine_of.get(x["ip"]), **x,
                               "tok_s_seq": x["gen_tok_s"] / x["running"] if x["running"] else None})
    return run, steps, hosts, decode


def build(runs_dir: Path, sacct: Path, unfinished: tuple) -> dict[str, pd.DataFrame]:
    """glm_rl_runs, glm_rl_steps, glm_rl_hosts, glm_rl_decode. Only jobs that have ended (the collector's sacct snapshot)."""
    states = {}
    if sacct.exists():
        s = pd.read_csv(sacct, sep="|", dtype=str)
        states = dict(zip(s.JobID, zip(s.State, s.Start, s.End, s.NNodes)))
    runs, steps, hosts, decode = [], [], [], []
    for d in sorted(runs_dir.glob("*.*.*")) if runs_dir.is_dir() else []:
        entry, job = d.name.split(".")[-1], d.name.split(".")[-2]
        if entry not in SCALE_ENTRIES or not job.isdigit() or not (d / f"{entry}.0.data").exists():
            continue
        state, start, end, nodes = states.get(job, ("RUNNING", None, None, None))
        if state.startswith(unfinished):
            continue  # a job still running is not a failed run: left out until it ends
        r, st, h, dc = build_run(d, datetime.fromisoformat(start) if start else None)
        r.update(job_state=state.split()[0], start=start, end=end)
        runs.append(r)
        steps += st
        hosts += h
        decode += dc
    runs = pd.DataFrame(runs)
    if len(runs):
        runs = runs.sort_values("start").reset_index(drop=True)
    decode = pd.DataFrame(decode)
    if len(decode):
        # a rank's first line after the engine sat idle averages its throughput over the idle time (under 1 token/s): not a speed
        decode["idle"] = decode.gen_tok_s < 1
        decode["slow"] = ~decode.idle & (decode.tok_s_seq < SLOW_DECODE)
    return {"glm_rl_runs": runs, "glm_rl_steps": pd.DataFrame(steps), "glm_rl_hosts": pd.DataFrame(hosts), "glm_rl_decode": decode}

