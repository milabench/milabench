"""Synthetic Trillium data in the target raw formats (../../FORMAT.md), for every experiment of
DESIGN.md not run yet (D1-D4, E1, R, A-Q, A-G, B, C, F1-F4).

Written under ``data/synthetic/trillium/`` with the layout of ``$SCRATCH/trillium/`` (runs/,
health/, nccl/, nodemon/), so that build.py reads it with the same parsers as the measured data.
Every row built from it carries ``synthetic=True`` and is drawn hollow, under a SYNTHETIC mark.

The numbers come from a toy model. None of them is a measurement; they only exercise the
pipeline and give the figures a plausible shape. The model:

- Machine: N_NODES nodes (machine.py: 52, trig0101-trig0152, or 47) x 8 B200, subnets of 7, 7, 7, 7, 4,
  7, 7, 6 nodes,
  800 Gb/s per node (100 GB/s, 85 GB/s achieved). MFU denominator: 2.25 PFLOP/s, the B200 dense
  BF16 *spec* peak (SciNet measured 1,668 TFLOP/s on a BF16 8192^2 matmul).
- Step = compute / (1 - pipeline bubble) + exposed communication (+ NVLink TP traffic).
  Compute: tokens x flops/token x recompute factor / (50 % of the measured 1,668 TFLOP/s),
  slightly more efficient with more tokens per GPU, less with TP; FP8 recipes speed up the
  matmul share (70 %) of it.
- Qwen communication (DESIGN.md "network arithmetic"): ~8 B/param/node/step for every
  data-parallel layout with an FP32 gradient reduction (6 with BF16; 4 for HSDP's all-reduce),
  divided by the PP degree, x (n-1)/n; a layout hides at most a fixed share of it behind
  compute (FSDP 45 %, HSDP-8 80 %).
- GLM-5: Megatron carves EP out of the per-stage data-parallel ranks (8N % PP == 0 and
  8N/PP % EP == 0), so a model copy is PP x EP GPUs. Global batches as in benchmarks/megatron/dev.yaml
  (machine.py): weak 256 / 512 / 768 / 1024 at 13 / 26 / 39 / 52 nodes (32 micro-batches per data-parallel
  rank), strong 768 at all four (96 / 48 / 32 / 24 micro-batches), 64 as a small-batch point at 13 nodes.
  Micro-batch 1, 1F1B bubble (PP - 1) / (m + PP - 1), expert all-to-all over NVLink when the EP group is in
  a node and over the HCAs otherwise (node-level deduplication halves it), gradient reduce-scatter +
  all-gather between copies, which is what makes the toy lose a step from 1 to 2 copies. The runs are
  tagged as the launchers tag them, so that the runs serving two experiments (select.SERVES_TWICE) go
  through the same path as the measured ones. Rate events as Megatron writes them: from batch_id 0,
  warm-up included; loss events numbered from step 1.
- Noise: ~0.3 % run to run, ~1 % step to step; one slow GPU (trig0134 GPU 5, -8 %), one slow
  node (trig0147, -4 %), one half-rate HCA (trig0109), one node with low NVLink bandwidth
  (trig0140); E1: ~1.5 % slowdown when the 4 jobs run together; R: a ~1.2 % drift over the
  window. Deterministic (fixed seeds).
"""

from __future__ import annotations

import json
import math
import shutil
import zlib
from pathlib import Path

import numpy as np

from .machine import (CLUSTER, GLM_AG_RUNS, GLM_COPY_NODES, GLM_SEQS_PER_COPY, GLM_SMALL_SEQS, GLM_STRONG_SEQS,
                      GLM_TOP_COPIES, GLM_TOP_NODES, GPUS_PER_NODE, CKPT_FULL_B_PER_PARAM, CKPT_WEIGHTS_B_PER_PARAM, MATMUL_BF16, MATMUL_FP8, NODE_NET_GBS, NODES,
                      PEAK_BF16, SUBNET_OF)

N = len(NODES)

SCHEMA = "milabench.pretrain.v1"
T0 = 1795006800.0  # window start (placeholder): 2026-11-18 08:00 EST
NET = 0.85 * NODE_NET_GBS * 1e9  # achieved inter-node bandwidth per node, B/s
NVLINK = 350e9  # achieved all-to-all bandwidth per GPU inside a node, B/s
GPU_NET = NET / GPUS_PER_NODE

QWEN = {"model": "qwen3_8-27B", "framework": "torchtitan", "n_layers": 64, "n_layers_full": 64,
        "params_total": 26.9e9, "params_active": 26.9e9, "flops_per_token": 159258915840,
        "seq_len": 4096, "optimizer": "AdamW", "optimizer_state_dtype": "fp32", "compile": "regions"}
GLM = {"model": "GLM-5.3", "framework": "megatron", "n_layers": 78, "n_layers_full": 78,  # the benchmark's name
       "params_total": 744e9, "params_active": 40e9, "flops_per_token": 6 * 40e9,
       "seq_len": 8192, "optimizer": "AdamW", "optimizer_state_dtype": "bfloat16", "compile": None,
       "moe": {"n_experts": 256, "top_k": 8, "n_shared_experts": 1, "routing": "aux-loss-free",
               "dispatcher": "alltoall"}}
DEVIATIONS = [{"id": "optimizer", "text": "AdamW (not Muon)"}, {"id": "mtp", "text": "MTP off"},
              {"id": "data", "text": "random tokens"}]

# Placeholder anomalies, visible in the health and straggler figures.
SLOW_GPU = ("trig0134", 5, 0.92)
SLOW_NODE = ("trig0147", 0.96)
HALF_HCA = ("trig0109", "mlx5_2")
LOW_NVLINK = ("trig0140", 620.0)
HCAS = [f"mlx5_{i}" for i in range(4)]  # 4 x 200 Gb/s per node: a guess (HCA count unknown)

# -- Qwen toy model ------------------------------------------------------------

AC_RECOMPUTE = {"none": 1.0, "selective": 1.1, "full": 4 / 3}
# Activations, GiB per token per GPU (the estimates of config/torchtitan_native_trillium.yaml: at 48k, selective
# AC on hsdp8 is an OOM risk, no AC fits at 12k only).
AC_ACT_GB_PER_TOKEN = {"none": 6e-3, "selective": 2.9e-3, "full": 0.65e-3}
# Matmul speedup of each recipe over BF16 (on the 70 % of compute that is matmul).
RECIPE_SPEEDUP = {"bf16": 1.0, "fp8-tensorwise": 1.55, "fp8-blockwise": 1.35, "mxfp8": 1.60}
# Recipes each framework offers (TorchTitan's pin has no Float8; Megatron through Transformer Engine).
QWEN_RECIPES = ("bf16", "mxfp8")
GLM_RECIPES = ("bf16", "fp8-tensorwise", "fp8-blockwise", "mxfp8")
# layout: (tp, pp, kernel efficiency, largest share of the inter-node traffic hidden behind compute)
QWEN_LAYOUTS = {"fsdp": (1, 1, 1.0, 0.45), "hsdp8": (1, 1, 1.0, 0.80), "hsdp16": (1, 1, 1.0, 0.65),
                "tp2-fsdp": (2, 1, 0.95, 0.50), "tp4-fsdp": (4, 1, 0.88, 0.55),
                "pp2-fsdp": (1, 2, 1.0, 0.45), "pp4-fsdp": (1, 4, 1.0, 0.45)}


def recipe_factor(precision: str) -> float:
    return 1 / (0.3 + 0.7 / RECIPE_SPEEDUP[precision])


def qwen_parallelism(layout: str, nodes: int) -> dict:
    tp, pp, _, _ = QWEN_LAYOUTS[layout]
    gpus = GPUS_PER_NODE * nodes
    replicate = {"hsdp8": nodes, "hsdp16": nodes // 2}.get(layout, 1)
    return {"layout": layout, "gpus": gpus, "nodes": nodes, "dp_replicate": replicate,
            "dp_shard": gpus // (tp * pp * replicate), "tp": tp, "pp": pp, "cp": 1, "ep": 1,
            "pp_schedule": "1F1B" if pp > 1 else None, "copies": replicate}


def qwen_valid(layout: str, nodes: int) -> bool:
    """hsdp16 needs an even node count (13 nodes = 104 GPUs: no 16-GPU shard groups); PP needs
    at least one node per stage."""
    tp, pp, _, _ = QWEN_LAYOUTS[layout]
    return not (layout == "hsdp16" and nodes % 2) and nodes >= pp


def qwen_step(layout, nodes, tokens, grad="fp32", ac="selective", precision="bf16"):
    """(step s, compute s, peak GiB per GPU); step is None when it does not fit in memory."""
    tp, pp, eff, hide = QWEN_LAYOUTS[layout]
    p = QWEN["params_total"]
    par = qwen_parallelism(layout, nodes)
    state = 16 * p / (par["dp_shard"] * tp * pp) / 2**30
    mem = state + AC_ACT_GB_PER_TOKEN[ac] * tokens / tp + 10
    tok_eff = 1 + 0.04 * math.log2(tokens / 12288)
    compute = (tokens * QWEN["flops_per_token"] * AC_RECOMPUTE[ac]
               / (0.5 * MATMUL_BF16 * eff * tok_eff * recipe_factor(precision)))
    m = tokens * pp / QWEN["seq_len"]
    bubble = (pp - 1) / (m + pp - 1)
    groups = nodes // pp
    if groups <= 1:
        exposed = 0.03 * compute  # NVLink only
    else:
        bpp = {"hsdp8": 8 if grad == "fp32" else 4, "hsdp16": 8 if grad == "fp32" else 5}.get(
            layout, 8 if grad == "fp32" else 6)
        comm = bpp * p / pp * (groups - 1) / groups / NET * (1 + 0.015 * math.log2(groups))
        exposed = max((1 - hide) * comm, comm - 0.8 * compute)  # a layout hides at most `hide` of it
    tp_comm = {1: 0, 2: 0.04, 4: 0.07}[tp] * compute
    step = compute / (1 - bubble) + exposed + tp_comm + (0.05 if pp > 1 else 0)
    return (step if mem <= 175 else None), compute, mem


# -- GLM-5 toy model -----------------------------------------------------------

GLM_LAYOUTS = {"pp13-ep8": (13, 8), "pp13-ep16": (13, 16), "pp13-ep32": (13, 32), "pp26-ep8": (26, 8),
               "pp6-ep16": (6, 16)}


def glm_parallelism(layout: str, nodes: int, batch: int) -> dict:
    """batch: global, in sequences; micro-batch 1, so a data-parallel rank has batch / dp micro-batches."""
    pp, ep = GLM_LAYOUTS[layout]
    gpus = GPUS_PER_NODE * nodes
    dp = gpus // pp
    return {"layout": layout, "gpus": gpus, "nodes": nodes, "dp_replicate": 1, "dp_shard": dp,
            "expert_dp": gpus // (ep * pp), "tp": 1, "pp": pp, "cp": 1, "ep": ep, "pp_schedule": "1F1B",
            "copies": gpus // (pp * ep), "microbatches_per_pipeline": batch // dp}


def glm_step(layout, nodes, batch, ac="full", precision="bf16"):
    """(step s, compute s, all-to-all s): a copy is PP x EP GPUs; copies reduce their gradients."""
    pp, ep = GLM_LAYOUTS[layout]
    gpus_copy = pp * ep
    copies = GPUS_PER_NODE * nodes // gpus_copy
    tokens_copy = batch / copies * GLM["seq_len"]
    compute = (tokens_copy * GLM["flops_per_token"] * AC_RECOMPUTE[ac]
               / (gpus_copy * MATMUL_BF16 * 0.35 * recipe_factor(precision)))
    m = max(glm_parallelism(layout, nodes, batch)["microbatches_per_pipeline"], 1)
    bubble = (pp - 1) / (m + pp - 1)
    # Expert all-to-all: dispatch + combine, forward + backward, hidden 6144 in BF16, top-8.
    a2a_bytes = tokens_copy / ep * (78 / pp) * 6144 * 2 * 8 * 2 * 2
    off_node = 1 - 8 / ep if ep > 8 else 0.0
    a2a = (a2a_bytes * (1 - off_node) / NVLINK + 0.5 * a2a_bytes * off_node / GPU_NET) * 0.75
    sync = 0.0
    if copies > 1:  # distributed optimizer: reduce-scatter grads + all-gather params, BF16
        per_node = 2 * 2 * GLM["params_total"] / gpus_copy * GPUS_PER_NODE * (copies - 1) / copies
        sync = 0.7 * per_node / NET
    step = compute / (1 - bubble) + a2a + sync
    return step, compute, a2a


# -- writers -------------------------------------------------------------------

def _uuid(host: str, gpu: int) -> str:
    h = zlib.crc32(f"{host}:{gpu}".encode())
    return f"GPU-{h:08x}-0000-0000-0000-{h:012x}"


def loss_at(step: int, offset=0.0, lr_warmup=0) -> float:
    """With a warm-up (GLM: lr from 0): steps 1 and 2 see the same weights, then the lr ramps up."""
    if lr_warmup:
        x = max(step - 2, 0)
        step = x * x / (2 * lr_warmup) if x < lr_warmup else x - lr_warmup / 2
    return 2.3 + 8.6 * math.exp(-step / 45) + 0.7 * math.exp(-step / 400) + offset


def write_run(root, name, bench, w, par, hosts, step_s, rng, *, experiment, variant, repetition,
              t_start, tokens_gpu, grad_accum=1, global_tokens=None, grad="fp32", ac="full",
              precision="bf16", warmup=30, n_steps=None, budget=300.0, rc=0, mem=60.0,
              loss_offset=0.0, loss_seed=None, run_extra=None, ep_a2a=None, checkpoints=(),
              power_busy=0.8, nodemon=True, nodemon_extra=None, microbatch=None, seed=1234, first_loss_step=1, lr_warmup=0):
    """One milabench run directory: run_start, rate and loss events, checkpoint events, summary,
    end. Returns the end time."""
    gpus = par["gpus"]
    global_tokens = global_tokens or tokens_gpu * gpus
    big = w is GLM
    init = (240 + 12 * par["nodes"] if big else 25 + 0.4 * par["nodes"]) * (1 + rng.normal(0, 0.04))
    first = init + (60 + 2 * par["nodes"] if big else 35 + 0.3 * par["nodes"]) * (1 + rng.normal(0, 0.05))
    run = {"cluster": CLUSTER, "campaign": "trillium-window", "experiment": experiment, "variant": variant,
           "repetition": repetition, "slurm_job_id": str(2000000 + zlib.crc32(name.encode()) % 99999),
           "slurm_reservation": "trillium-b200", "exclusive_cluster": True, **(run_extra or {})}
    workload = {k: w[k] for k in ("model", "framework", "n_layers", "n_layers_full", "params_total",
                                  "params_active", "seq_len", "optimizer", "optimizer_state_dtype",
                                  "compile")}
    workload.update({"params_trainable_fraction": 1.0,
                     "microbatch_sequences": microbatch or tokens_gpu // grad_accum // w["seq_len"],
                     "grad_accum": grad_accum, "tokens_per_gpu_per_step": tokens_gpu,
                     "global_batch_tokens": global_tokens, "precision": precision,
                     "grad_reduce_dtype": grad, "activation_checkpointing": ac, "seed": seed,
                     "moe": ({**w["moe"], "ep": par["ep"]} if big else None)})
    placement = {"hosts": hosts, "family": "slurm", "excluded_nodes": {}}
    events = [{"event": "config", "data": {"run_name": name}},
              {"event": "start", "data": {"time": t_start}},
              {"event": "data", "data": {"run_start": {"run": run, "workload": workload,
                                                         "parallelism": par, "placement": placement}}}]
    t = t_start + first
    n_steps = n_steps or max(15, int(budget / step_s))
    lrng = np.random.default_rng(loss_seed) if loss_seed is not None else rng
    measure_start = t + warmup * step_s
    for i in range(warmup + n_steps if rc == 0 else 0):  # a failed attempt crashed before measuring
        s = step_s * (1 + rng.normal(0, 0.01)) * (2.5 if i == 0 and warmup else 1.0)
        if i < warmup:  # the warm-up takes step_s each on the clock (the first step's extra is in `first`)
            t += step_s
        else:
            t += s
        step = i + first_loss_step
        loss = loss_at(step, loss_offset, lr_warmup) + lrng.normal(0, 0.03) + rng.normal(0, 0.004)
        if lr_warmup:  # the early unstable phase, different in every run
            loss += rng.normal(0, 0.15 * math.exp(-step / 15))
        if i >= warmup or big:  # Megatron's rate events start at the warm-up, numbered from 0; TorchTitan's after it
            events.append({"event": "data", "data": {"rate": global_tokens / s, "units": "tokens/s", "task": "train",
                                                     "time": t, "elapsed": s, "batch_id": i if big else i + 1}})
        # Loss: its own event from step 1, measured or not; TorchTitan numbers it batch_id, Megatron step.
        events.append({"event": "data", "data": {"loss": loss, "step" if big else "batch_id": step,
                                                 "task": "train", "time": t}})
    for c in checkpoints:
        t += c["blocked_s"]
        c = {**c, "start_epoch": t - c["blocked_s"], "end_epoch": t - c["blocked_s"] + c["total_s"]}
        events.append({"event": "data", "data": {"checkpoint": c, "task": "train"}})
    if rc == 0:
        summary = {
            "schema": SCHEMA, "run": run, "workload": workload, "parallelism": par, "placement": placement,
            "measurement": {"warmup_steps": warmup, "measured_steps": n_steps, "measure_seconds_budget": budget,
                            "measurement_start_epoch": measure_start, "measurement_end_epoch": t},
            "mfu_inputs": {"flops_per_token": w["flops_per_token"], "peak_flops_per_gpu": PEAK_BF16,
                           "formula": "flops_per_token x tokens/s per GPU / peak (B200 dense BF16 spec)"},
            "memory_gib": {"peak_allocated": {"min": mem - 2, "max": mem},
                           "peak_reserved": {"min": mem + 4, "max": mem + 6}},
            "startup": {"process_start_epoch": {"min": t_start, "max": t_start},
                        "model_initialized": {"min_s": init * 0.97, "max_s": init},
                        "first_step_done": {"min_s": first * 0.99, "max_s": first},
                        "measurement_start": {"min_s": measure_start - t_start, "max_s": measure_start - t_start},
                        "compile_cache": "warm"},
            "per_rank": [], "deviations": DEVIATIONS,
        }
        if ep_a2a is not None:
            summary["per_ep_group"] = [{"hosts": hosts[i:i + max(1, par["ep"] // 8)],
                                        "a2a_s_per_step": ep_a2a * (1 + rng.normal(0, 0.03))}
                                       for i in range(0, len(hosts), max(1, par["ep"] // 8))]
        events.append({"event": "data", "data": {"summary": summary, "task": "train"}})
    events.append({"event": "end", "data": {"time": t, "return_code": rc}})
    run_dir = root / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / f"{bench}.0.data").write_text("\n".join(json.dumps(e) for e in events) + "\n")
    if nodemon and rc == 0:
        write_nodemon(root, name, bench, hosts, measure_start, t, power_busy, precision != "bf16",
                      **(nodemon_extra or {}))
    return t


GPU_COLUMNS = ["time_ms", "gpu", "sm_clock_mhz", "clocks_event_reasons", "power_w", "temp_c", "util",
               "util_mean", "util_min", "util_samples", "xid"]
CPU_COLUMNS = ["time_ms", "package", "busy_cores", "cpu_mhz_busy_mean", "cpu_mhz_min", "cpu_mhz_max",
               "cpu_mhz_mean", "cpu_mhz_cap", "cpu_util", "cpu_temp_c", "thermal_throttle_count", "governor"]
GPM_COLUMNS = ["time_ms", "gpu", "sm_util", "sm_occupancy", "tensor_util", "dram_bw_util"]


def _tsv(path, columns, rows):
    path.write_text("\n".join(["\t".join(columns), *("\t".join(str(v) for v in r) for r in rows)]) + "\n")


def write_nodemon(root, run_name, bench, hosts, t0, t1, busy, fp8, period=None, slow=None, cpu_slow=None,
                  cpu=False):
    """nodemon directory of one run (gpu collector; cpu and gpm too for the long run).
    slow = (host, gpu, start s, end s): throttled GPU; cpu_slow = (host, start s, end s)."""
    rng = np.random.default_rng(zlib.crc32(run_name.encode()))
    period = period or max(5.0, round((t1 - t0) / 12))
    out = root / "nodemon" / run_name / "nodes"
    out.mkdir(parents=True, exist_ok=True)
    times = np.arange(math.ceil(t0 / period) * period, t1, period)
    for h in hosts:
        gpu_rows, gpm_rows, cpu_rows = [], [], []
        for t in times:
            ms = f"{t * 1000:.0f}"
            dip = slow is not None and slow[2] <= t - t0 < slow[3]
            for g in range(GPUS_PER_NODE):
                hot = dip and h == slow[0] and g == slow[1]
                clock = 1650 if hot else 1965 + rng.normal(0, 10)
                b = min(0.99, (0.97 if hot else busy * (0.92 if dip else 1.0)) + rng.normal(0, 0.006))
                power = 200 + 780 * b * (1.05 if fp8 else 1.0) * (0.85 if hot else 1.0) + rng.normal(0, 12)
                gpu_rows.append([ms, g, f"{clock:.0f}", "0x8" if hot else "0x0", f"{power:.1f}",
                                 f"{(83 if hot else 55) + rng.normal(0, 1.5):.1f}", f"{b:.3f}", f"{b:.3f}",
                                 f"{b - 0.08:.3f}", 50, ""])
                gpm_rows.append([ms, g, "", "", f"{(0.42 if hot else 0.33 * b / busy) + rng.normal(0, 0.005):.3f}", ""])
            if cpu:
                low = cpu_slow is not None and h == cpu_slow[0] and cpu_slow[1] <= t - t0 < cpu_slow[2]
                for package in (0, 1):
                    mhz = (2600 if low else 3500) + rng.normal(0, 25)
                    cpu_rows.append([ms, package, 48, f"{mhz:.0f}", f"{mhz - 800:.0f}", f"{mhz + 100:.0f}",
                                     f"{mhz - 300:.0f}", "", f"{0.4 + rng.normal(0, 0.01):.3f}",
                                     f"{(70 if low else 55) + rng.normal(0, 1):.1f}", "", ""])
        _tsv(out / f"{h}.gpu.tsv", GPU_COLUMNS, gpu_rows)
        if cpu:
            _tsv(out / f"{h}.gpm.tsv", GPM_COLUMNS, gpm_rows)
            _tsv(out / f"{h}.cpu.tsv", CPU_COLUMNS, cpu_rows)
        gpus = {str(g): _uuid(h, g) for g in range(GPUS_PER_NODE)}
        (out.parent / f"{h}.status.json").write_text(json.dumps({
            "schema": "benchmate.nodemon.v1", "state": "closed", "bench": bench, "run_name": run_name,
            "start": round(t0 * 1000), "end": round(t1 * 1000), "flush_every": 10.0,
            "collectors": {"gpu": {"enabled": True, "period_s": period, "gpus": gpus},
                           "gpm": {"enabled": cpu, "period_s": period, "gpus": gpus},
                           "cpu": {"enabled": cpu, "period_s": period}},
            "clock": {"synced": True, "maxerror_us": 40000}}))


def write_health(root, job, rng):
    """D1: one srygaard results directory over every node, with the Trillium additions
    (per-GPU matmul and per-HCA loopback in nodes/<host>.json)."""
    run = root / "health" / job
    (run / "groups").mkdir(parents=True, exist_ok=True)
    (run / "nodes").mkdir(parents=True, exist_ok=True)
    (run / "switches.json").write_text(json.dumps({"unknown": NODES}))
    tests, report = [], {}
    for h in NODES:
        node_f = SLOW_NODE[1] if h == SLOW_NODE[0] else 1.0
        gpus = []
        for g in range(GPUS_PER_NODE):
            f = node_f * (SLOW_GPU[2] if (h, g) == SLOW_GPU[:2] else 1.0)
            gpus.append({"gpu": g, "uuid": _uuid(h, g), "size": 8192,
                         "bf16_tflops": round(MATMUL_BF16 / 1e12 * f * (1 + rng.normal(0, 0.006)), 1),
                         "fp8_tflops": round(MATMUL_FP8 / 1e12 * f * (1 + rng.normal(0, 0.008)), 1)})
        hcas = [{"hca": hca, "port": 1, "rate_gbps": 100 if (h, hca) == HALF_HCA else 200, "numa": i // 2,
                 "gpus": [2 * i, 2 * i + 1],
                 "loopback_gbs": round((11.8 if (h, hca) == HALF_HCA else 23.6) + rng.normal(0, 0.15), 2)}
                for i, hca in enumerate(HCAS)]
        (run / "nodes" / f"{h}.json").write_text(json.dumps({
            "host": h, "job_id": job,
            "matmul": {"status": "ok", "data": {"gpus": gpus}},
            "hca": {"status": "ok", "data": {"hcas": hcas}}}))
        nv = LOW_NVLINK[1] if h == LOW_NVLINK[0] else 720.0
        tests.append({"kind": "intra", "nodes": [h], "status": "ok", "peak_busbw": round(nv + rng.normal(0, 4), 1)})
        findings = []
        if h == LOW_NVLINK[0]:
            findings.append({"severity": "warn", "check": "nccl.intra", "detail": f"busbw {nv:.0f} < 90% of median"})
        if h == HALF_HCA[0]:
            findings.append({"severity": "fail", "check": "hca.rate", "detail": f"{HALF_HCA[1]} at 100 Gb/s"})
        report[h] = {"status": "fail" if any(f["severity"] == "fail" for f in findings) else
                     ("warn" if findings else "pass"), "switch": "unknown", "findings": findings}
    (run / "groups" / f"{job}.json").write_text(json.dumps({"job_id": job, "nodes": NODES, "tests": tests}))
    (run / "report.json").write_text(json.dumps({"nodes": report}))


def _round_robin(nodes):
    """All pairs in len-1 rounds of disjoint pairs (circle method; an odd count: a bye in every round)."""
    n = list(nodes) + ([None] if len(nodes) % 2 else [])
    rounds = []
    for _ in range(len(n) - 1):
        rounds.append([(n[i], n[-1 - i]) for i in range(len(n) // 2) if None not in (n[i], n[-1 - i])])
        n = [n[0], n[-1], *n[1:-1]]
    return rounds


def _pair_bw(a, b, rng):
    bw = 91.0 * (0.875 if HALF_HCA[0] in (a, b) else 1.0) * (0.99 if SUBNET_OF[a] != SUBNET_OF[b] else 1.0)
    return bw * (1 + rng.normal(0, 0.006))


def write_pairs(root, job, rng, t):
    """D2: every node pair, N//2 disjoint pairs per round, N-1 rounds (52 nodes: 26 and 51; 47: 23 and 47)."""
    out = root / "nccl" / job
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for r, pairs in enumerate(_round_robin(NODES), 1):
        for a, b in pairs:
            lat = (2.1 if SUBNET_OF[a] == SUBNET_OF[b] else 2.6) + rng.normal(0, 0.05)
            rows.append({"schema": "trillium.pairs.v1", "job_id": job, "round": r, "a": a, "b": b,
                         "tool": "ib_write_bw/ib_write_lat, all HCAs", "size_bytes": 2**30,
                         "bw_gbs": round(_pair_bw(a, b, rng), 2), "lat_us": round(lat, 2), "lat_size_bytes": 8,
                         "time": t + 17 * r, "status": "ok"})
    (out / "pairs.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def _busbw(coll, nodes, size, rng):
    if nodes == 1:
        peak = {"all_reduce": 760, "all_gather": 680, "reduce_scatter": 680, "alltoall": 620}[coll]
        half = 4 * 2**20
    elif coll == "alltoall":
        peak = 0.8 * NODE_NET_GBS * nodes / (GPUS_PER_NODE * (nodes - 1))
        half = 16 * 2**20 * nodes / 2
    else:
        peak = 0.9 * NODE_NET_GBS * (1 - 0.02 * math.log2(nodes)) * (0.97 if coll != "all_reduce" else 1.0)
        half = 8 * 2**20 * max(1, nodes / 4)
    return peak * size / (size + half) * (1 + rng.normal(0, 0.008))


def write_collectives(root, job, rng, t):
    """D3 (collectives vs nodes, 1 GiB; size sweep at 2 and N nodes) and D4 (bisection, 3 cuts)."""
    out = root / "nccl" / job
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    sizes = [2**k for k in range(10, 34, 2)]
    for coll in ("all_reduce", "all_gather", "reduce_scatter", "alltoall"):
        for nodes in sorted({n for n in (1, 2, 4, 8, 13, 26) if n < N} | {N}):
            hosts = NODES[:nodes]
            for rep in (1, 2):
                for size in (sizes if nodes in (2, N) and rep == 1 else [2**30]):
                    bw = _busbw(coll, nodes, size, rng)
                    factor = {"all_reduce": 2, "alltoall": 1}.get(coll, 1) * (nodes * 8 - 1) / (nodes * 8)
                    alg = bw / factor
                    rows.append({"schema": "trillium.collectives.v1", "experiment": "D3", "job_id": job,
                                 "collective": coll, "nodes": nodes, "gpus": 8 * nodes, "hosts": hosts,
                                 "size_bytes": size, "time_us": round(size / (alg * 1e3), 1),
                                 "algbw_gbs": round(alg, 2), "busbw_gbs": round(bw, 2), "cut": None,
                                 "repetition": rep, "time": t, "nccl_version": "2.27.7", "status": "ok"})
                    t += 2
    for cut in (1, 2, 3):
        order = list(np.random.default_rng(cut).permutation(NODES))
        half = N // 2
        a, b = sorted(order[:half]), sorted(order[half:2 * half])
        pairs = [{"a": x, "b": y, "bw_gbs": round(_pair_bw(x, y, rng) * 0.98, 2)}
                 for x, y in zip(order[:half], order[half:])]
        rows.append({"schema": "trillium.collectives.v1", "experiment": "D4", "job_id": job,
                     "collective": "bisection", "nodes": 2 * half, "gpus": 16 * half, "hosts": a + b, "size_bytes": 2**30,
                     "busbw_gbs": round(float(np.median([p["bw_gbs"] for p in pairs])), 2), "algbw_gbs": None,
                     "time_us": None, "cut": cut, "halves": [a, b], "pairs": pairs, "repetition": 1,
                     "time": t, "nccl_version": "2.27.7", "status": "ok"})
        t += 60
    (out / "collectives.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def _checkpoint(op, mode, content, nbytes, nodes, rng, files, fmt, rep, **extra):
    """F1 toy model: VAST over NFS (the nodes are diskless) at <= 5 GB/s write per node, 100 GB/s
    in total; reads <= 8 GB/s per node, 150 GB/s; async blocks while the state is staged to host
    memory (20 GB/s per node, 40 pinned; the first pinned save also allocates the buffers, 10 GB/s).
    x1.0-1.3: time of day (other users' load)."""
    tod = rng.uniform(1.0, 1.3)
    if op == "save":
        total = (nbytes / 1e9 / min(5 * nodes, 100) + 5) * tod
        staging = {"sync": None, "async": 20, "async_pinned": 40}[mode]
        blocked = total if staging is None else nbytes / 1e9 / (staging * nodes) + 1
        total = total * (1.08 if staging else 1.0)
        if mode == "async_pinned":
            extra["pinned_buffers_allocated"] = rep == 1
            blocked += nbytes / 1e9 / (10 * nodes) if rep == 1 else 0
    else:
        total = (nbytes / 1e9 / min(8 * nodes, 150) + 10) * tod
        blocked = total
    return {"op": op, "step": 40, "repetition": rep, "mode": mode, "content": content, "format": fmt,
            "bytes": nbytes, "blocked_s": round(blocked, 2), "total_s": round(total, 2),
            "path": "/scratch/<user>/trillium/checkpoints/<run>", **extra}


# -- experiments ---------------------------------------------------------------

def _hosts(rng, n, exclude=()):
    pool = [h for h in NODES if h not in exclude]
    return sorted(rng.choice(pool, n, replace=False).tolist()) if n < len(pool) else pool[:n]


def generate(out: Path) -> None:
    root = out / CLUSTER
    if out.exists():
        shutil.rmtree(out)
    root.mkdir(parents=True)
    rng = np.random.default_rng(2026)
    write_health(root, "3000001", rng)
    write_pairs(root, "3000002", rng, T0 + 1800)
    write_collectives(root, "3000003", rng, T0 + 3600)

    def qwen(name, layout, nodes, tokens, *, experiment, rep, t, grad="fp32", ac="selective", precision="bf16",
             accum=1, hosts=None, extra=None, factor=1.0, **kw):
        if not qwen_valid(layout, nodes):
            return t
        step, compute, mem = qwen_step(layout, nodes, tokens // accum, grad, ac, precision)
        par = qwen_parallelism(layout, nodes)
        hosts = hosts or NODES[:nodes]
        variant = f"{layout}-{tokens // 1024}k-{grad}-{ac}-{precision}-n{nodes}"
        if step is None:  # out of memory: a failed attempt, recorded
            return write_run(root, name, "tt-qwen", QWEN, par, hosts, compute, rng, experiment=experiment,
                             variant=variant, repetition=rep, t_start=t, tokens_gpu=tokens, rc=1, mem=mem,
                             grad=grad, ac=ac, precision=precision, run_extra=extra)
        step = step * accum * factor * (1 + rng.normal(0, 0.003))
        return write_run(root, name, "tt-qwen", QWEN, par, hosts, step, rng, experiment=experiment,
                         variant=variant, repetition=rep, t_start=t, tokens_gpu=tokens, grad_accum=accum,
                         grad=grad, ac=ac, precision=precision, mem=mem, run_extra=extra,
                         power_busy=min(0.95, compute * accum / step + 0.05), **kw)

    # E1: disjoint random sets of 13 nodes (4 of them, or 3 at 47 nodes), alone then together; 2 draws.
    t = T0 + 2 * 3600
    for draw in (1, 2):
        order = list(np.random.default_rng(100 + draw).permutation(NODES))
        sets = [sorted(order[13 * i:13 * i + 13]) for i in range(N // 13)]
        for cond in ("alone", "together"):
            t_cond = t
            for i, hosts in enumerate(sets, 1):
                slow = (1.015 + rng.normal(0, 0.005)) if cond == "together" else 1.0
                end = qwen(f"syn-e1-d{draw}-s{i}-{cond}", "hsdp8", 13, 24576, experiment="E1", rep=draw,
                           t=t_cond, hosts=hosts, factor=slow, extra={"node_set": f"d{draw}-s{i}", "condition": cond})
                t_cond = end + 30 if cond == "alone" else t_cond
                t = max(t, end) + 30

    # R: reference run every ~3 h through the window, slow drift of the background load.
    for k in range(18):
        drift = 1 + 0.012 * k / 17 + 0.003 * math.sin(k / 2.5)
        qwen(f"syn-r-{k + 1:02d}", "fsdp", 4, 12288, experiment="R", rep=k + 1, t=T0 + 1.5 * 3600 + k * 3 * 3600,
             hosts=NODES[:4], factor=drift)

    # A-Q: layout x tokens per GPU at 13 nodes; reduction dtype; activation checkpointing; scales.
    t = T0 + 6 * 3600
    tokens = (12288, 24576, 49152)
    for layout in QWEN_LAYOUTS:
        for tok in tokens:
            for rep in (1, 2):
                t = qwen(f"syn-aq-{layout}-{tok // 1024}k-n13-r{rep}", layout, 13, tok, experiment="A-Q", rep=rep,
                         t=t) + 30
    best = sorted((l for l in QWEN_LAYOUTS if qwen_valid(l, 13)),
                  key=lambda l: min(qwen_step(l, 13, k)[0] / k for k in tokens if qwen_step(l, 13, k)[0]))
    for layout in best[:2]:
        for tok in tokens:
            for rep in (1, 2):
                t = qwen(f"syn-aq-{layout}-{tok // 1024}k-bf16red-n13-r{rep}", layout, 13, tok, experiment="A-Q",
                         rep=rep, t=t, grad="bf16") + 30
    for nodes in (1, 13):
        for layout in ("fsdp", best[0]) if nodes == 13 else ("fsdp",):
            for ac in ("full", "none"):
                for rep in (1, 2):
                    t = qwen(f"syn-aq-{layout}-12k-ac{ac}-n{nodes}-r{rep}", layout, nodes, 12288, experiment="A-Q",
                             rep=rep, t=t, ac=ac) + 30
    for nodes, layouts in ((2, best[:3]), (N, best[:3]), (1, ["fsdp", "tp2-fsdp", "tp4-fsdp"])):
        for layout in layouts:
            for tok in tokens:
                for rep in (1, 2):
                    t = qwen(f"syn-aq-{layout}-{tok // 1024}k-n{nodes}-r{rep}", layout, nodes, tok,
                             experiment="A-Q", rep=rep, t=t) + 30

    # B-Q: weak scaling (24k tokens per GPU), best layout and fsdp; strong scaling (fixed batch:
    # N x 8 GPUs x 24k tokens).
    weak = sorted({n for n in (1, 2, 4, 8, 13, 26) if n < N} | {N})
    for layout in dict.fromkeys([best[0], "fsdp"]):
        for nodes in weak:
            t = qwen(f"syn-bqw-{layout}-n{nodes}", layout, nodes, 24576, experiment="B-Q-weak", rep=1, t=t) + 30
    for nodes in (n for n in (13, 26, 52) if n <= N):  # tri-bqs-*: the batch of 52 nodes, 52 / n accumulation steps
        accum = 52 // nodes
        t = qwen(f"syn-bqs-{best[0]}-n{nodes}", best[0], nodes, 24576 * accum, experiment="B-Q-strong", rep=1,
                 t=t, accum=accum) + 30

    # C (Qwen): recipes at 1 node and N nodes; loss over 500 steps at 13 nodes.
    for nodes in (1, N):
        for prec in QWEN_RECIPES:
            for rep in (1, 2):
                t = qwen(f"syn-cs-qwen-{prec}-n{nodes}-r{rep}", "fsdp" if nodes == 1 else best[0], nodes, 24576,
                         experiment="C-speed", rep=rep, t=t, precision=prec) + 30
    loss_offset = {"bf16": 0.0, "fp8-tensorwise": 0.012, "fp8-blockwise": 0.004, "mxfp8": 0.008}
    # Same-recipe repeats, for the noise band of precision_loss: BF16 again with the same seed, and with another seed.
    repeats = [("r2", 2, 1234, 0), ("seed2", 1, 4321, 1)]
    for prec in QWEN_RECIPES:
        t = qwen(f"syn-cl-qwen-{prec}", best[0], 13, 24576, experiment="C-loss", rep=1, t=t, precision=prec,
                 warmup=0, n_steps=500, loss_offset=loss_offset[prec], loss_seed=7, nodemon=False) + 30
    for tag, rep, seed, other in repeats:
        t = qwen(f"syn-cl-qwen-bf16-{tag}", best[0], 13, 24576, experiment="C-loss", rep=rep, t=t, seed=seed, warmup=0,
                 n_steps=500, loss_offset=0.003 * other, loss_seed=7 + other * 100, nodemon=False) + 30

    # GLM-5: 20 iterations, the first one is warm-up (main.py WARMUP_STEPS); global batches of machine.py.
    def glm(name, layout, nodes, batch, *, experiment, rep, t, ac="full", precision="bf16", **kw):
        step, compute, a2a = glm_step(layout, nodes, batch, ac, precision)
        par = glm_parallelism(layout, nodes, batch)
        step *= 1 + rng.normal(0, 0.004)
        global_tokens = batch * GLM["seq_len"]
        mem = 168 if layout == "pp6-ep16" else 125  # GiB per GPU, toy (Rorqual's 13 TiB on 104 GPUs)
        suffix = {"selective": "-sel"}.get(ac, "") + ("" if precision == "bf16" else f"-{precision}")
        return write_run(root, name, "glm5", GLM, par, NODES[:nodes], step, rng, experiment=experiment,
                         variant=f"megatron-glm5-tri-{layout}-n{nodes}-gbs{batch}{suffix}", repetition=rep,
                         t_start=t, tokens_gpu=global_tokens // par["gpus"], global_tokens=global_tokens,
                         ac=ac, precision=precision, mem=mem, ep_a2a=a2a, microbatch=1, grad="bf16",
                         power_busy=min(0.95, compute / step + 0.1), **{"warmup": 1, "n_steps": 19, **kw})

    weak_batch = {c * GLM_COPY_NODES: c * GLM_SEQS_PER_COPY for c in range(1, GLM_TOP_COPIES + 1)}
    # A-G: the layouts at the node counts of machine.py; on pp13-ep8: selective recompute and the small batch.
    for layout, nodes, batch in GLM_AG_RUNS:
        for rep in (1, 2):
            t = glm(f"syn-ag-{layout}-n{nodes}-gbs{batch}-r{rep}", layout, nodes, batch, experiment="A-G", rep=rep,
                    t=t) + 60
    for rep in (1, 2):
        for nodes in (n for n in (13, 52) if n <= N):
            t = glm(f"syn-ag-pp13-ep8-sel-n{nodes}-r{rep}", "pp13-ep8", nodes, weak_batch[nodes], experiment="A-G",
                    rep=rep, t=t, ac="selective") + 60
        t = glm(f"syn-ag-pp13-ep8-small-n13-r{rep}", "pp13-ep8", 13, GLM_SMALL_SEQS, experiment="A-G", rep=rep, t=t) + 60
    # B-G, tagged as glm/20_bg.sh does: the weak runs at 13 and 52 nodes are A-G's above, 39 nodes is one
    # run for weak and strong, and 26 nodes also serves A-G (select.SERVES_TWICE).
    for nodes in (n for n in (26, 39) if n <= N):
        t = glm(f"syn-bg-weak-n{nodes}", "pp13-ep8", nodes, weak_batch[nodes], experiment="B-G-weak", rep=1, t=t) + 60
    for nodes in (n for n in (13, 26, 52) if n <= N):
        t = glm(f"syn-bg-strong-n{nodes}", "pp13-ep8", nodes, GLM_STRONG_SEQS, experiment="B-G-strong", rep=1, t=t) + 60
    # C (GLM): recipes at 1 copy and at the machine; loss over 500 steps at 13 nodes.
    for nodes in (13, GLM_TOP_NODES):
        for prec in GLM_RECIPES:
            t = glm(f"syn-cs-glm-{prec}-n{nodes}", "pp13-ep8", nodes, weak_batch[nodes], experiment="C-speed", rep=1, t=t,
                    precision=prec) + 60
    for prec in GLM_RECIPES:
        t = glm(f"syn-cl-glm-{prec}", "pp13-ep8", 13, GLM_SEQS_PER_COPY, experiment="C-loss", rep=1, t=t, precision=prec,
                warmup=1, n_steps=499, loss_offset=loss_offset[prec], loss_seed=8, lr_warmup=50, nodemon=False) + 60
    for tag, rep, seed, other in repeats:
        t = glm(f"syn-cl-glm-bf16-{tag}", "pp13-ep8", 13, GLM_SEQS_PER_COPY, experiment="C-loss", rep=rep, t=t, seed=seed,
                warmup=1, n_steps=499, loss_offset=0.003 * other, loss_seed=8 + other * 100, lr_warmup=50,
                nodemon=False) + 60

    # F1: 12.0 B/param full, 2.0 B/param weights (machine.py), 3 repetitions at different times of day.
    # Qwen: one run per entry, each saves and loads twice (tri-ckpt-*, always fsdp).
    case = 0
    for nodes in (1, 13, N):
        step, _, train_mem = qwen_step("fsdp", nodes, 12288)
        for content, mode in (("full", "sync"), ("full", "async"), ("full", "async_pinned"), ("weights", "sync")):
            size = QWEN["params_total"] * (CKPT_FULL_B_PER_PARAM if content == "full" else CKPT_WEIGHTS_B_PER_PARAM)
            for rep in (1, 2, 3):
                ck = []
                extra = 1.5 if content == "full" else 1.0  # toy, as GLM below
                fields = dict(params_total=QWEN["params_total"], peak_allocated_gib=round(train_mem * extra, 1))
                for r in (1, 2):
                    ck += [_checkpoint("save", mode, content, size, nodes, rng, 8 * nodes + 1, "torch.dcp", r, **fields),
                           _checkpoint("load", "sync", content, size, nodes, rng, 8 * nodes + 1, "torch.dcp", r, **fields)]
                write_run(root, f"syn-f1-qwen-{content}-{mode}-n{nodes}-r{rep}", "tt-qwen", QWEN,
                          qwen_parallelism("fsdp", nodes), NODES[:nodes], step, rng, experiment="F1",
                          variant=f"tri-ckpt-{'full' if content == 'full' else 'model'}-{mode.replace('async_', '')}",
                          repetition=rep, t_start=T0 + 30 * 3600 + rep * 8 * 3600 + case * 300, tokens_gpu=12288,
                          warmup=10, n_steps=10, checkpoints=ck, nodemon=False)
                case += 1

    # GLM-5: separate runs, as glm/40_f1.sh: save (sync), load, save (async), load, the checkpoint deleted after each
    # load. Memory: a full save and a load hold more than training (toy: x1.5 and x1.54, from the Fir ratios).
    def glm_f1(content, nodes, layout, batch, reps, train_mem):
        nonlocal case
        size = GLM["params_total"] * (CKPT_FULL_B_PER_PARAM if content == "full" else CKPT_WEIGHTS_B_PER_PARAM)
        step, _, _ = glm_step(layout, nodes, batch)
        par = glm_parallelism(layout, nodes, batch)
        common = dict(experiment="F1", tokens_gpu=batch * GLM["seq_len"] // par["gpus"], microbatch=1, grad="bf16",
                      global_tokens=batch * GLM["seq_len"], nodemon=False, warmup=1)
        where = f"n{nodes}"
        tag = "-ep16" if layout == "pp13-ep16" else ""
        for rep in reps:
            for mode in ("sync", "async"):
                save_name = f"syn-f1-glm-{content}{tag}-{mode}-save-{where}-r{rep}"
                ts = T0 + 30 * 3600 + rep * 8 * 3600 + case * 300
                case += 1
                fields = dict(step=10, params_total=GLM["params_total"])
                save = _checkpoint("save", mode, content, size, nodes, rng, par["gpus"] + 4, "torch_dist", rep, **fields)
                write_run(root, save_name, "glm5", GLM, par, NODES[:nodes], step, rng, repetition=rep, t_start=ts,
                          variant=f"megatron-glm5-tri-ckpt-{content}{tag}-{mode}-save-{where}", checkpoints=[save],
                          n_steps=19, mem=train_mem * (1.5 if content == "full" else 1.0), **common)
                peak = round(train_mem * (1.54 if content == "full" else 1.0), 1)
                load = _checkpoint("load", "sync", content, size, nodes, rng, par["gpus"] + 4, "torch_dist", rep,
                                   peak_allocated_gib=peak, **fields)
                write_run(root, f"syn-f1-glm-{content}{tag}-load-{mode}-{where}-r{rep}", "glm5", GLM, par, NODES[:nodes], step,
                          rng, repetition=rep, t_start=ts + 900, variant=f"megatron-glm5-tri-ckpt-{content}{tag}-load-{where}",
                          checkpoints=[load], n_steps=4, first_loss_step=11, mem=peak, **common)

    for content in ("weights", "full"):
        for nodes in (13, GLM_TOP_NODES):
            glm_f1(content, nodes, "pp13-ep8", weak_batch[nodes], (1, 2, 3), 125)
    # Fallback if a full checkpoint does not fit at 13 nodes: half the state per GPU.
    glm_f1("full", 26, "pp13-ep16", 512, (1,), 70)

    # F2: long run, best Qwen layout on every node, 70 min; one GPU throttles for 8 minutes and
    # one node's CPU clock drops for 10 (the step slows by ~6 % meanwhile).
    layout = best[0]
    step, compute, mem = qwen_step(layout, N, 24576)
    t_long = T0 + 40 * 3600
    write_run(root, "syn-f2-long", "tt-qwen", QWEN, qwen_parallelism(layout, N), NODES, step, rng,
              experiment="F2", variant=f"{layout}-24k-long", repetition=1, t_start=t_long, tokens_gpu=24576,
              n_steps=int(70 * 60 / step), budget=4200, power_busy=compute / step + 0.05,
              nodemon_extra={"period": 60.0, "slow": (SLOW_GPU[0], SLOW_GPU[1], 1500, 1980),
                             "cpu_slow": ("trig0118", 3000, 3600), "cpu": True})
    _f2_dips(root / "runs" / "syn-f2-long" / "tt-qwen.0.data", ((1500, 1980), (3000, 3600)))


def _f2_dips(path: Path, windows) -> None:
    """Slow the long run's steps inside the straggler windows (seconds since the measurement start)."""
    events = [json.loads(line) for line in path.read_text().splitlines()]
    rates = [e["data"] for e in events if e["event"] == "data" and "rate" in e["data"]]
    t0 = rates[0]["time"] - rates[0]["elapsed"]
    for d in rates:
        if any(a <= d["time"] - t0 < b for a, b in windows):
            d["rate"] /= 1.06
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n")
