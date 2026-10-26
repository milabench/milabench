"""Synthetic data in the target formats (FORMAT.md), for experiments not run yet.

Writes milabench run data (``milabench.pretrain.v1`` summaries + per-step
``rate`` events), srygaard health-check directories and a DCGM export under
``data/synthetic/<cluster>/``. Numbers come from a toy model calibrated on
the preliminary measurements (step = compute + exposed communication); they
only exist to exercise the pipeline and the figures. Every row built from
them carries ``synthetic=True`` and is drawn as such.
"""

from __future__ import annotations

import json
import math
import shutil
import zlib
from pathlib import Path

import numpy as np

from .topology import load_topology

SCHEMA = "milabench.pretrain.v1"
PEAK = 989e12
T0 = 1793361600.0  # 2026-10-30 08:00 EDT

QWEN = {
    "model": "qwen3_8-27B", "framework": "torchtitan", "n_layers_full": 64,
    "flops_per_token": 159258915840, "seq_len": 4096, "microbatch_sequences": 3,
    "precision": "bf16", "optimizer": "AdamW",
}
GLM = {
    "model": "glm5", "framework": "megatron", "n_layers_full": 78,
    "flops_per_token": 6 * 40e9, "seq_len": 4096, "microbatch_sequences": 2,
    "precision": "bf16", "optimizer": "AdamW",
}
DEVIATIONS = [
    {"id": "optimizer", "text": "AdamW (not Muon)"},
    {"id": "mtp", "text": "MTP off"},
    {"id": "data", "text": "random tokens"},
]
# Expected throughput (tokens/s per GPU, Qwen3.8-27B, 12,288 tokens per GPU, FP32
# moments), anchored on measurements: Rorqual 12 GPUs 1228 (P3), 24 GPUs 1213 (H6 = S6
# within 0.2 %), 48 GPUs 1160 (6+6 across ihpc218/316); Fir 12 GPUs 2148 (P3).
# Interpolated between anchors in log2(GPUs); beyond the last anchor, the average decline
# per doubling of Rorqual 12 -> 48 GPUs (-2.8 %), i.e. ~1080 at 252 GPUs (extrapolation).
ANCHORS = {"rorqual": ([12, 24, 48], [1228, 1213, 1160]),
           "fir": ([12, 24, 48], [2148, 2135, 2115]), "tamia": ([12, 24, 48], [2140, 2127, 2105])}
DECLINE_PER_DOUBLING = {"rorqual": 0.0277, "fir": 0.006, "tamia": 0.006}
SMALL_LEAVES = {f"ihpc{i}u25" for i in (134, 135, 136, *range(317, 327))}


def full_gpu_nodes(gres_file: Path) -> set[str] | None:
    """Nodes whose Slurm Gres is 4 full H100 (gpu:h100:4), from `node_gres.txt` ("<node> <Gres>"
    lines, written by analyze.py collect); MIG-partitioned nodes have per-slice Gres instead."""
    if not gres_file.exists():
        return None
    nodes = set()
    for line in gres_file.read_text().splitlines():
        node, _, gres = line.partition(" ")
        if gres.strip().split("(")[0] == "gpu:h100:4":
            nodes.add(node)
    return nodes


def rorqual_full_nodes(leaf_of: dict[str, str], full: set[str] | None = None) -> dict[str, list[str]]:
    """Leaf -> full (non-MIG) GPU nodes. From the nodes' Gres when known; otherwise rg nodes outside
    leaves ihpc125-133 (all MIG on 2026-10-23)."""
    mig = {f"ihpc{i}u25" for i in range(125, 134)}
    by_leaf: dict[str, list[str]] = {}
    for host, leaf in sorted(leaf_of.items()):
        if (host in full) if full is not None else (host.startswith("rg") and leaf not in mig):
            by_leaf.setdefault(leaf, []).append(host)
    return by_leaf


def pick(family: str, n: int, by_leaf: dict[str, list[str]]) -> list[str]:
    big = sorted((l for l in by_leaf if len(by_leaf[l]) > 3), key=lambda l: -len(by_leaf[l]))
    small = sorted(l for l in by_leaf if len(by_leaf[l]) == 3)
    if family == "S":
        return by_leaf["ihpc316u25"][:n]
    if family == "H":  # 3-node leaves first (the plan's H); big leaves only beyond 13 leaves
        leaves = small + big
        return [h for l in leaves[: n // 3] for h in by_leaf[l][:3]]
    if family == "O":  # one node per switch
        return [by_leaf[l][0] for l in (small + big)[:n]]
    out = []  # P / ALL: biggest leaves first
    for l in big + small:
        out += by_leaf[l][: n - len(out)]
        if len(out) == n:
            break
    return out


def expected_tok_s_gpu(cluster, hosts, leaf_of):
    """Expected throughput for a placement; placement effects kept within 1 %, as measured."""
    xs, ys = ANCHORS[cluster]
    gpus = 4 * len(hosts)
    lx = np.log2(xs)
    tok = float(np.interp(math.log2(gpus), lx, ys))
    if gpus > xs[-1]:
        tok = ys[-1] * (1 - DECLINE_PER_DOUBLING[cluster] * math.log2(gpus / xs[-1]))
    leaves = {leaf_of[h] for h in hosts}
    if len(leaves) == len(hosts) > 1:
        # One node per switch: every ring hop crosses the spine. Guess: up to ~1.5 % at 16 leaves.
        tok *= 1 - (0.004 if cluster == "rorqual" else 0.001) * math.log2(len(leaves))
    if cluster == "rorqual" and len(leaves) > 1:
        if leaves & SMALL_LEAVES:
            tok *= 0.997  # small leaves share their uplinks with CPU-node jobs (hypothesis)
        if "ihpc316u25" in leaves:
            tok *= 0.995  # ihpc316 uplinks slower in NCCL tests; rings cross them only twice
    return tok


def step_time(cluster, hosts, leaf_of, rng, tokens_per_gpu=12288):
    # Run-to-run spread ~0.1 % (repeats in one allocation measured within 0.1 %).
    return tokens_per_gpu / expected_tok_s_gpu(cluster, hosts, leaf_of) * (1 + rng.normal(0, 0.001))


def write_run(root, cluster, name, bench, w, hosts, leaf_of, step_s, rng, *, variant, family,
              repetition, campaign="window-1030", n_layers=None, rc=0, n_steps=30, t_start=T0,
              excluded=None, dip=None, ep=1, ep_groups=None, ep_group_a2a=None, trainable=1.0):
    gpus = 4 * len(hosts)
    n_layers = n_layers or w["n_layers_full"]
    tokens_gpu = w["seq_len"] * w["microbatch_sequences"]
    global_tokens = tokens_gpu * gpus
    run_dir = root / cluster / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    events = [
        {"event": "config", "data": {"run_name": name}},
        {"event": "start", "data": {"time": t_start}},
    ]
    t = t_start + 40 + 30 * step_s
    measure_start = t
    for i in range(n_steps if rc == 0 else n_steps // 3):
        s = step_s * (1 + rng.normal(0, 0.0015))
        if dip and dip[0] <= t - measure_start < dip[1]:
            s *= dip[2]
        t += s
        events.append({"event": "data", "data": {
            "rate": global_tokens / s, "units": "items/s", "task": "train",
            "time": t, "elapsed": s, "step": 30 + i + 1}})
    if rc == 0:
        summary = {
            "schema": SCHEMA,
            "run": {"cluster": cluster, "campaign": campaign, "variant": variant,
                    "repetition": repetition, "slurm_job_id": str(1030000 + zlib.crc32(name.encode()) % 9999)},
            "workload": {**{k: w[k] for k in ("model", "framework", "n_layers_full", "seq_len",
                                               "microbatch_sequences", "precision", "optimizer")},
                         "n_layers": n_layers, "grad_accum": 1, "params_trainable_fraction": trainable,
                         "tokens_per_gpu_per_step": tokens_gpu, "global_batch_tokens": global_tokens,
                         "compile": "regions", "activation_checkpointing": "full",
                         "optimizer_state_dtype": "fp32", "torch_version": "2.15.0.dev20261005+cu130"},
            "parallelism": {"gpus": gpus, "nodes": len(hosts), "dp_shard": gpus // ep, "ep": ep},
            "placement": {"hosts": hosts, "family": family, "excluded_nodes": excluded or {},
                          **({"ep_groups": ep_groups} if ep_groups else {})},
            **({"per_ep_group": [{"hosts": g, "a2a_s_per_step": a}
                                 for g, a in zip(ep_groups, ep_group_a2a)]} if ep_group_a2a else {}),
            "measurement": {"warmup_steps": 30, "measured_steps": n_steps,
                            "measurement_start_epoch": measure_start, "measurement_end_epoch": t},
            "mfu_inputs": {"flops_per_token": w["flops_per_token"] * n_layers / w["n_layers_full"],
                           "peak_flops_per_gpu": PEAK},
            "memory_gib": {"peak_allocated": {"min": 30.0, "max": 31.0},
                           "peak_reserved": {"min": 44.0, "max": 45.0}},
            "startup": {"model_initialized": {"max_s": 10.5 + 0.03 * len(hosts) + rng.normal(0, 0.3)},
                        "first_step_done": {"max_s": 29 + 0.08 * len(hosts) + rng.normal(0, 0.5)},
                        "measurement_start": {"max_s": measure_start - t_start}},
            "per_rank": [],
            "deviations": DEVIATIONS,
        }
        events.append({"event": "data", "data": {"summary": summary, "task": "train"}})
    events.append({"event": "end", "data": {"time": t, "return_code": rc}})
    (run_dir / f"{bench}.0.data").write_text("\n".join(json.dumps(e) for e in events) + "\n")
    return t


def write_srygaard(root, cluster, job, by_leaf, rng, bad=None, mode="switch", whole=None):
    """One healthcheck.py run: --job-per switch (a group test per leaf) or cluster (one over all
    nodes; `whole` = its (all-reduce, all-to-all) peak busbw)."""
    run = root / cluster / "srygaard" / job
    (run / "groups").mkdir(parents=True, exist_ok=True)
    bad = bad or {}
    hosts = [h for l in by_leaf for h in by_leaf[l]]
    (run / "switches.json").write_text(json.dumps(by_leaf))
    sizes = [2**k for k in range(3, 34, 2)]

    def rows(peak):
        return [{"size": s, "busbw": round(peak * s / (s + 8 * 2**20), 2)} for s in sizes]

    tests = []
    leaf_of = {h: l for l, hs in by_leaf.items() for h in hs}
    for h in hosts:
        level = 321 if leaf_of[h] == "ihpc316u25" else 338
        tests.append({"kind": "intra", "nodes": [h], "status": "ok",
                      "peak_busbw": round(bad.get(h, level) + rng.normal(0, 1.5), 1)})
        tests.append({"kind": "loopback", "nodes": [h], "status": "ok",
                      "peak_busbw": round(6.4 + rng.normal(0, 0.1), 2)})
    for a, b in zip(hosts, hosts[1:] + hosts[:1]):
        slow = {a, b} & {"rg21706", "rg31606"}  # slow even alone in the uplink tests (10-24)
        tests.append({"kind": "pair", "nodes": [a, b], "status": "ok",
                      "peak_busbw": round((25 if slow else 40) + rng.normal(0, 0.6), 1)})
    for leaf, hs in (by_leaf.items() if mode == "switch" else ()):
        peak = 24.6 + rng.normal(0, 0.1)
        tests.append({"kind": "group", "nodes": hs, "status": "ok", "peak_busbw": round(peak, 2),
                      "rows": rows(peak)})
        tests.append({"kind": "alltoall", "nodes": hs, "status": "ok",
                      "peak_busbw": round(6.8 + rng.normal(0, 0.1), 2)})
    for kind, peak in (zip(("group", "alltoall"), whole) if mode == "cluster" else ()):
        tests.append({"kind": kind, "nodes": hosts, "status": "ok", "peak_busbw": peak})
    (run / "groups" / f"{job}.json").write_text(json.dumps({"job_id": job, "nodes": hosts, "tests": tests}))
    report = {"nodes": {h: {
        "status": "fail" if h in bad else "pass", "switch": leaf_of[h],
        "findings": [{"severity": "fail", "check": "nccl.intra", "detail": f"busbw {bad[h]} < 300"}]
        if h in bad else []} for h in hosts}}
    (run / "report.json").write_text(json.dumps(report))


def write_dcgm(root, cluster, name, hosts, t0, t1, slow=None):
    """DCGM export (Prometheus range query flattened): time, host, gpu, metric, value."""
    rng = np.random.default_rng(7)
    lines = ["time,host,gpu,metric,value"]
    for t in np.arange(t0, t1, 30.0):
        for h in hosts:
            for g in range(4):
                is_slow = slow and h == slow[0] and g == slow[1] and slow[2] <= t - t0 < slow[3]
                clock = 1410 if is_slow else 1755 + rng.normal(0, 8)
                tensor = 0.27 if is_slow else (0.19 if slow and slow[2] <= t - t0 < slow[3] else 0.21)
                for metric, value in (
                    ("DCGM_FI_DEV_SM_CLOCK", clock),
                    ("DCGM_FI_DEV_POWER_USAGE", (430 if is_slow else 560) + rng.normal(0, 15)),
                    ("DCGM_FI_DEV_GPU_TEMP", (84 if is_slow else 62) + rng.normal(0, 1.5)),
                    ("DCGM_FI_PROF_PIPE_TENSOR_ACTIVE", tensor + rng.normal(0, 0.005)),
                ):
                    lines.append(f"{t:.0f},{h},{g},{metric},{value:.3f}")
    out = root / cluster / "dcgm"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.csv").write_text("\n".join(lines) + "\n")


def generate(raw: Path, out: Path) -> None:
    if out.exists():
        shutil.rmtree(out)
    rng = np.random.default_rng(1030)
    leaf_of = load_topology(raw / "rorqual" / "topology.txt")
    by_leaf = rorqual_full_nodes(leaf_of, full_gpu_nodes(raw / "rorqual" / "node_gres.txt"))
    excluded = {"rg32101": "intra-node all-reduce 281 < 90% of 336 GB/s"}
    healthy = {l: [h for h in hs if h not in excluded] for l, hs in by_leaf.items()}
    healthy = {l: hs for l, hs in healthy.items() if len(hs) >= 3 or len(by_leaf[l]) > 3}
    # Window plan: per leaf (default), whole machine, and the 217+316 pair of leaves.
    # Cross-leaf values from the colleague's NCCL tests (10-24): 21 nodes on 3 leaves 19.2 / 3.0,
    # 7+7 on 217+316 15.9 (all-to-all halves with 316).
    write_srygaard(out, "rorqual", "1030001", by_leaf, rng, bad={"rg32101": 281})
    write_srygaard(out, "rorqual", "1030002", by_leaf, rng, bad={"rg32101": 281}, mode="cluster",
                   whole=(17.5, 2.6))
    write_srygaard(out, "rorqual", "1030003", {l: by_leaf[l][:7] for l in ("ihpc217u25", "ihpc316u25")},
                   rng, mode="cluster", whole=(15.9, 3.4))

    t = T0 + 600
    # Nodes per switch at 3-12 nodes (S on ihpc316, H on 3-node leaves, O = 1 per switch, up
    # to 16), then the at-scale pairs (H48 falls back to H45: one 3-node leaf lost to rg32101).
    plan = [(f, n) for n in (3, 6, 9, 12) for f in ("S", "H", "O")] + [("O", 16)]
    plan += [("H", 15), ("P", 15), ("H", 27), ("P", 27), ("H", 45), ("P", 48), ("ALL", 63)]
    for family, n in plan:
        hosts = pick("P" if family == "ALL" else family, n, healthy)
        for rep in (1, 2):
            variant = f"{family}{n}"
            rc = 1 if (variant == "H45" and rep == 2) else 0
            t = write_run(out, "rorqual", f"syn-{variant.lower()}-{rep}", f"tt-w-n{n}", QWEN, hosts,
                          leaf_of, step_time("rorqual", hosts, leaf_of, rng), rng, variant=variant,
                          family=family, repetition=rep, rc=rc, t_start=t, excluded=excluded) + 60
    # Long run on every healthy node, with one GPU throttling for 10 minutes.
    hosts = pick("P", 63, healthy)
    t_long = t
    s = step_time("rorqual", hosts, leaf_of, rng)
    t_end = write_run(out, "rorqual", "syn-all63-long", "tt-w-n63", QWEN, hosts, leaf_of, s, rng,
                      variant="ALL63-long", family="ALL", repetition=1, n_steps=int(3600 / s),
                      t_start=t_long, excluded=excluded, dip=(2100, 2700, 1.09))
    measure_start = t_long + 40 + 30 * s
    write_dcgm(out, "rorqual", "syn-all63-long", hosts, measure_start, t_end,
               slow=("rg31605", 2, 2100, 2700))

    # Same placement families on the 4-HCA clusters, on their real topologies: Fir has 8 GPU
    # leaves of 20 nodes (C1xx), Tamia 5 leaves of 8-16. S and H cover 3-12 nodes on both; 1 per
    # switch is limited by the leaf count (Fir 8, Tamia 5).
    for cluster, prefix in (("fir", "fc1"), ("tamia", "tg")):
        lo_all = load_topology(raw / cluster / "topology.txt")
        lo = {h: l for h, l in lo_all.items() if h.startswith(prefix)}
        leaves: dict[str, list[str]] = {}
        for h, l in sorted(lo.items()):
            leaves.setdefault(l, []).append(h)
        by_size = sorted(leaves, key=lambda l: (-len(leaves[l]), l))
        plan = [("S", n) for n in (3, 6, 9, 12)] + [("H", n) for n in (3, 6, 9, 12) if n // 3 <= len(leaves)]
        plan += [("O", n) for n in (3, 6, 9, 12, 16) if n <= len(leaves)]
        if len(leaves) not in (3, 6, 9, 12, 16):
            plan.append(("O", len(leaves)))
        for family, n in plan:
            if family == "S":
                hosts = leaves[by_size[0]][:n]
            elif family == "H":
                hosts = [h for l in by_size[1:1 + n // 3] or by_size[: n // 3] for h in leaves[l][:3]]
            else:
                hosts = [leaves[l][-1] for l in by_size[:n]]
            for rep in (1, 2):
                variant = f"{family}{n}"
                write_run(out, cluster, f"syn-{cluster}-{variant.lower()}-{rep}", f"tt-w-n{n}", QWEN, hosts,
                          lo, step_time(cluster, hosts, lo, rng), rng, variant=variant, family=family,
                          repetition=rep, campaign="prelim-synthetic")
        (out / cluster / "topology.txt").write_text((raw / cluster / "topology.txt").read_text())

    # GLM-5 (MoE, Megatron): truncated models at small scale, full model at 52+ nodes;
    # expert parallelism inside a node (EP4) or across nodes (EP8, EP16); FP8 vs BF16.
    # Guesses, not anchored on any measurement: MFU ~20 % with experts inside the node,
    # far lower once all-to-all crosses the single HCA (~6.8 GB/s busbw in a leaf, ~3 across);
    # FP8 speeds up compute only, so it helps little when all-to-all dominates.
    t = t_end + 600
    base_mfu = {4: 0.20, 8: 0.11, 16: 0.08}
    fp8_gain = {4: 1.20, 8: 1.07, 16: 1.04}
    for n, layers in ((4, 12), (8, 24), (16, 40), (52, 78), (63, 78)):
        hosts = pick("P", n, healthy)
        for ep in (4, 8, 16):
            for prec in ("bf16", "fp8"):
                w = {**GLM, "precision": prec}
                mfu = base_mfu[ep] * (fp8_gain[ep] if prec == "fp8" else 1) * (1 - 0.025 * math.log2(n / 4))
                flops = GLM["flops_per_token"] * layers / GLM["n_layers_full"]
                tokens = GLM["seq_len"] * GLM["microbatch_sequences"]
                st = flops * tokens / (mfu * PEAK) * (1 + rng.normal(0, 0.002))
                variant = f"GLM-EP{ep}-{prec}"
                t = write_run(out, "rorqual", f"syn-glm-n{n}-ep{ep}-{prec}", f"glm5-n{n}", w, hosts,
                              leaf_of, st, rng, variant=variant, family="P", repetition=1,
                              n_layers=layers, t_start=t, ep=ep) + 60
    # Expert-group placement (GLM-5 full model, EP group = 8 GPUs = 2 nodes), at the largest scale
    # where all three layouts fit on the same nodes: an even number of nodes per leaf (13 x 2 on the
    # 3-node leaves + 12 + 8 + 6 = 52 nodes, 208 GPUs). Megatron forms groups from consecutive ranks,
    # so only the node order differs. Guesses: all-to-all through the HCA costs 0.6x the no-network
    # step, crossing a switch 1.0x; in the mixed layout the crossing groups gate the step (0.9x:
    # fewer crossings share the uplinks).
    t = expert_layouts(out, "rorqual", by_leaf, leaf_of, rng, t + 600, excluded=excluded,
                       cost={"aligned": 0.6, "crossed": 1.0})
    (out / "rorqual" / "topology.txt").write_text((raw / "rorqual" / "topology.txt").read_text())


def expert_layouts(out, cluster, by_leaf, leaf_of, rng, t, *, excluded, cost, campaign="window-1030"):
    """EP8 aligned / mixed / crossed on the same even-per-leaf node set (three node orders), and
    mixed on every healthy node (the largest achievable run)."""
    w = {**GLM, "precision": "bf16"}
    layers = GLM["n_layers_full"]
    tokens = GLM["seq_len"] * GLM["microbatch_sequences"]
    base = GLM["flops_per_token"] * tokens / (0.20 * PEAK)
    usable = {l: [h for h in hs if h not in excluded] for l, hs in sorted(by_leaf.items())}
    aligned = [h for hs in usable.values() for h in hs[: len(hs) // 2 * 2]]  # leaf order, even per leaf
    half = len(aligned) // 2
    crossed = [h for i in range(half) for h in (aligned[i], aligned[half + i])]  # pairs from two leaves
    # Mixed: aligned order, except that 3 pairs of 2-node leaves swap partners (A1 B1, A2 B2): about
    # one group in four crosses, close to the realistic full-machine share (6 of 31).
    mixed = list(aligned)
    two = [l for l, hs in usable.items() if len(hs[: len(hs) // 2 * 2]) == 2]
    for la, lb in list(zip(two[0::2], two[1::2]))[:3]:
        ia, ib = mixed.index(usable[la][0]), mixed.index(usable[lb][0])
        a1, a2, b1, b2 = mixed[ia], mixed[ia + 1], mixed[ib], mixed[ib + 1]
        mixed[ia:ia + 2], mixed[ib:ib + 2] = [a1, b1], [a2, b2]
    # Largest achievable: every healthy node in leaf order (Megatron's default), an even count.
    every = [h for hs in usable.values() for h in hs]
    every = every[: len(every) // 2 * 2]
    for variant, hosts in (("EP8-aligned", aligned), ("EP8-mixed", mixed), ("EP8-crossed", crossed),
                           ("EP8-mixed-all", every)):
        groups = [hosts[i:i + 2] for i in range(0, len(hosts) - 1, 2)]
        a2a = []
        for g in groups:
            crossing = len({leaf_of[h] for h in g}) > 1
            if variant.startswith("EP8-mixed"):
                # More groups cross at once on the full machine: slightly more uplink contention.
                share = 0.92 if variant == "EP8-mixed-all" else 0.9
                a2a.append((share * cost["crossed"] if crossing else cost["aligned"]) * base)
            else:
                a2a.append(cost["aligned" if variant == "EP8-aligned" else "crossed"] * base)
        a2a = [x * (1 + rng.normal(0, 0.01)) for x in a2a]
        # Larger data-parallel group: the measured Qwen decline (-2.8 % per doubling of nodes).
        dp = 1 + 0.0277 * math.log2(len(hosts) / len(aligned))
        for rep in (1, 2):
            step = (base + max(a2a)) * dp * (1 + rng.normal(0, 0.002))
            t = write_run(out, cluster, f"syn-{cluster}-glm-{variant.lower()}-{rep}", f"glm5-n{len(hosts)}",
                          w, hosts, leaf_of, step, rng, variant=variant, family="EP", repetition=rep,
                          n_layers=layers, t_start=t, ep=8, campaign=campaign, excluded=excluded,
                          ep_groups=groups, ep_group_a2a=a2a) + 60
    return t
