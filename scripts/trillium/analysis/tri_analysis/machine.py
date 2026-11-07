"""Trillium B200 partition: constants shared by the parsers, the synthetic data and the figures."""

from __future__ import annotations

import os
from pathlib import Path

CLUSTER = "trillium"
GPUS_PER_NODE = 8
ALL_NODES = [f"trig{101 + i:04d}" for i in range(52)]  # trig0101 - trig0152

# THE setting: nodes available in the window. 52, or 47 if trig0148-trig0152 stay with SciNet's inference
# service (open question; DESIGN.md "If only 47 nodes"). Everything below, the synthetic data, the
# figures and the report's scales follow from it. TRI_N_NODES overrides it to try 47 without editing.
N_NODES = int(os.environ.get("TRI_N_NODES", 52))

NODES = ALL_NODES[:N_NODES]
GLM_COPY_NODES = 13  # one pp13-ep8 model copy
GLM_TOP_COPIES = N_NODES // GLM_COPY_NODES  # 4 at 52 nodes, 3 at 47
GLM_TOP_NODES = GLM_TOP_COPIES * GLM_COPY_NODES  # the largest pp13-ep8 run: 52, or 39
# Weak scaling and the layout comparison at a node count: 32 micro-batches per data-parallel rank of pp13-ep8.
GLM_SEQS_PER_COPY = 256
GLM_STRONG_SEQS = 768  # strong scaling: fixed global batch at every node count (96 / 48 / 32 / 24 micro-batches)
GLM_SMALL_SEQS = 64  # the small-batch point at 13 nodes: 8 micro-batches, ~60 % bubble
# A-G runs (layout, nodes, global batch); pp13-ep8 at 26 nodes is B-G-weak's run (select.SERVES_TWICE).
GLM_AG_RUNS = [("pp13-ep8", 13, 256), ("pp13-ep16", 26, 512), ("pp26-ep8", 26, 512),
               ("pp13-ep8", 52, 1024), ("pp13-ep16", 52, 1024), ("pp13-ep32", 52, 1024), ("pp26-ep8", 52, 1024),
               ("pp6-ep16", 12, 256), ("pp6-ep16", 48, 1024)]
# Added for the 40 clean nodes and the 50 available; no synthetic stand-ins: they come from measured runs only.
GLM_AG_RUNS_NEW = [("pp10-ep8", 10, 200), ("pp10-ep8", 20, 400), ("pp10-ep8", 30, 600), ("pp10-ep8", 40, 800), ("pp10-ep8", 50, 1000),
                   ("pp25-ep8", 25, 512), ("pp25-ep8", 50, 1024), ("pp6-ep16", 36, 768)]
GLM_DROPPED_BELOW_52 = {"pp26-ep8", "pp13-ep32"}  # no full-machine run left to compare them on
if N_NODES < 52:
    GLM_AG_RUNS = [r for r in GLM_AG_RUNS if r[1] <= N_NODES and r[0] not in GLM_DROPPED_BELOW_52]

# Network sweeps: the jobs that are the result, among the bring-up tests under nccl/. D2 on GPUs 0, 2, 4, 6 (each shares a PCIe
# switch with a NIC), two sets of 25 nodes; D3 up to 25 nodes, the job limit then.
NET_D2_JOBS = {1239766, 1239768}
NET_D2_ALL_GPUS_JOB = 1239660
NET_D3_JOB = 1239634
NET_D3_INSIDE_JOB = 1250124  # the same ladder to 26 nodes, all inside trig0101-0128; the first one crosses groups from 8 nodes up
# Nov 7, 50 nodes (all but trig0114, trig0138): D2 over all 1,225 pairs, D3 to 26 nodes (the 26-node step slow, its all-to-all hung,
# D3 at 50 cancelled), D4 in 3 cuts of 25 + 25.
NET_FULL_JOB = 1258790
# Nov 7, 15:08, 48 nodes (also without trig0125 and trig0132): D2 over all 1,128 pairs, twice. The Nov 4 groups, pair for pair;
# 1258790 (11:25-11:37) is the exception, inferred a temporary state of the fabric.
NET_D2_AGAIN_JOB = 1259968

# Checkpoint size per parameter, measured on truncated models (Rorqual, Fir): full 12.0, weights 2.0.
CKPT_FULL_B_PER_PARAM = 12.0
CKPT_WEIGHTS_B_PER_PARAM = 2.0
# /etc/hosts puts the nodes in 8 subnets (racks?), in host order. The switch layout is unknown.
# The admins' kernel change on 2026-11-05: boot option intel_pstate=passive (driver intel_pstate -> intel_cpufreq), rolled
# out by reboot after root's "Reboot ASAP" at 17:28:57. This file holds each node's first boot with it (scontrol's BootTime).
PSTATE_BOOTS = Path(__file__).resolve().parents[1] / "pstate_boots.tsv"

SUBNET_SIZES = [7, 7, 7, 7, 4, 7, 7, 6]
SUBNET_OF = {}
_i = 0
for _g, _n in enumerate(SUBNET_SIZES, 1):
    for _h in ALL_NODES[_i:_i + _n]:
        SUBNET_OF[_h] = f"s{_g}"
    _i += _n

# MFU denominator: B200 dense BF16 *spec* peak, for every precision recipe (MFU stays in BF16
# units so that recipes compare on one scale). SciNet measured 1,668 TFLOP/s BF16 and 3,041
# TFLOP/s FP8 on 8192^2 matmuls.
PEAK_BF16 = 2.25e15
MATMUL_BF16 = 1668e12
MATMUL_FP8 = 3041e12
NODE_NET_GBS = 200.0  # 4 x 400 Gb/s, the ports NCCL uses (measured 2026-11-04)
GPU_MEMORY_GIB = 180e9 / 2**30  # 180 GB
