"""K3-XN: GPU kernel time per decode step, by kind, from one rank's torch profiler trace (vLLM's --profiler-config).

    python3 profile_summary.py TRACE.pt.trace.json.gz [...]

Kernel durations summed by kind (regexes below), divided by the steps in the trace: the MLA attention kernel's calls / 24
(K3 has 24 MLA layers, one call each per step). Overlapping streams make the sum exceed a step's wall time a little.
"""
import collections
import gzip
import json
import re
import sys

KINDS = [
    ("all-to-all: DeepEP dispatch", r"dispatch_impl|dispatch_copy_epilogue"),
    ("all-to-all: DeepEP combine", r"combine_impl|combine_reduce_epilogue"),
    ("all-to-all: NCCL all-gather", r"ncclDevKernel_AllGather"),
    ("all-to-all: NCCL reduce-scatter", r"ncclDevKernel_ReduceScatter"),
    ("NCCL other", r"ncclDevKernel"),
    ("TP collectives in the node", r"lamport|allreduce|all_reduce|AllReduce|oneshot|twoshot"),
    ("MoE expert GEMMs", r"^bmm_|MxE2m1"),
    ("MoE routing, top-k, finalize", r"topk|routing|finalize|moe_align|permute"),
    ("attention (MLA, KDA)", r"fmha|kda|Kda|KDA|flash|mla|Mla|attn"),
    ("dense GEMMs", r"nvjet|gemm|Gemm|splitKreduce|cutlass"),
]


def summarize(path):
    with gzip.open(path, "rt") as f:
        events = json.load(f)["traceEvents"]
    total = collections.Counter()
    calls = collections.Counter()
    mla_calls = 0
    for e in events:
        if e.get("cat") != "kernel":
            continue
        name = e.get("name", "")
        kind = next((k for k, rx in KINDS if re.search(rx, name)), "other")
        total[kind] += e.get("dur", 0)
        calls[kind] += 1
        if name.startswith("fmhaSm100"):
            mla_calls += 1
    steps = max(mla_calls / 24, 1)
    print(f"{path.split('/')[-1]}: {steps:.0f} steps, {sum(total.values()) / steps / 1000:.1f} ms of kernels per step")
    for kind, us in total.most_common():
        print(f"  {kind:34s} {us / steps / 1000:7.2f} ms/step  {calls[kind] / steps:6.1f} calls/step")


for p in sys.argv[1:]:
    summarize(p)
