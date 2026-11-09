"""One PyTorch-profiler trace of an OF3 run (variant prof): per profiled step, the GPU's busy time (union of kernels,
copies and sets on all streams) and how it splits: kernel classes by name (GEMMs by precision, attention kernels,
normalization, softmax, elementwise, reductions, copies, NCCL), forward kernels by model part (Lightning's
`[pl][module]` annotations; backward kernels outside a forward annotation, recompute included, are counted as
"backward"), NCCL time and its part not overlapped with other kernels, kernel count and launches. The traced steps run
slower than the others (profiler overhead on the CPU side); GPU times are the kernels' own.
Usage: python prof.py TRACE.json"""
import bisect
import collections
import json
import re
import sys

trace = json.load(open(sys.argv[1]))["traceEvents"]
steps = sorted((e for e in trace if e.get("name", "").startswith("ProfilerStep") and e.get("cat") == "user_annotation"),
               key=lambda e: e["ts"])
gpu = sorted((e for e in trace if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")), key=lambda e: e["ts"])
# kernel launches: runtime API, and driver API (cuBLAS and NCCL launch through cuLaunchKernelEx)
launch = {e["args"]["correlation"]: e for e in trace
          if e.get("cat") in ("cuda_runtime", "cuda_driver") and "correlation" in e.get("args", {})}
notes = collections.defaultdict(list)
for e in trace:
    if e.get("cat") == "gpu_user_annotation" and e["name"].startswith("[pl][module]"):
        notes[e["tid"]].append((e["ts"], e["ts"] + e["dur"], e["name"].split(": ", 1)[-1]))
for v in notes.values():
    v.sort()
starts = {k: [a[0] for a in v] for k, v in notes.items()}


def kind(name):
    n = name.lower()
    for label, pattern in [("NCCL", "nccl"), ("GEMM FP32 (SIMT)", "simt_sgemm|sgemm"), ("GEMM TF32", "tf32"),
                           ("attention kernels (cuEq, DeepSpeed, Triton, flash)",
                            "triangle|cueq|cuequivariance|attention_kernel|evoformer|fmha|flash|_attn_fwd|_attn_bwd"),
                           ("GEMM BF16/other", "gemm|nvjet|xmma|cutlass"), ("layer norm", "layer_norm|layernorm|rowwisemoments|gammabeta"),
                           ("softmax", "softmax"), ("reduction", "reduce_kernel"), ("copy, cast, cat, index",
                            "copy|cat|index|gather|scatter|memcpy|memset|fill"), ("elementwise", "elementwise"),
                           ("Triton (other)", "^triton_")]:
        if re.search(pattern, n):
            return label
    return "other"


def part(event):
    """Innermost module annotation over the kernel's start on its stream."""
    v, s = notes.get(event["tid"]), starts.get(event["tid"])
    if not v:
        return "backward and unannotated"
    i = bisect.bisect_right(s, event["ts"]) - 1
    best = None
    while i >= 0 and i >= bisect.bisect_right(s, event["ts"]) - 400:
        a, b, path = v[i]
        if a <= event["ts"] < b and (best is None or b - a < best[1] - best[0]):
            best = (a, b, path)
        i -= 1
    if best is None:
        return "backward and unannotated"
    p = best[2]
    for label, pattern in [("triangle multiplication", "tri_mul"), ("triangle attention", "tri_att"),
                           ("MSA module", "msa_module"), ("template module", "template_embedder"),
                           ("pairformer, other", "pairformer_stack"), ("diffusion module", "diffusion_module"),
                           ("confidence heads", "aux_heads|confidence|heads"), ("input embedder", "input_embedder")]:
        if re.search(pattern, p):
            return label
    return "other modules"


def union(intervals):
    total, end = 0.0, -1.0
    for a, b in sorted(intervals):
        if b > end:
            total += b - max(a, end)
            end = b
    return total


for s in steps:
    t0, t1 = s["ts"], s["ts"] + s["dur"]
    # GPU work launched during this step (the GPU runs behind the CPU)
    own = [e for e in gpu if e.get("args", {}).get("correlation") in launch
           and t0 <= launch[e["args"]["correlation"]]["ts"] < t1]
    busy = union([(e["ts"], e["ts"] + e["dur"]) for e in own])
    span = max(e["ts"] + e["dur"] for e in own) - min(e["ts"] for e in own)
    nccl = [(e["ts"], e["ts"] + e["dur"]) for e in own if "nccl" in e["name"].lower()]
    other = [(e["ts"], e["ts"] + e["dur"]) for e in own if "nccl" not in e["name"].lower()]
    exposed = busy - union(other)
    by_kind, by_part = collections.Counter(), collections.Counter()
    for e in own:
        by_kind[kind(e["name"])] += e["dur"]
        if e.get("cat") == "kernel" and "nccl" not in e["name"].lower():
            by_part[part(e)] += e["dur"]
    total = sum(by_kind.values())
    kernels = [e["dur"] for e in own if e.get("cat") == "kernel"]
    kernels.sort()
    launches = [launch[e["args"]["correlation"]]["dur"] for e in own]
    print(f"{s['name']}: CPU {s['dur'] / 1e6:.3f} s, GPU span {span / 1e6:.3f} s, busy {busy / 1e6:.3f} s"
          f" ({busy / span:.1%} of the span); {len(kernels)} kernels, median {kernels[len(kernels) // 2]:.1f} us,"
          f" {sum(k < 10 for k in kernels) / len(kernels):.0%} under 10 us; launch calls {sum(launches) / 1e6:.3f} s on the CPU")
    spans, end, gaps = sorted((e["ts"], e["ts"] + e["dur"]) for e in own), None, collections.Counter()
    for a, b in spans:
        if end is not None and a > end:
            gap = a - end
            gaps["<10 us" if gap < 10 else "10-100 us" if gap < 100 else "0.1-1 ms" if gap < 1000 else ">1 ms"] += gap
        end = b if end is None else max(end, b)
    print("  GPU idle within the span, by gap length: " + ", ".join(
        f"{k} {gaps[k] / 1e6:.3f} s" for k in ("<10 us", "10-100 us", "0.1-1 ms", ">1 ms")))
    print(f"  NCCL {union(nccl) / 1e6:.3f} s, not overlapped {exposed / 1e6:.3f} s; NCCL kernels in order, ms (the first ones"
          f" also wait for the slowest rank): " + " ".join(f"{(b - a) / 1e3:.1f}" for a, b in sorted(nccl)))
    print("  by kernel class: " + ", ".join(f"{k} {v / 1e6:.3f} s ({v / total:.0%})" for k, v in by_kind.most_common()))
    compute = sum(by_part.values())
    print("  compute kernels by part: " + ", ".join(f"{k} {v / 1e6:.3f} s ({v / compute:.0%})" for k, v in by_part.most_common()))
