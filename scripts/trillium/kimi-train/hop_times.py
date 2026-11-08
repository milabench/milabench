"""NCCL kernel times of one profiled step (``--profile-freq``), per rank and CUDA stream, and its optimizer steps.

    python hop_times.py <run dir>/traces/iteration_<N>

Pipeline send/recv and the expert all-to-all both run as ``SendRecv`` kernels, on the stream of their own
communicator: per stream, the count tells them apart (PP2 with 1F1B: one forward and one backward hop per
micro-batch on each rank; all-to-all: several per MoE layer and micro-batch). A kernel lasts from its launch
until the peer has sent or received, so the shortest ones are closest to the transfer itself.
"""

import gzip
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def optimizer_steps(events: list[dict]) -> list[str]:
    """Each OptimizersContainer.step: CPU time, and the GPU span of the kernels it launched (correlation ids)."""
    launches = [e for e in events if e.get("cat") == "cuda_runtime" and "correlation" in e.get("args", {})]
    kernels = {e["args"]["correlation"]: e for e in events if e.get("cat") == "kernel" and "correlation" in e.get("args", {})}
    out = []
    for op in events:
        if op.get("name") != "Optimizer.step#OptimizersContainer.step" or op.get("ph") != "X":
            continue
        end = op["ts"] + op["dur"]
        ks = [kernels[c] for c in (e["args"]["correlation"] for e in launches
              if e.get("tid") == op.get("tid") and op["ts"] <= e["ts"] <= end) if c in kernels]
        span = (max(k["ts"] + k["dur"] for k in ks) - min(k["ts"] for k in ks)) / 1000 if ks else 0.0
        out.append(f"optimizer CPU {op['dur'] / 1000:.0f} ms, GPU span {span:.0f} ms ({len(ks)} kernels)")
    return out


def main(trace_dir: str) -> None:
    for path in sorted(Path(trace_dir).glob("rank*_trace.json*"), key=lambda p: int(p.name[4:].split("_")[0])):
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt") as f:
            events = json.load(f)["traceEvents"]
        step = max((e["dur"] for e in events if e.get("name", "").startswith("ProfilerStep#")), default=0)
        streams = defaultdict(lambda: defaultdict(list))
        for e in events:
            if e.get("cat") == "kernel" and "nccl" in e.get("name", "").lower():
                kind = e["name"].split("(")[0].replace("ncclDevKernel_", "").replace("ncclKernel_", "")
                streams[e.get("args", {}).get("stream", e.get("tid"))][kind].append(e["dur"] / 1000)
        print(f"{path.name}: step {step / 1e6:.2f} s; " + ", ".join(optimizer_steps(events)))
        for stream, kinds in sorted(streams.items(), key=lambda kv: str(kv[0])):
            for kind, ms in sorted(kinds.items()):
                ms.sort()
                print(
                    f"  stream {stream} {kind}: n={len(ms)} sum={sum(ms):.0f} ms min={ms[0]:.2f} "
                    f"p10={ms[len(ms) // 10]:.2f} median={statistics.median(ms):.2f} max={ms[-1]:.2f} ms"
                )


if __name__ == "__main__":
    main(sys.argv[1])
