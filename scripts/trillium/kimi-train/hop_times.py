"""NCCL kernel times of one profiled step (``--profile-freq``), per rank and CUDA stream.

    python hop_times.py <run dir>/traces/iteration_<N>

Pipeline send/recv and the expert all-to-all both run as ``SendRecv`` kernels, on the stream of their own
communicator: per stream, the count tells them apart (PP2 with 1F1B: one forward and one backward hop per
micro-batch on each rank; all-to-all: several per MoE layer and micro-batch). A kernel lasts from its launch
until the peer has sent or received, so the shortest ones are closest to the transfer itself.
"""

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def main(trace_dir: str) -> None:
    for path in sorted(Path(trace_dir).glob("rank*_trace.json"), key=lambda p: int(p.name[4:].split("_")[0])):
        events = json.loads(path.read_text())["traceEvents"]
        step = max((e["dur"] for e in events if e.get("name", "").startswith("ProfilerStep#")), default=0)
        streams = defaultdict(lambda: defaultdict(list))
        for e in events:
            if e.get("cat") == "kernel" and "nccl" in e.get("name", "").lower():
                kind = e["name"].split("(")[0].replace("ncclDevKernel_", "").replace("ncclKernel_", "")
                streams[e.get("args", {}).get("stream", e.get("tid"))][kind].append(e["dur"] / 1000)
        print(f"{path.name}: step {step / 1e6:.2f} s")
        for stream, kinds in sorted(streams.items(), key=lambda kv: str(kv[0])):
            for kind, ms in sorted(kinds.items()):
                ms.sort()
                print(
                    f"  stream {stream} {kind}: n={len(ms)} sum={sum(ms):.0f} ms min={ms[0]:.2f} "
                    f"p10={ms[len(ms) // 10]:.2f} median={statistics.median(ms):.2f} max={ms[-1]:.2f} ms"
                )


if __name__ == "__main__":
    main(sys.argv[1])
