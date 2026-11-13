"""Streaming read of one profiler trace (no full JSON load: a 1.4 GB trace in about 1 GB of memory), for
ProfilerStep#27 and #28: their absolute start and end against the logged step ends of STEPS (rank 0's
steps-r0.jsonl); the GPU work launched in each (kernels, copies, sets, matched to their launch call by correlation,
runtime or driver API): span, busy time, idle time by gap length (as prof.py); the launch calls' CPU time by call
length, and the GPU backlog each launch met (start of its GPU work minus the end of the call), overall and for calls
over 100 us; busy time without NCCL (rank 0's NCCL kernels mostly wait for the other ranks); for every idle gap of
10 us or more, whether the GPU event that ends it was launched by a call that returned inside the gap (the GPU
waiting on the host), and how much of the 10-100 us gap time the launch calls themselves cover; GPU-bound stretches:
runs of consecutive calls whose GPU work started over 1 ms after the call returned.
Launches and backlogs are counted per GPU event, which is per call in these traces (no cudaGraphLaunch, at most one
GPU event per correlation); with CUDA graphs one call launches many GPU events and would be counted many times.
Usage: python trace_gaps.py TRACE STEPS"""
import bisect
import json
import sys

dec = json.JSONDecoder()


def events(f):
    """The objects of the trace's traceEvents array, one at a time."""
    buf = ""
    while '"traceEvents"' not in buf:
        buf += f.read(1 << 20)
    buf = buf[buf.index("[", buf.index('"traceEvents"')) + 1:]
    pos, eof = 0, False
    while True:
        if not eof and len(buf) - pos < (1 << 20):
            chunk = f.read(1 << 22)
            eof = not chunk
            buf, pos = buf[pos:] + chunk, 0
        while pos < len(buf) and buf[pos] in " \n\r\t,":
            pos += 1
        if pos >= len(buf) or buf[pos] == "]":
            return
        try:
            obj, end = dec.raw_decode(buf, pos)
        except json.JSONDecodeError:
            if eof:
                raise
            chunk = f.read(1 << 22)
            eof = not chunk
            buf, pos = buf[pos:] + chunk, 0
            continue
        yield obj
        pos = end


path, steps_path = sys.argv[1:3]
with open(path) as f:
    head = f.read(1 << 16)
base = int(head[head.index('"baseTimeNanoseconds"'):].split(":", 1)[1].split(",", 1)[0]) / 1e9
steps, gpu, launch = {}, [], {}
with open(path) as f:
    for e in events(f):
        cat = e.get("cat")
        if cat == "user_annotation" and e.get("name", "").startswith("ProfilerStep"):
            steps[e["name"]] = (e["ts"], e["dur"])
        elif cat in ("kernel", "gpu_memcpy", "gpu_memset"):
            gpu.append((e["ts"], e["dur"], e.get("args", {}).get("correlation"),
                        cat == "kernel" and "nccl" in e["name"].lower(), cat == "kernel"))
        elif cat in ("cuda_runtime", "cuda_driver") and "correlation" in e.get("args", {}):
            launch[e["args"]["correlation"]] = (e["ts"], e["dur"])
logged = {r["step"]: r["t"] for r in map(json.loads, open(steps_path)) if "step" in r}
for name in sorted(steps, key=lambda n: steps[n][0]):
    t0, dur = steps[name]
    a, b = base + t0 / 1e6, base + (t0 + dur) / 1e6
    end = min(logged, key=lambda s: abs(logged[s] - b))
    start = min(logged, key=lambda s: abs(logged[s] - a))
    print(f"{name}: {dur / 1e6:.3f} s, starts {a - logged[start]:+.3f} s from logged step {start}'s end, ends"
          f" {b - logged[end]:+.3f} s from logged step {end}'s end")
for name in ("ProfilerStep#27", "ProfilerStep#28"):
    t0, dur = steps[name]
    own = sorted((g for g in gpu if g[2] in launch and t0 <= launch[g[2]][0] < t0 + dur), key=lambda g: g[0])
    def union(events):
        """Busy time of the events, and the idle gaps between them."""
        busy, last, gaps = 0.0, None, []
        for ts, d, *_ in events:
            if last is not None and ts > last:
                gaps.append((last, ts))
            busy += d if last is None or ts >= last else max(0.0, ts + d - last)
            last = ts + d if last is None else max(last, ts + d)
        return busy, gaps, last

    busy, gaps, last = union(own)
    idle = {"<10 us": 0.0, "10-100 us": 0.0, "0.1-1 ms": 0.0, ">1 ms": 0.0}
    for a, b in gaps:
        g = b - a
        idle["<10 us" if g < 10 else "10-100 us" if g < 100 else "0.1-1 ms" if g < 1000 else ">1 ms"] += g
    span = last - own[0][0]
    no_nccl = union([g for g in own if not g[3]])[0]
    kernels_no_nccl = union([g for g in own if g[4] and not g[3]])[0]
    calls = sorted(launch[g[2]][1] for g in own)
    backlog = [g[0] - (launch[g[2]][0] + launch[g[2]][1]) for g in own]
    long_ = [(g[0] - sum(launch[g[2]]), launch[g[2]][1]) for g in own if launch[g[2]][1] > 100]
    long_backlog = sorted(b for b, _ in long_)
    print(f"{name}: span {span / 1e6:.3f} s, busy {busy / 1e6:.3f} s ({busy / span:.1%}), without NCCL kernels"
          f" {no_nccl / 1e6:.3f} s, non-NCCL kernels only {kernels_no_nccl / 1e6:.3f} s; idle by gap "
          + ", ".join(f"{k} {v / 1e6:.3f} s" for k, v in idle.items()))
    print(f"  launch calls: {len(calls)}, {sum(calls) / 1e6:.3f} s, median {calls[len(calls) // 2]:.1f} us; up to 20 us"
          f" {sum(c for c in calls if c <= 20) / 1e6:.3f} s, 20-100 us {sum(c for c in calls if 20 < c <= 100) / 1e6:.3f} s,"
          f" over 100 us {sum(c for c in calls if c > 100) / 1e6:.3f} s ({sum(c > 100 for c in calls)} calls), of which"
          f" {sum(d for b, d in long_ if b > 1000) / 1e6:.3f} s met over 1 ms of queued work")
    print(f"  launches meeting over 1 ms of queued GPU work: {sum(b > 1000 for b in backlog) / len(backlog):.0%};"
          f" calls over 100 us: {sum(b > 1000 for b in long_backlog) / max(1, len(long_backlog)):.0%} of them, median"
          f" backlog {long_backlog[len(long_backlog) // 2] / 1e3 if long_backlog else 0:.1f} ms")
    # the GPU event ending each gap of 10 us or more: was its launch call still running when the gap began?
    first = {}
    for g in own:
        first.setdefault(g[0], []).append(g)
    waited, total = 0.0, 0.0
    for a, b in gaps:
        if b - a >= 10:
            nxt = min(first[b], key=lambda g: sum(launch[g[2]]))
            total += b - a
            waited += (b - a) if sum(launch[nxt[2]]) > a else 0.0
    calls_iv = sorted((launch[g[2]][0], sum(launch[g[2]])) for g in own)
    starts = [c[0] for c in calls_iv]
    covered, g10 = 0.0, [(a, b) for a, b in gaps if 10 <= b - a < 100]
    for a, b in g10:
        i = bisect.bisect_left(starts, a - 1e4)
        while i < len(calls_iv) and calls_iv[i][0] < b:
            covered += max(0.0, min(calls_iv[i][1], b) - max(calls_iv[i][0], a))
            i += 1
    print(f"  gaps of 10 us or more ending at GPU work whose launch call returned inside the gap: {waited / total:.1%} of"
          f" their time; launch calls cover {covered / 1e6:.3f} s of the {sum(b - a for a, b in g10) / 1e6:.3f} s of"
          f" 10-100 us gaps")
    runs, cur = [], None
    for ts, d, b in sorted((launch[g[2]][0], launch[g[2]][1], g[0] - sum(launch[g[2]])) for g in own):
        if b > 1000:
            cur = [ts, ts + d] if cur is None else [cur[0], ts + d]
        elif cur is not None:
            runs.append(cur[1] - cur[0])
            cur = None
    if cur is not None:
        runs.append(cur[1] - cur[0])
    runs.sort()
    print(f"  GPU-bound stretches (consecutive calls meeting over 1 ms of queued work): {sum(runs) / dur:.1%} of the step's"
          f" CPU time, {sum(r > 1e4 for r in runs)} over 10 ms, longest {runs[-1] / 1e6 if runs else 0:.2f} s")
