"""Streaming read of one profiler trace (no full JSON load: a 1.4 GB trace in about 1 GB of memory), for
ProfilerStep#27 and #28: their absolute start and end against the logged step ends of STEPS (rank 0's
steps-r0.jsonl); the GPU work launched in each (kernels, copies, sets, matched to their launch call by correlation,
runtime or driver API): span, busy time, idle time by gap length (as prof.py); the launch calls' CPU time by call
length, and the GPU backlog each launch met (start of its GPU work minus the end of the call), overall and for calls
over 100 us.
Usage: python trace_gaps.py TRACE STEPS"""
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
            gpu.append((e["ts"], e["dur"], e.get("args", {}).get("correlation")))
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
    busy, idle, last = 0.0, {"<10 us": 0.0, "10-100 us": 0.0, "0.1-1 ms": 0.0, ">1 ms": 0.0}, None
    for ts, d, _ in own:
        if last is not None and ts > last:
            gap = ts - last
            idle["<10 us" if gap < 10 else "10-100 us" if gap < 100 else "0.1-1 ms" if gap < 1000 else ">1 ms"] += gap
        busy += d if last is None or ts >= last else max(0.0, ts + d - last)
        last = ts + d if last is None else max(last, ts + d)
    span = last - own[0][0]
    calls = sorted(launch[g[2]][1] for g in own)
    backlog = [g[0] - (launch[g[2]][0] + launch[g[2]][1]) for g in own]
    long_backlog = sorted(g[0] - sum(launch[g[2]]) for g in own if launch[g[2]][1] > 100)
    print(f"{name}: span {span / 1e6:.3f} s, busy {busy / 1e6:.3f} s ({busy / span:.1%}), idle by gap "
          + ", ".join(f"{k} {v / 1e6:.3f} s" for k, v in idle.items()))
    print(f"  launch calls: {len(calls)}, {sum(calls) / 1e6:.3f} s, median {calls[len(calls) // 2]:.1f} us; up to 20 us"
          f" {sum(c for c in calls if c <= 20) / 1e6:.3f} s, 20-100 us {sum(c for c in calls if 20 < c <= 100) / 1e6:.3f} s,"
          f" over 100 us {sum(c for c in calls if c > 100) / 1e6:.3f} s ({sum(c > 100 for c in calls)} calls)")
    print(f"  launches meeting over 1 ms of queued GPU work: {sum(b > 1000 for b in backlog) / len(backlog):.0%};"
          f" calls over 100 us: {sum(b > 1000 for b in long_backlog) / max(1, len(long_backlog)):.0%} of them, median"
          f" backlog {long_backlog[len(long_backlog) // 2] / 1e3 if long_backlog else 0:.1f} ms")
