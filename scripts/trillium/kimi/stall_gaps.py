"""Engine-wide stalls of a serving run, from the client's per-request token times (benchmark_results.db).

    python3 -I stall_gaps.py RUN_DIR [MIN_S]

For each pass: the intervals longer than MIN_S (1 s) in which requests were in flight and no request received a token,
their offset from the pass's first request, wall-clock start (Toronto, as the server logs) and requests in flight.
Token times: start + ttft, then + each inter-token latency (itl_json). The db stores start times on the client's
monotonic clock; the wall clock comes from the run's created_at (UTC), written when the pass's last request ended (it
matches the server's log: K3-INF-02's conc64 zero window, README), so a time t is created_at - (last end - t).
"""
import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

run_dir, min_s = sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
db = sqlite3.connect(f"file:{run_dir}/benchmark_results.db?mode=ro", uri=True)
for run_id, created, desc in db.execute("select run_id, created_at, description from runs order by run_id"):
    rows = db.execute("select start_time, ttft, itl_json, latency from requests where run_id = ? and success",
                      (run_id,)).fetchall()
    if not rows:
        continue
    t0 = min(r[0] for r in rows)
    tokens, spans = [], []
    for start, ttft, itl, latency in rows:
        t = start + ttft
        tokens.append(t)
        for d in json.loads(itl or "[]"):
            t += d
            tokens.append(t)
        spans.append((start, start + latency))
    tokens.sort()
    end = max(e for s, e in spans)
    wall0 = datetime.fromisoformat(created).replace(tzinfo=timezone.utc).astimezone(ZoneInfo("America/Toronto"))
    wall0 -= timedelta(seconds=end - t0)
    gaps = []
    for a, b in zip(tokens, tokens[1:]):
        if b - a > min_s:
            inflight = sum(1 for s, e in spans if s <= a and e > b)
            if inflight:
                gaps.append((a, b, inflight))
    first = tokens[0] - t0
    name = desc.split("/")[0].split(".", 1)[-1]
    line = f"{name:16s} n={len(rows):5d} dur={end - t0:7.1f}s first_token=+{first:.2f}s"
    print(line, "gaps>%gs:" % min_s, len(gaps))
    for a, b, n in gaps:
        w = wall0 + timedelta(seconds=a - t0)
        print(f"    {b - a:6.2f} s  +{a - t0:7.2f} to +{b - t0:7.2f}  {w:%H:%M:%S.%f}"[:-3] + f"  in flight {n}")
