"""Every node's metrics second by second over a time window, from benchmate.nodemon's files, to line up with a stall.

    python3 -I stall_nodes.py NODEMON_DIR FROM TO        (FROM, TO: 2026-11-09T18:40:00, Toronto time as the logs)

One line per second, one column group per node: GPU utilization (mean of the 8, %), GPU power (sum, kW), the lowest SM
clock (MHz) and the clock event reasons seen (OR of the 8, hex), GPM SM activity (mean, %), InfiniBand traffic (sum of
tx and rx over the HCAs, GB/s) and CPU utilization (mean of the packages, %). A blank is a sample the node did not write.
"""
import csv
import glob
import os
import sys
from collections import defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo

root, t_from, t_to = sys.argv[1:4]
tz = ZoneInfo("America/Toronto")
lo, hi = (int(datetime.fromisoformat(t).replace(tzinfo=tz).timestamp()) for t in (t_from, t_to))


def rows(path):
    with open(path) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            s = int(r["time_ms"]) // 1000
            if lo <= s <= hi:
                yield s, r


def num(v):
    return float(v) if v not in ("", None) else None


hosts = sorted(os.path.basename(p).split(".")[0] for p in glob.glob(f"{root}/nodes/*.gpu.tsv"))
data = defaultdict(dict)  # (second, host) -> metrics
for h in hosts:
    acc = defaultdict(lambda: defaultdict(list))
    for s, r in rows(f"{root}/nodes/{h}.gpu.tsv"):
        a = acc[s]
        a["util"].append(num(r["util_mean"]) if r["util_mean"] else num(r["util"]))
        a["power"].append(num(r["power_w"]))
        a["sm"].append(num(r["sm_clock_mhz"]))
        a["reasons"].append(int(r["clocks_event_reasons"], 16) if r["clocks_event_reasons"] else 0)
    if os.path.exists(f"{root}/nodes/{h}.gpm.tsv"):
        for s, r in rows(f"{root}/nodes/{h}.gpm.tsv"):
            if r["sm_util"]:
                acc[s]["gpm"].append(num(r["sm_util"]))
    for s, r in rows(f"{root}/nodes/{h}.ib.tsv"):
        tx, rx = num(r["ib_tx_bytes_per_s"]), num(r["ib_rx_bytes_per_s"])
        if tx is not None and rx is not None:
            acc[s]["ib"].append(tx + rx)
    for s, r in rows(f"{root}/nodes/{h}.cpu.tsv"):
        if r["cpu_util"]:
            acc[s]["cpu"].append(num(r["cpu_util"]))
    for s, a in acc.items():
        m = {}
        if a["util"]:
            m["util"] = 100 * sum(x for x in a["util"] if x is not None) / len(a["util"])
            m["kw"] = sum(x for x in a["power"] if x is not None) / 1000
            m["sm"] = min(x for x in a["sm"] if x is not None)
            m["reasons"] = 0
            for x in a["reasons"]:
                m["reasons"] |= x
        if a["gpm"]:
            m["gpm"] = 100 * sum(a["gpm"]) / len(a["gpm"])
        if a["ib"]:
            m["ib"] = sum(a["ib"]) / 1e9
        if a["cpu"]:
            m["cpu"] = 100 * sum(a["cpu"]) / len(a["cpu"])
        data[(s, h)] = m


def fmt(m, k, f):
    return f.format(m[k]) if k in m else " " * len(f.format(0))


print("time     " + "".join(f"| {h:<44s}" for h in hosts))
print("         " + "| util  kW   sm  reasons  gpm   IB GB/s cpu  " * len(hosts))
for s in range(lo, hi + 1):
    line = datetime.fromtimestamp(s, tz).strftime("%H:%M:%S ")
    for h in hosts:
        m = data.get((s, h), {})
        line += ("| " + fmt(m, "util", "{:4.0f}") + " " + fmt(m, "kw", "{:4.1f}") + " " + fmt(m, "sm", "{:4.0f}")
                 + " " + fmt(m, "reasons", "{:#8x}") + " " + fmt(m, "gpm", "{:4.0f}") + " " + fmt(m, "ib", "{:7.2f}")
                 + " " + fmt(m, "cpu", "{:4.1f}") + " ")
    print(line)
