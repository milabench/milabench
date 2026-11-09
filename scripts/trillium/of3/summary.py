"""Summary of one OF3 run directory (job.sbatch): from steps-r*.jsonl, the step period (time between successive step
ends, rank 0, the throughput step time), the step itself and the gap before it (waiting for the batch), samples/s per GPU
(batch 1), peak memory over ranks, loss, crop sizes; from nodemon/nodes/*.tsv over the same window, GPU power and SM
clock, CPU clock of the busy cores and CPU use. Steps after the first SKIP (default 5) only.
Usage: python summary.py RUN [SKIP]"""
import csv
import glob
import json
import math
import statistics as st
import sys
from pathlib import Path

run = Path(sys.argv[1])
skip = int(sys.argv[2]) if len(sys.argv) > 2 else 5
ranks = {}
for f in glob.glob(str(run / "steps-r*.jsonl")):
    rows = [json.loads(line) for line in open(f)]
    ranks[int(f.rsplit("-r", 1)[1].split(".")[0])] = [r for r in rows if "step" in r]
r0 = ranks[0]
period = [b["t"] - a["t"] for a, b in zip(r0, r0[1:])][skip:]
kept = {k: v[skip + 1:] for k, v in ranks.items()}
med = st.median(period)
print(f"{run.name}: {len(ranks)} ranks, {len(r0)} steps, statistics over steps {skip + 2}-{len(r0)}")
print(f"  step period median {med:.3f} s (p10 {st.quantiles(period, n=10)[0]:.3f}, p90 {st.quantiles(period, n=10)[-1]:.3f}),"
      f" {1 / med:.3f} samples/s per GPU, {len(ranks) / med:.2f} samples/s in all")
gap_share = {k: sum(r["gap_s"] for r in v) / sum(r["gap_s"] + r["step_s"] for r in v) for k, v in kept.items()}
print(f"  rank 0: step {st.median(r['step_s'] for r in kept[0]):.3f} s, gap {st.median(r['gap_s'] for r in kept[0]):.3f} s median;"
      f" gap share of time, mean over ranks {st.mean(gap_share.values()):.1%}, max {max(gap_share.values()):.1%} (rank {max(gap_share, key=gap_share.get)})")
print(f"  peak allocated {max(r['alloc_peak_gib'] for v in ranks.values() for r in v):.1f} GiB,"
      f" reserved {max(r['reserved_peak_gib'] for v in ranks.values() for r in v):.1f} GiB (max over ranks)")
losses = [[r["loss"] for r in v] for v in ranks.values()]
n = min(map(len, losses))
mean_loss = [st.mean(l[i] for l in losses) for i in range(n)]
bad = sum(1 for l in losses for x in l if x is None or not math.isfinite(x))
w = max(1, min(10, n // 4))
print(f"  loss, mean over ranks: steps 1-{w} {st.mean(mean_loss[:w]):.3f}, last {w} {st.mean(mean_loss[-w:]):.3f}; non-finite {bad}")
toks = [r["tokens"] for v in ranks.values() for r in v]
atoms = [r["atoms"] for v in ranks.values() for r in v]
msa = [r["msa_shape"][-2] for v in ranks.values() for r in v]
print(f"  tokens per sample median {st.median(toks)} (min {min(toks)}), atoms median {st.median(atoms)} (max {max(atoms)}),"
      f" MSA rows median {st.median(msa)} (max {max(msa)})")
t0, t1 = r0[skip]["t"] * 1000, r0[-1]["t"] * 1000


def window(kind):
    rows = []
    for f in glob.glob(str(run / f"nodemon/nodes/*.{kind}.tsv")):
        for row in csv.reader(open(f), delimiter="\t"):
            if row and row[0].isdigit() and t0 <= int(row[0]) <= t1:
                rows.append(row)
        header = next(csv.reader(open(f), delimiter="\t"))
    return header, rows


def col(header, rows, name):
    i = header.index(name)
    return [float(r[i]) for r in rows if r[i] not in ("", "None")]


h, g = window("gpu")
if g:
    power, clock = col(h, g, "power_w"), col(h, g, "sm_clock_mhz")
    print(f"  GPU: power mean {st.mean(power):.0f} W, max {max(power):.0f} W; SM clock median {st.median(clock):.0f} MHz;"
          f" util_mean {st.mean(col(h, g, 'util_mean')):.0%} ({len(g)} samples)")
h, c = window("cpu")
if c:
    print(f"  CPU: busy-core clock median {st.median(col(h, c, 'cpu_mhz_busy_mean')):.0f} MHz (min {min(col(h, c, 'cpu_mhz_busy_mean')):.0f}),"
          f" busy cores per package median {st.median(col(h, c, 'busy_cores')):.0f}, use {st.mean(col(h, c, 'cpu_util')):.0%}")
