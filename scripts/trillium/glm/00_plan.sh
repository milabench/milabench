#!/usr/bin/env bash
# Step 0, login node, no GPU: every GLM-5 layout of the campaign against Megatron's rules (placement.check_layout) and
# the rank groups it gives on 8-GPU nodes (placement.describe: how many nodes each tp / dp / pp / ep group spans).
#   ./00_plan.sh                    the table of the dev.yaml entries (layout, nodes, batch: copies, micro-batches)
#   ./00_plan.sh pp13 8 13 256      one layout in detail: PP, EP, nodes, global batch [micro-batch]
# Hostnames are made up (trig0101, ...): Trillium has no Slurm topology, every node is on switch "unknown".
set -euo pipefail
here=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")
megatron=$here/../../../benchmarks/megatron
PY=${PYTHON:-python3}

if [ $# -ge 4 ]; then
    pp=${1#pp} ep=$2 nodes=$3 gbs=$4 mbs=${5:-1}
    exec "$PY" "$megatron/placement.py" --hosts "trig[0101-$(printf '%04d' $((100 + nodes)))]" --topology-file /dev/null \
        --gpus-per-node 8 --pp "$pp" --ep "$ep" --gbs "$gbs" --mbs "$mbs"
fi

"$PY" - "$megatron" <<'PYEOF'
import sys, warnings
sys.path.insert(0, sys.argv[1])
import placement

# (layout, PP, EP, nodes, global batch, micro-batch): the A-G / B-G / C / F1 entries of benchmarks/megatron/dev.yaml
runs = [("pp13-ep8", 13, 8, n, g, m) for n, g, m in [(13, 256, 1), (26, 512, 1), (39, 768, 1), (52, 1024, 1),
                                                        (13, 768, 1), (26, 768, 1), (52, 768, 1), (13, 64, 1)]]
runs += [("pp13-ep16", 13, 16, 26, 512, 1), ("pp13-ep16", 13, 16, 52, 1024, 1), ("pp13-ep32", 13, 32, 52, 1024, 1),
         ("pp26-ep8", 26, 8, 26, 512, 1), ("pp26-ep8", 26, 8, 52, 1024, 1),
         ("pp6-ep16", 6, 16, 12, 256, 1), ("pp6-ep16", 6, 16, 48, 1024, 1), ("pp6-ep16", 6, 16, 36, 768, 1),
         ("pp10-ep8", 10, 8, 10, 200, 1), ("pp10-ep8", 10, 8, 40, 800, 1), ("pp10-ep8", 10, 8, 50, 1000, 1),
         ("pp25-ep8", 25, 8, 25, 512, 1), ("pp25-ep8", 25, 8, 50, 1024, 1)]
print(f"{'layout':<10} {'nodes':>5} {'gbs':>5} {'mbs':>3}  {'copies':>6} {'dp':>3} {'micro-batches':>13} {'bubble':>6}")
for layout, pp, ep, nodes, gbs, mbs in runs:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            r = placement.check_layout(nodes * 8, pp, ep, gbs, mbs)
        except ValueError as exc:
            print(f"{layout:<10} {nodes:>5} {gbs:>5} {mbs:>3}  INFEASIBLE: {exc}")
            continue
    note = "  (fewer micro-batches than stages)" if w else ""
    m = r["microbatches_per_pipeline"]
    bubble = (pp - 1) / (m + pp - 1)  # 1F1B, no virtual stages
    print(f"{layout:<10} {nodes:>5} {gbs:>5} {mbs:>3}  {r['copies']:>6} {r['dp']:>3} {m:>13} {bubble:>6.0%}{note}")
PYEOF
