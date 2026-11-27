#!/bin/bash
# Interconnect topology test: the same 4-node run on different node layouts
# inside one allocation, one layout at a time (no concurrent traffic from us).
#
#   sbatch --nodes=8 --time=01:00:00 --account=... scripts/slurm/tt_native_topology.sh
#
# Layouts, built from the allocation and `scontrol show topology` (leaf
# switches = Level=0): packed (4 nodes on one leaf), split2x2 (2+2 on two
# leaves), spread4 (1 node on each of 4 leaves). Layouts the allocation cannot
# provide are skipped. The packed layout runs again at the end to check noise.
#
#SBATCH --job-name=tt-topo
#SBATCH --gpus-per-node=h100:4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --output=%x-%j.out

set -euo pipefail

TT38=${TT38:-$SCRATCH/tt38}
BENCH=${1:-tt-qwen38-n4-topo}
unset PYTHONPATH PIP_CONFIG_FILE
export PATH=$TT38/env/bin:$HOME/.local/bin:$PATH

LAYOUTS=$(python - <<'EOF'
import subprocess

def hostnames(spec):
    out = subprocess.run(["scontrol", "show", "hostnames", spec], capture_output=True, text=True)
    return out.stdout.split()

import os
allocated = hostnames(os.environ["SLURM_JOB_NODELIST"])
leaf_of = {}
topo = subprocess.run(["scontrol", "show", "topology"], capture_output=True, text=True).stdout
for line in topo.splitlines():
    fields = dict(f.split("=", 1) for f in line.split() if "=" in f)
    if fields.get("Level") == "0" and "Nodes" in fields:
        for node in hostnames(fields["Nodes"]):
            leaf_of[node] = fields["SwitchName"]

by_leaf = {}
for node in allocated:
    by_leaf.setdefault(leaf_of.get(node, "unknown"), []).append(node)
leaves = sorted(by_leaf.items(), key=lambda kv: -len(kv[1]))
for leaf, nodes in leaves:
    print(f"# leaf {leaf}: {','.join(nodes)}")

layouts = []
if leaves and len(leaves[0][1]) >= 4:
    layouts.append(("packed", leaves[0][1][:4]))
two = [nodes for _, nodes in leaves if len(nodes) >= 2]
if len(two) >= 2:
    layouts.append(("split2x2", two[0][:2] + two[1][:2]))
if len(leaves) >= 4:
    layouts.append(("spread4", [nodes[0] for _, nodes in leaves[:4]]))
if layouts and layouts[0][0] == "packed":
    layouts.append(("packed-repeat", layouts[0][1]))
for name, nodes in layouts:
    print(name, ",".join(nodes))
EOF
)
echo "$LAYOUTS" | sed 's/^/[tt-topo] /'

echo "$LAYOUTS" | grep -v '^#' | while read -r NAME NODES; do
    [ -n "$NAME" ] || continue
    FIRST=${NODES%%,*}
    echo "[tt-topo] layout=$NAME nodes=$NODES start=$(date +%s)"
    srun --overlap --nodes=1 --ntasks=1 -w "$FIRST" \
        env TT_NODES="$NODES" TT_RUN_TAG="topo-$NAME" TT_COMPILE_CACHE= \
        bash "$TT38/milabench/scripts/slurm/tt_native.sh" "$BENCH" \
        < /dev/null || echo "[tt-topo] layout=$NAME failed rc=$?"
    echo "[tt-topo] layout=$NAME end=$(date +%s)"
done
