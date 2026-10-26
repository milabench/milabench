#!/bin/bash
# Run topology variants inside one allocation (window protocol).
#
#   sbatch --nodes=N [-w list] ... scripts/slurm/tt_variants.sh \
#       [NAME=value ...] FAMILY:NODES:BENCH[:LEAF] ...
#
#   e.g. tt_variants.sh TT_CONFIG=window H:6:tt-w-n6 S:6:tt-w-n6:ihpc316u25 H:6:tt-w-n6
#
# 1. Health check on every allocated node (tt_healthcheck.sh); failed nodes are
#    excluded for the rest of the job.
# 2. For each variant, in order: pick the nodes with tt_pick_nodes.py among the
#    healthy nodes of this allocation (FAMILY H / S / P / O, NODES, optional LEAF),
#    then launch tt_native.sh on that subset (milabench runs on the subset's
#    first node; TT_NODES restricts the workers). The variant (FAMILY + NODES),
#    its repetition in the campaign, the family, the health check and a
#    topology snapshot are recorded in the run (tt_launch_lib.sh).
#    A variant that can't be built is skipped and reported.
# Leading NAME=value arguments are exported (as in tt_native.sh), e.g.
# TT_CAMPAIGN=window-1030 TT_EXCLUSIVE_CLUSTER=1.
#
#SBATCH --job-name=tt-variants
#SBATCH --gpus-per-node=h100:4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --output=%x-%j.out

set -uo pipefail

while [[ "${1:-}" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; do
    export "$1"
    shift
done
[ $# -gt 0 ] || { echo "usage: tt_variants.sh [NAME=value ...] FAMILY:NODES:BENCH[:LEAF] ..."; exit 2; }

TT38=${TT38:-$SCRATCH/tt38}
unset PYTHONPATH PIP_CONFIG_FILE
PY=$TT38/env/bin/python
SCRIPTS=$TT38/milabench/scripts/slurm
source "$SCRIPTS/tt_launch_lib.sh"

echo "[tt-variants] job=$SLURM_JOB_ID nodes=$SLURM_JOB_NODELIST variants=$*"
bash "$SCRIPTS/tt_healthcheck.sh"
HEALTHY=$(paste -sd, "$TT38/health/$SLURM_JOB_ID/healthy.txt")
echo "[tt-variants] healthy=$HEALTHY"
tt_snapshot_topology

for spec in "$@"; do
    IFS=: read -r family nodes bench leaf <<<"$spec"
    nodelist=$("$PY" "$SCRIPTS/tt_pick_nodes.py" --family "$family" --nodes "$nodes" \
        --candidates "$HEALTHY" ${leaf:+--leaf "$leaf"} 2> >(sed 's/^/[tt-variants] /' >&2))
    if [ -z "$nodelist" ]; then
        echo "[tt-variants] variant=$spec SKIPPED (cannot be built from healthy nodes)"
        continue
    fi
    variant=${family}${nodes}
    rep=$(tt_next_repetition "$variant" "$bench")
    tag="topo-${variant}-${rep}"
    first=${nodelist%%,*}
    echo "[tt-variants] variant=$spec tag=$tag nodes=$nodelist start=$(date +%s)"
    srun --overlap --nodes=1 --ntasks=1 -w "$first" \
        bash "$SCRIPTS/tt_native.sh" TT_NODES="$nodelist" TT_RUN_TAG="$tag" \
            TT_VARIANT="$variant" TT_REPETITION="$rep" TT_FAMILY="$family" "$bench" </dev/null \
        || echo "[tt-variants] variant=$spec tag=$tag FAILED rc=$?"
    echo "[tt-variants] variant=$spec tag=$tag end=$(date +%s)"
done
