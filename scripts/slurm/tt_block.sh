#!/bin/bash
# One window block: pick each run's nodes from the healthy set, then launch the
# runs at the same time (tt_parallel.sh) inside the current allocation.
#
#   bash scripts/slurm/tt_block.sh [NAME=value ...] FAMILY:NODES[:LEAF][:BENCH] ...
#
#   e.g. block 4:  tt_block.sh TT_CAMPAIGN=window-1030 H:27 S:12:ihpc316u25
#        block 2:  tt_block.sh ALL
#        block 1:  tt_block.sh O:1::tt-w-smoke          (next to GLM's 52 nodes)
#
# FAMILY is H / S / P / O (tt_pick_nodes.py) or ALL (every healthy node not used
# by the block's earlier runs; NODES is ignored). Runs of one block never share
# nodes: each spec picks among the nodes the previous specs left. BENCH defaults
# to tt-w-nodes (any node count); the variant is FAMILY + node count.
#
# Healthy nodes: TT_HEALTHY (hostlist) if set, else the latest healthy.txt of
# this job (tt_healthcheck.sh), else every node of the allocation. TT_EXCLUDE
# (hostlist) removes more nodes. Prints the plan, then runs it; TT_DRY_RUN=1
# prints only.

set -uo pipefail

while [[ "${1:-}" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; do
    export "$1"
    shift
done
[ $# -gt 0 ] || { echo "usage: tt_block.sh [NAME=value ...] FAMILY:NODES[:LEAF][:BENCH] ..."; exit 2; }

TT38=${TT38:-$SCRATCH/tt38}
unset PYTHONPATH PIP_CONFIG_FILE
PY=$TT38/env/bin/python
SCRIPTS=$TT38/milabench/scripts/slurm
export TT_CONFIG=${TT_CONFIG:-window}  # tt-w-nodes is a window entry

if [ -n "${TT_HEALTHY:-}" ]; then
    healthy=$TT_HEALTHY
elif [ -f "$TT38/health/${SLURM_JOB_ID:-none}/healthy.txt" ]; then
    healthy=$(paste -sd, "$TT38/health/$SLURM_JOB_ID/healthy.txt")
    export TT_HEALTH_DIR=${TT_HEALTH_DIR:-$TT38/health/$SLURM_JOB_ID}
else
    healthy=${SLURM_JOB_NODELIST:?no allocation: set TT_HEALTHY}
fi
used=$(scontrol show hostnames "${TT_EXCLUDE:-}" 2>/dev/null | paste -sd,)

runs=()
for spec in "$@"; do
    IFS=: read -r family nodes leaf bench <<<"$spec"
    bench=${bench:-tt-w-nodes}
    if [ "$family" = ALL ]; then
        nodelist=$(scontrol show hostnames "$healthy" | grep -vxF -f <(tr , '\n' <<<"$used") | paste -sd,)
    else
        nodelist=$("$PY" "$SCRIPTS/tt_pick_nodes.py" --family "$family" --nodes "$nodes" \
            --candidates "$healthy" --exclude "$used" ${leaf:+--leaf "$leaf"} 2> >(sed 's/^/[tt-block] /' >&2))
    fi
    if [ -z "$nodelist" ]; then
        echo "[tt-block] spec=$spec SKIPPED (cannot be built from healthy, unused nodes)"
        continue
    fi
    n=$(tr , '\n' <<<"$nodelist" | wc -l)
    variant=$family$n
    [ "$bench" = tt-w-nodes ] || variant=$variant-${bench#tt-w-}
    echo "[tt-block] spec=$spec variant=$variant bench=$bench nodes=$nodelist"
    runs+=("$nodelist:$bench:$variant:$family")
    used=${used:+$used,}$nodelist
done
[ ${#runs[@]} -gt 0 ] || exit 1
[ "${TT_DRY_RUN:-0}" = 1 ] && exit 0
exec bash "$SCRIPTS/tt_parallel.sh" "${runs[@]}"
