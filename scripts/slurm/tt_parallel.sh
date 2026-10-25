#!/bin/bash
# Run several benches at the same time on disjoint node lists of one allocation
# (window pairings, e.g. a spine-crossing run next to a one-switch run).
#
#   bash scripts/slurm/tt_parallel.sh [NAME=value ...] NODELIST:BENCH:TAG ...
#
#   e.g. tt_parallel.sh TT_CONFIG=window "rg[21701-21707,21801-21807]:tt-w-n14:par-P14" \
#                                        "rg[31601-31607]:tt-w-leaf7:par-S7"
#
# Each run is tt_native.sh on its node list (TT_NODES), started from the list's
# first node; TT_RUN_TAG records TAG. Waits for all runs; exit status 1 if any
# failed. Leading NAME=value arguments are exported (as in tt_native.sh).

set -uo pipefail

while [[ "${1:-}" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; do
    export "$1"
    shift
done
[ $# -gt 0 ] || { echo "usage: tt_parallel.sh [NAME=value ...] NODELIST:BENCH:TAG ..."; exit 2; }

TT38=${TT38:-$SCRATCH/tt38}
SCRIPTS=$TT38/milabench/scripts/slurm

pids=()
specs=()
for spec in "$@"; do
    IFS=: read -r nodelist bench tag <<<"$spec"
    nodes=$(scontrol show hostnames "$nodelist" | paste -sd,)
    first=${nodes%%,*}
    echo "[tt-parallel] start tag=$tag bench=$bench nodes=$nodes t=$(date +%s)"
    srun --overlap --nodes=1 --ntasks=1 -w "$first" \
        bash "$SCRIPTS/tt_native.sh" TT_NODES="$nodes" TT_RUN_TAG="$tag" "$bench" </dev/null \
        > >(sed -u "s/^/[$tag] /") 2>&1 &
    pids+=($!)
    specs+=("$tag")
done

rc=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "[tt-parallel] done tag=${specs[$i]} t=$(date +%s)"
    else
        echo "[tt-parallel] FAILED tag=${specs[$i]} t=$(date +%s)"
        rc=1
    fi
done
exit $rc
