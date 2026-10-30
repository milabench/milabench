#!/usr/bin/env bash
# Step 1, BEFORE the G runs: can GLM-5 train fully (nothing frozen, 744B) on the window's node sets, and at which rung of
# the fallback ladder (benchmarks/megatron/dev.yaml, "fallbacks, for the window")? 4 steps per attempt.
#
#   G1's set, 50 nodes, EP8 x PP25 (G1, G2, G3, G5 full):  megatron-glm5-check-fit-full-50n, then optimizer offload 0.25,
#                                                          0.5, 1.0 (-offload025/05/1-50n)
#   G4's set, 64 nodes, EP8 x PP16:                        megatron-glm5-check-fit-full-64n, then the same offloads
#
# Each set is one job (66-node allocation, the entry picks its own set) that stops at the first attempt that completes.
# The job log ends with one line per attempt (PASS / FAIL, duration, largest per-rank peak memory) and the rung to use,
# also written to $MILABENCH_BASE/slurm/glm5-full-rung.{50n,64n}: the full G runs (02-06) read their rung from there.
# Peak memory per rank and pipeline stage: `peak_memory_*` in the run data.
#
#   ./01_check_full.sh [g1|g4|both] [-- extra sbatch args]          default both (g4 waits for g1)
#
# env (see _common.sh for the rest: NODES, EXCLUDE, NODELIST, AFTER, PREPARE, NO_TAIL, TEST_ONLY):
#   TIME        per job (default 03:00:00: up to 4 attempts of model build + 4 steps)
#   BALANCED=1  also try balanced PP25 stages without offload (--pipeline-model-parallel-layout, not yet run with GLM-5),
#               right after the as-configured attempt: if it fits, the full runs need no CPU optimizer step
#   LADDER      the entries to try, space separated, instead of the default ladder
#
# Read in the log: an OOM ("CUDA out of memory") in the failed attempts, and for the offload rungs the host memory
# (each node keeps the optimizer state of its 4 ranks: up to ~200-400 GB at fraction 1.0, PP25). If the aligned set has
# 52 nodes (rg21701 healthy) the 50n entries fail at launch in place_nodes(): EXCLUDE one node of a 2-node leaf group.
# If no rung completes for g1, full training is G4 alone.
set -euo pipefail
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/_common.sh"

which=${1:-both}
[ $# -gt 0 ] && shift
[ $# -gt 0 ] && [ "$1" = -- ] && shift
TIME=${TIME:-03:00:00}
export STOP_AT_FIRST=1

ladder() {
    local n=$1 base=megatron-glm5-check-fit-full
    if [ -n "${LADDER:-}" ]; then echo "$LADDER"; return; fi
    local rungs=("$base-${n}n")
    [ "${BALANCED:-0}" = 1 ] && [ "$n" = 50 ] && rungs+=("$base-balanced-${n}n")
    rungs+=("$base-offload025-${n}n" "$base-offload05-${n}n" "$base-offload1-${n}n")
    echo "${rungs[*]}"
}

case $which in
    g1) RESULT_FILE=$(rung_file 50) submit glm5-check-full-g1 "$TIME" $(ladder 50) -- "$@" ;;
    g4) RESULT_FILE=$(rung_file 64) submit glm5-check-full-g4 "$TIME" $(ladder 64) -- "$@" ;;
    both)
        out=$(RESULT_FILE=$(rung_file 50) NO_TAIL=1 submit glm5-check-full-g1 "$TIME" $(ladder 50) -- "$@" | tee /dev/stderr)
        id=$(job_id "$out")
        [ -n "$id" ] || [ -n "${TEST_ONLY:-}" ] || { echo "no job id for g1, g4 not submitted" >&2; exit 1; }
        AFTER=${id:-${AFTER:-}} RESULT_FILE=$(rung_file 64) submit glm5-check-full-g4 "$TIME" $(ladder 64) -- "$@"
        ;;
    *) echo "usage: $0 [g1|g4|both] [-- sbatch args]" >&2; exit 2 ;;
esac
