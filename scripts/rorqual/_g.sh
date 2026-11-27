#!/usr/bin/env bash
# Shared by 02-06: one GLM-5 placement block, one job on the window's allocation, ONE benchmark:
#   full training if it fits on the run's node set, frozen experts only if it does not.
#   _g.sh G1..G5 [pass1|pass2] [-- extra sbatch args]
#
# What fits comes from 01_check_full.sh, per node set (50 nodes: G1, G2, G3, G5; 64 nodes: G4), or from
# FULL (''|offload025|offload05|offload1|balanced|none). none = full training does not fit there: frozen experts.
# A rung other than '' changes the run: add it to the summary's `deviations` (optimizer on the CPU, balanced layout).
#
# env (see _common.sh for the rest: NODES, EXCLUDE, NODELIST, AFTER, PREPARE, NO_TAIL, TEST_ONLY):
#   TIME   default 01:00:00 (12 steps, max_duration 3600 s with startup)
set -euo pipefail
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/_common.sh"

run=${1:?usage: _g.sh G1..G5 [pass1|pass2] [-- sbatch args]}
shift
pass=pass1
if [ $# -gt 0 ] && [ "$1" != -- ]; then pass=$1; shift; fi
[ $# -gt 0 ] && [ "$1" = -- ] && shift

case $run in
    G1) entry=megatron-glm5-g1-ep8-aligned   set=50 family=EP ;;
    G2) entry=megatron-glm5-g2-ep8-crossed   set=50 family=EP ;;
    G3) entry=megatron-glm5-g3-ep8-mixed     set=50 family=EP ;;
    G4) entry=megatron-glm5-g4-ep8-all-nodes set=64 family=ALL ;;
    G5) entry=megatron-glm5-g5-fp8           set=50 family=EP ;;
    *) echo "unknown run $run (G1-G5)" >&2; exit 2 ;;
esac
# run_start record (FORMAT.md): run.variant, run.repetition (pass1 -> 1, pass2 -> 2), placement.family
export VARIANT=$run TT_FAMILY=${TT_FAMILY:-$family}
case $pass in pass[0-9]*) export REPETITION=${pass#pass} ;; *) export REPETITION=${REPETITION:-1} ;; esac

rung=$(full_rung "$set")
if [ "$rung" = none ]; then
    echo "[$run] full training does not fit on the ${set}-node set: frozen experts instead" >&2
else
    [ "$set" = 64 ] && [ "$rung" = -balanced ] && { echo "no balanced layout for PP16 (G4)" >&2; exit 2; }
    entry=$entry-full$rung
fi

submit "glm5-$(echo "$run" | tr A-Z a-z)-$pass" "${TIME:-01:00:00}" "$entry" -- "$@"
