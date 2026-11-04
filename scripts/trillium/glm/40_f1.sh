#!/usr/bin/env bash
# F1: GLM-5 checkpoint save and load, sync vs async. One job runs, per content, save (sync), load, save (async), load: the
# checkpoint is deleted (timed) after each load, so only one exists at a time (job.sbatch), weights first: a full checkpoint
# that runs out of memory (see the README) then costs little.
#   ./40_f1.sh 13|52 [all|weights|full] [-- extra sbatch args]     default all
#   ./40_f1.sh 26 [-- extra sbatch args]                           fallback: full, pp13-ep16, if full does not fit at 13 nodes
# Escalation (DESIGN.md, with the admins' agreement): weights 13 -> weights 52 -> full 13 -> full 52. Expected sizes: weights
# ~1.5 TB, full ~10.4 TB (estimates). Repeat at different times of day with REPETITION=2, 3.
# Limits (estimates): 5.8 GB/s worst case, so ~30 min per full save or load and ~5 min per weights one, + ~15 min of startup per entry.
# env: TIME (default all 05:30:00, full 04:00:00, weights 02:00:00), CKPT_ROOT (default $SCRATCH/trillium/checkpoints),
# REPETITION, and _common.sh's
set -euo pipefail
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/_common.sh"
nodes=${1:?usage: $0 13|52 [all|weights|full] | 26 [-- sbatch args]}
shift
content=all
if [ $# -gt 0 ] && [ "$1" != -- ]; then content=$1; shift; fi
[ $# -gt 0 ] && [ "$1" = -- ] && shift
export TT_EXPERIMENT=F1 JOB_NAME=glm5-f1 SINGLETON=1   # singleton: one F1 job at a time
ckpt_root_empty

e=megatron-glm5-tri-ckpt
cycle() {  # cycle CONTENT SUFFIX: the 4 entries of a content
    echo "$e-$1-sync-save$2" "$e-$1-load$2" "$e-$1-async-save$2" "$e-$1-load$2"
}
case $nodes:$content in
    13:all|52:all)  read -ra entries <<< "$(cycle weights "-n$nodes") $(cycle full "-n$nodes")"; time=05:30:00 ;;
    13:weights|52:weights) read -ra entries <<< "$(cycle weights "-n$nodes")"; time=02:00:00 ;;
    13:full|52:full) read -ra entries <<< "$(cycle full "-n$nodes")"; time=04:00:00 ;;
    26:all|26:full) read -ra entries <<< "$(cycle full-ep16 -n26)"; time=04:00:00; content=full-ep16 ;;
    *) echo "F1: 13 or 52 nodes with all, weights or full; 26 nodes for the full pp13-ep16 fallback" >&2; exit 2 ;;
esac
submit "glm5-f1-$content-n$nodes-r$TT_REPETITION" "${TIME:-$time}" "$nodes" "${entries[@]}" -- "$@"
