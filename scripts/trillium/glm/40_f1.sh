#!/usr/bin/env bash
# F1: GLM-5 checkpoint save and load, sync vs async, at 13 or 52 nodes. One job runs save (sync), load, save (async),
# load: the checkpoint is deleted (timed) after each load, so only one exists at a time (job.sbatch).
#   ./40_f1.sh weights|full 13|52 [-- extra sbatch args]
# Escalation (DESIGN.md, with the admins' agreement): weights 13 -> weights 52 -> full 13 -> full 52. Expected sizes:
# weights ~1.5 TB, full ~10 TB (estimates). Repeat at different times of day with REPETITION=2, 3.
# env: TIME (default 02:00:00, an estimate), CKPT_ROOT (default $SCRATCH/trillium/checkpoints), REPETITION, and _common.sh's
set -euo pipefail
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/_common.sh"
content=${1:?usage: $0 weights|full 13|52 [-- sbatch args]}
nodes=${2:?usage: $0 weights|full 13|52 [-- sbatch args]}
shift 2
[ $# -gt 0 ] && [ "$1" = -- ] && shift
case $content in weights|full) ;; *) echo "content: weights or full" >&2; exit 2 ;; esac
case $nodes in 13|52) ;; *) echo "F1 runs on 13 or 52 nodes" >&2; exit 2 ;; esac
export TT_EXPERIMENT=F1

e=megatron-glm5-tri-ckpt-$content
submit "glm5-f1-$content-n$nodes-r$TT_REPETITION" "${TIME:-02:00:00}" "$nodes" \
    "$e-sync-save-n$nodes" "$e-load-n$nodes" "$e-async-save-n$nodes" "$e-load-n$nodes" -- "$@"
