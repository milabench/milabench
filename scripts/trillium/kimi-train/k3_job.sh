#!/bin/bash
# Slurm job of one Kimi K3 run (submitted by k3_train.sh): the B200 power check on every node, then the shared
# torchtitan_native launcher (scripts/slurm/tt_native.sh), which runs it only for TT_CONFIG=trillium (the Qwen file).
#
#   k3_job.sh NAME=value ... <bench>    (arguments passed on to tt_native.sh)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
srun --overlap --nodes="$SLURM_JOB_NUM_NODES" --ntasks="$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 bash "$HERE/../power_check.sh"
echo "[k3-train] power_check passed on $SLURM_JOB_NUM_NODES nodes"
exec bash "$HERE/../../slurm/tt_native.sh" "$@"
