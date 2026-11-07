#!/bin/bash
# Slurm job of one Kimi K3 run (submitted by k3_train.sh): the B200 power check on every node, then the shared
# torchtitan_native launcher (scripts/slurm/tt_native.sh), which runs it only for TT_CONFIG=trillium (the Qwen file).
#
#   k3_job.sh TT38=<base> NAME=value ... <bench>    (arguments passed on to tt_native.sh)
#
# Slurm runs a copy of this script from its spool directory: the repository is found through TT38, as in tt_native.sh.
set -euo pipefail
for arg in "$@"; do
    [[ "$arg" == TT38=* ]] && export "$arg"
done
REPO=${TT38:?TT38=<base> argument missing}/milabench
srun --overlap --nodes="$SLURM_JOB_NUM_NODES" --ntasks="$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 bash "$REPO/scripts/trillium/power_check.sh"
echo "[k3-train] power_check passed on $SLURM_JOB_NUM_NODES nodes"
exec bash "$REPO/scripts/slurm/tt_native.sh" "$@"
