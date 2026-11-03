#!/bin/bash
# Submit one Qwen3.8-27B TorchTitan run on Trillium's B200 partition
# (config/torchtitan_native_trillium.yaml, scripts/trillium/DESIGN.md).
#
#   scripts/trillium/qwen/tt_trillium.sh [--nodes N] [--time HH:MM:SS] [--dry-run] \
#       [NAME=value ...] <bench> [milabench run args]
#
#   scripts/trillium/qwen/tt_trillium.sh --nodes 13 --time 00:30:00 tri-aq-hsdp8-24k
#   scripts/trillium/qwen/tt_trillium.sh --dry-run TT_CAMPAIGN=window-1 tri-smoke
#
# The job runs scripts/slurm/tt_native.sh with TT_CONFIG=trillium. Leading
# NAME=value arguments are exported for the job (TT_CAMPAIGN, TT_VARIANT,
# TT_REPETITION, TT_EXPERIMENT, TT_NODE_SET, TT_CONDITION, BENCHMATE_WANDB,
# NCCL_DEBUG, ...). TT_EXPERIMENT (run.experiment, scripts/trillium/FORMAT.md)
# defaults from the entry name (tri-aq-* -> A-Q; B-Q-weak runs pass it).
# Paths ($TT_TRI_BASE = $SCRATCH/trillium/qwen, the TT38 of tt_native.sh):
#   $TT_TRI_BASE/{env,milabench,output}   venv, this repository, MILABENCH_BASE (README.md)
#   $TT_TRI_BASE/logs/<bench>-<id>.out    Slurm output
#   $SCRATCH/trillium/runs/<run>          milabench run data
#   $SCRATCH/trillium/nodemon/<run>       node telemetry
#   $SCRATCH/trillium/checkpoints/<run>   tri-ckpt-* checkpoints (deleted after the load)
#
# Environment: TRI_ACCOUNT (rrg-bengioy-ad), TRI_PARTITION
# (compute_b200_full_node), TRI_RESERVATION (sbatch --reservation, if any).

set -euo pipefail

NODES=1
TIME=00:30:00
DRY_RUN=0
while [ $# -gt 0 ]; do
    case "$1" in
        --nodes) NODES=$2; shift 2 ;;
        --time) TIME=$2; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        *) break ;;
    esac
done

ENVS=()
while [[ "${1:-}" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; do
    ENVS+=("$1")
    shift
done
BENCH=${1:?usage: tt_trillium.sh [--nodes N] [--time T] [--dry-run] [NAME=value ...] <bench> [milabench args]}
shift

BASE=${TT_TRI_BASE:-${SCRATCH:?}/trillium/qwen}
REPO=$BASE/milabench
CONFIG=$REPO/config/torchtitan_native_trillium.yaml
[ -f "$CONFIG" ] || CONFIG=$(cd "$(dirname "$0")/../../.." && pwd)/config/torchtitan_native_trillium.yaml
if ! grep -q "^$BENCH:" "$CONFIG"; then
    echo "tt_trillium.sh: no entry '$BENCH' in $CONFIG" >&2
    exit 2
fi

case "$BENCH" in
    tri-aq-*) EXPERIMENT=A-Q ;;
    tri-bqs-*) EXPERIMENT=B-Q-strong ;;
    tri-closs-*) EXPERIMENT=C-loss ;;
    tri-c-*) EXPERIMENT=C-speed ;;
    tri-ckpt-*) EXPERIMENT=F1 ;;
    tri-ref) EXPERIMENT=R ;;
    tri-smoke*) EXPERIMENT=smoke ;;
    *) EXPERIMENT= ;;
esac

TRI=$SCRATCH/trillium
# Run settings for tt_native.sh; the leading NAME=value arguments come after
# them, so they win.
JOB_ENV=(
    TT38="$BASE"
    TT_CONFIG=trillium
    TT_MODULES=cuda/12.9
    CC_CLUSTER=trillium
    TT_EXPERIMENT="$EXPERIMENT"
    MILABENCH_DIRS_RUNS="$TRI/runs"
    TT_NODEMON_ROOT="$TRI/nodemon"
    TT_CKPT_ROOT="$TRI/checkpoints"
    # No HTTP proxy on Trillium compute nodes: wandb offline, `wandb sync` from a login node.
    WANDB_MODE=offline
    "${ENVS[@]}"
)

CMD=(
    sbatch
    --account="${TRI_ACCOUNT:-rrg-bengioy-ad}"
    --partition="${TRI_PARTITION:-compute_b200_full_node}"
    --gpus-per-node=b200:8
    --nodes="$NODES"
    --time="$TIME"
    --job-name="$BENCH"
    --output="$BASE/logs/%x-%j.out"
    ${TRI_RESERVATION:+--reservation="$TRI_RESERVATION"}
    "$REPO/scripts/slurm/tt_native.sh"
    "${JOB_ENV[@]}"
    "$BENCH"
    "$@"
)

if [ "$DRY_RUN" = 1 ]; then
    printf '%q ' "${CMD[@]}"
    echo
    exit 0
fi
mkdir -p "$BASE/logs" "$TRI/runs" "$TRI/nodemon" "$TRI/checkpoints"
"${CMD[@]}"
