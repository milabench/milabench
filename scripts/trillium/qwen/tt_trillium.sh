#!/bin/bash
# Submit one Qwen3.8-27B TorchTitan run on Trillium's B200 partition
# (config/torchtitan_native_trillium.yaml, scripts/trillium/DESIGN.md).
#
#   scripts/trillium/qwen/tt_trillium.sh [--nodes N] [--time HH:MM:SS] [--dry-run] \
#       [NAME=value ...] <bench> [milabench run args]
#
#   scripts/trillium/qwen/tt_trillium.sh --nodes 13 --time 00:30:00 tri-aq-hsdp8-24k
#   scripts/trillium/qwen/tt_trillium.sh --dry-run TT_CAMPAIGN=window-1 tri-smoke
#   scripts/trillium/qwen/tt_trillium.sh --nodes 13 TT_REPETITION=2 tri-closs-bf16    # C-loss, same seed again
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
# Nodes and time default from the entry (entry_info.py, milabench's config
# resolution): an entry with a fixed num_machines runs on that many nodes (a
# different --nodes is refused); an entry using every allocated node
# (num_machines: 999) needs --nodes. Time = max_duration + 15 min (staging,
# milabench startup), capped at 24 h.
#
# Environment: TRI_ACCOUNT (rrg-bengioy-ad), TRI_PARTITION
# (compute_b200_full_node), TRI_RESERVATION (b200-friendly-users), TRI_QOS (bump), TRI_EXCLUDE (below),
# TT_PYTHON (Python with milabench: default $TT_TRI_BASE/env/bin/python).

set -euo pipefail

NODES=
TIME=
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
PY=${TT_PYTHON:-$BASE/env/bin/python}
[ -x "$PY" ] || PY=python3
INFO=$(PYTHONPATH="$(dirname "$CONFIG")/..${PYTHONPATH:+:$PYTHONPATH}" "$PY" "$(dirname "$0")/entry_info.py" "$CONFIG" "$BENCH") || {
    echo "tt_trillium.sh: cannot resolve entry '$BENCH' in $CONFIG" >&2
    exit 2
}
eval "$INFO"   # NUM_MACHINES MAX_DURATION LAYOUT

if [ "$NUM_MACHINES" -gt 0 ]; then
    if [ -n "$NODES" ] && [ "$NODES" != "$NUM_MACHINES" ]; then
        echo "tt_trillium.sh: $BENCH runs on $NUM_MACHINES nodes (num_machines), not --nodes $NODES" >&2
        exit 2
    fi
    NODES=$NUM_MACHINES
elif [ -z "$NODES" ]; then
    echo "tt_trillium.sh: $BENCH runs on every allocated node: pass --nodes" >&2
    exit 2
fi
# HSDP shard groups must divide the GPU count (8 per node).
if [[ "$LAYOUT" =~ ^hsdp([0-9]+)$ ]] && (( NODES * 8 % BASH_REMATCH[1] )); then
    echo "tt_trillium.sh: layout $LAYOUT needs a multiple of ${BASH_REMATCH[1]} GPUs, not $((NODES * 8))" >&2
    exit 2
fi
if [ -z "$TIME" ]; then
    seconds=$((MAX_DURATION + 900))
    (( seconds > 86400 )) && seconds=86400
    TIME=$(printf '%02d:%02d:00' $((seconds / 3600)) $((seconds % 3600 / 60)))
fi

DATA_DIR=
case "$BENCH" in
    tri-aq-*) EXPERIMENT=A-Q ;;
    tri-bqs-*) EXPERIMENT=B-Q-strong ;;
    tri-closs-*) EXPERIMENT=C-loss; DATA_DIR=$SCRATCH/trillium/data ;;  # 2 M documents (README.md); the others stay on 500,000
    tri-c-*) EXPERIMENT=C-speed ;;
    tri-ckpt-*) EXPERIMENT=F1 ;;
    tri-ref) EXPERIMENT=R ;;
    tri-smoke*) EXPERIMENT=smoke ;;
    *) EXPERIMENT= ;;  # not passed: run.experiment stays null
esac

TRI=$SCRATCH/trillium
# Run settings for tt_native.sh; the leading NAME=value arguments come after
# them, so they win.
JOB_ENV=(
    TT38="$BASE"
    TT_CONFIG=trillium
    TT_MODULES="StdEnv/2023 gcc/12.3 cuda/12.9"  # cuda needs them loaded first
    CC_CLUSTER=trillium
    ${EXPERIMENT:+TT_EXPERIMENT="$EXPERIMENT"}
    MILABENCH_DIRS_RUNS="$TRI/runs"
    ${DATA_DIR:+MILABENCH_DIRS_DATA="$DATA_DIR"}
    TT_NODEMON_ROOT="$TRI/nodemon"
    TT_CKPT_ROOT="$TRI/checkpoints"
    # /tmp holds 2 GB on the B200 nodes (TmpDisk=2048): compile cache, datasets
    # copy and TMPDIR in /dev/shm (RAM, 2 TB per node), removed at the end. Not
    # per job: compile cache entries store absolute paths.
    TT_LOCAL_ROOT="/dev/shm/$USER"
    TT_LOCAL_CLEANUP=1
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
    --reservation="${TRI_RESERVATION-b200-friendly-users}"
    --qos="${TRI_QOS-bump}"
    --exclude="${TRI_EXCLUDE-trig[0122-0123,0133-0135,0138,0140,0143,0145,0148,0152]}"  # GPUs capped at 200 W on 2026-11-05 (power_check.sh); trig0122 also has old mlx5_1 firmware, 14% slower pairs; trig0134,0135,0148: lowest CPU clocks, 13-node Qwen ~8.5% slower with one (E1)
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
