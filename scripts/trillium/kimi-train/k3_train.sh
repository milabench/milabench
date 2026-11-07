#!/bin/bash
# Submit one Kimi K3 pre-training run (TorchTitan kimi_k3, config/torchtitan_native_kimi.yaml) on Trillium's B200
# partition, as a bring-up job named nw-k3trn-<bench> (README.md).
#
#   scripts/trillium/kimi-train/k3_train.sh --nodelist L [--time HH:MM:SS] [--dry-run] [NAME=value ...] <bench>
#
#   scripts/trillium/kimi-train/k3_train.sh --nodelist trig0145 k3-1n-l4
#   scripts/trillium/kimi-train/k3_train.sh --nodelist trig0145,trig0146 --time 00:40:00 k3-pp2-l4
#
# --nodelist is required: exactly the entry's num_machines nodes, so that no job lands on a node the campaign
# excludes (sinfo shows trig0125 / trig0132 "reserved", which Slurm still allocates). Time defaults to the entry's
# max_duration + 15 min (startup), as tt_trillium.sh. The job runs k3_job.sh (power check, then scripts/slurm/tt_native.sh); leading
# NAME=value arguments are exported for it. Paths ($K3 = $SCRATCH/trillium/new/k3-trn, the TT38 of tt_native.sh):
#   $K3/{env,milabench,output}   milabench venv, this repository, MILABENCH_BASE (bench venv, FineWeb, tokenizer)
#   $K3/logs/<job name>-<id>.out Slurm output;  $K3/runs/<run>  milabench run data (+ traces/ of profiled runs)
#   $K3/cache/attn_gym           attn-gym's CuTe kernel cache ($HOME is read-only on compute nodes, job 1260617)
set -euo pipefail

TIME=
NODELIST=
DRY_RUN=0
while [ $# -gt 0 ]; do
    case "$1" in
        --time) TIME=$2; shift 2 ;;
        --nodelist) NODELIST=$2; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        *) break ;;
    esac
done
ENVS=()
while [[ "${1:-}" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; do
    ENVS+=("$1")
    shift
done
BENCH=${1:?usage: k3_train.sh --nodelist L [--time T] [--dry-run] [NAME=value ...] <bench>}
[ -n "$NODELIST" ] || { echo "k3_train.sh: pass --nodelist (an explicit node list)" >&2; exit 2; }

K3=${K3_BASE:-${SCRATCH:?}/trillium/new/k3-trn}
REPO=$K3/milabench
CONFIG=$REPO/config/torchtitan_native_kimi.yaml
eval "$(PYTHONPATH=$REPO "$K3/env/bin/python" "$REPO/scripts/trillium/qwen/entry_info.py" "$CONFIG" "$BENCH")"  # NUM_MACHINES ...
[ -n "$TIME" ] || TIME=$(printf '%02d:%02d:00' $(((MAX_DURATION + 900) / 3600)) $(((MAX_DURATION + 900) % 3600 / 60)))

JOB_ENV=(
    TT38="$K3"
    TT_CONFIG="$CONFIG"
    TT_MODULES="StdEnv/2023 gcc/12.3 cuda/12.9"
    CC_CLUSTER=trillium
    MILABENCH_DIRS_RUNS="$K3/runs"
    TT_NODEMON_ROOT="$K3/nodemon"
    # /tmp holds 2 GB on the B200 nodes: compile cache, datasets copy and TMPDIR in /dev/shm, removed at the end
    TT_LOCAL_ROOT="/dev/shm/$USER/k3trn"
    TT_LOCAL_CLEANUP=1
    ATTN_GYM_CUTE_CACHE_DIR="$K3/cache/attn_gym"
    WANDB_MODE=offline
    "${ENVS[@]}"
)
CMD=(
    /opt/slurm/bin/.sbatch --export=NONE --get-user-env
    --account="${TRI_ACCOUNT:-rrg-bengioy-ad}"
    --partition="${TRI_PARTITION:-compute_b200_full_node}"
    --gpus-per-node=b200:8 --ntasks-per-node=1 --exclusive --mem=0
    --nodes="$NUM_MACHINES"
    --nodelist="$NODELIST"
    --time="$TIME"
    --job-name="nw-k3trn-$BENCH"
    --output="$K3/logs/%x-%j.out"
    --reservation="${TRI_RESERVATION-b200-friendly-users}"
    --qos="${TRI_QOS-bump}"
    "$REPO/scripts/trillium/kimi-train/k3_job.sh"
    "${JOB_ENV[@]}"
    "$BENCH"
)
if [ "$DRY_RUN" = 1 ]; then
    printf '%q ' "${CMD[@]}"
    echo
    exit 0
fi
mkdir -p "$K3/logs" "$K3/runs" "$K3/nodemon" "$K3/cache/attn_gym"
"${CMD[@]}"
