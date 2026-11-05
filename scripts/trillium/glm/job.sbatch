#!/usr/bin/env bash
# Job script of _common.sh `submit`: GLM-5 Megatron entries on Trillium, one `milabench run` each, in the order given.
# Needs MILABENCH_BASE, WORKDIR, MILABENCH_VENV, CONFIG, ENTRIES (space separated) and RUN_PREFIX (--export=ALL).
#   PREPARE=0         skip `milabench prepare`
#   CUDA_MODULE       loaded first (default cuda/12.9, empty: none). The benchmark venv brings its own CUDA (torch,
#                     TE and nvcc wheels): the module must not come before the venv on PATH (slime's make_env puts the
#                     wheels' nvcc first)
# F1 (checkpoints): every entry gets TT_CKPT_DIR=$CKPT_ROOT/<job name>.<job id>, the same for the whole job. A `-save`
# entry starts from an empty directory; after a `-load` entry the directory is deleted: run them as save, load, save, load, ... so that only one checkpoint
# exists at a time. The directory is also deleted when the job ends, whatever happened.
# Prints one PASS / FAIL line per entry and the largest per-rank peak memory, then a summary.
set -uo pipefail
: "${MILABENCH_BASE:?}" "${WORKDIR:?}" "${MILABENCH_VENV:?}" "${CONFIG:?}" "${ENTRIES:?}" "${RUN_PREFIX:?}"

if [ -n "${CUDA_MODULE:-}" ]; then
    module load StdEnv/2023 gcc/12.3 "$CUDA_MODULE" 2>/dev/null || echo "[$RUN_PREFIX] WARNING: module load $CUDA_MODULE failed" >&2
fi
export PATH=$MILABENCH_VENV/bin:$PATH
export MILABENCH_GPU_ARCH=cuda
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# caches on scratch: $HOME is read-only from Trillium's compute nodes
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$MILABENCH_BASE/cache}
# node-local files in RAM: /tmp holds 2 GB on the B200 nodes (TmpDisk=2048), as the Qwen launcher: TMPDIR (wandb's temporary files
# among them) and the kernel compile caches, which every rank would otherwise lock on the shared filesystem. Removed at the end
# (not when the job is killed at its limit: /dev/shm then waits for Slurm's epilog)
LOCAL_ROOT=${LOCAL_ROOT:-/dev/shm/$USER}
srun --overlap --nodes="$SLURM_JOB_NUM_NODES" --ntasks="$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 mkdir -p "$LOCAL_ROOT/tmp" \
    || echo "[$RUN_PREFIX] WARNING: could not create $LOCAL_ROOT on every node" >&2
export TMPDIR=$LOCAL_ROOT/tmp
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$LOCAL_ROOT/triton}
export TILELANG_CACHE_DIR=${TILELANG_CACHE_DIR:-$LOCAL_ROOT/tilelang}
export FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE:-$LOCAL_ROOT}  # its import creates $HOME/.cache/flashinfer otherwise
echo "[$RUN_PREFIX] local_root=$LOCAL_ROOT free: $(df -h "$(dirname "$LOCAL_ROOT")" | tail -1)"
local_cleanup() {
    echo "[$RUN_PREFIX] local_root before cleanup: $(du -sh "$LOCAL_ROOT" 2>/dev/null | cut -f1) on this node"
    srun --overlap --nodes="$SLURM_JOB_NUM_NODES" --ntasks="$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 rm -rf "$LOCAL_ROOT" || true
}
trap local_cleanup EXIT
# wandb (the GLM entries turn it on): through the cluster's HTTP proxy if there is one, else offline (`wandb sync`
# from a login node afterwards). An explicit WANDB_MODE is kept
if [ -z "${WANDB_MODE:-}" ]; then
    if command -v module >/dev/null && module load httpproxy 2>/dev/null; then
        export WANDB_BASE_URL=${WANDB_BASE_URL:-https://api.wandb.ai}
    else
        export WANDB_MODE=offline
    fi
fi
export WANDB_DIR=${WANDB_DIR:-$MILABENCH_BASE/wandb}/$RUN_PREFIX.$SLURM_JOB_ID
mkdir -p "$WANDB_DIR"
echo "[$RUN_PREFIX] wandb: mode=${WANDB_MODE:-online} proxy=${https_proxy:-${HTTPS_PROXY:-none}} dir=$WANDB_DIR"
read -ra entries <<< "$ENTRIES"   # ENTRY or ENTRY@EXPERIMENT (run.experiment of that run, else TT_EXPERIMENT)
select=$(IFS=,; echo "${entries[*]%@*}")
system=$MILABENCH_BASE/slurm/system.$SLURM_JOB_ID.yaml
echo "[$RUN_PREFIX] job=$SLURM_JOB_ID nodes=$SLURM_JOB_NODELIST ($SLURM_JOB_NUM_NODES) experiment=${TT_EXPERIMENT:-} campaign=${TT_CAMPAIGN:-}"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | sort | uniq -c | sed "s/^/[$RUN_PREFIX] /"

# topology at allocation time, when Slurm has one (Trillium: none, `scontrol show topology` is empty)
export TT_TOPOLOGY=$MILABENCH_BASE/slurm/topology.$SLURM_JOB_ID.txt
scontrol show topology > "$TT_TOPOLOGY" 2>/dev/null && [ -s "$TT_TOPOLOGY" ] || { rm -f "$TT_TOPOLOGY"; unset TT_TOPOLOGY; }
export TT_EXCLUDED=${TT_EXCLUDED:-$(scontrol show hostnames "${EXCLUDE:-}" 2>/dev/null \
    | python3 -c 'import json, sys; print(json.dumps({h: "excluded by the launcher (EXCLUDE)" for h in sys.stdin.read().split()}))')}

milabench slurm system > "$system" || { echo "[$RUN_PREFIX] milabench slurm system failed" >&2; exit 1; }
cat "$system"

if [ "${PREPARE:-1}" = 1 ]; then
    srun --nodes=1 --ntasks=1 --nodelist="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)" \
        milabench prepare --config "$CONFIG" --select "$select" \
        || echo "[$RUN_PREFIX] prepare failed, see above" >&2
fi

# F1: one checkpoint directory for the job, deleted at the end whatever happens. A job that saves refuses to start beside
# anything already in CKPT_ROOT (a stale or concurrent checkpoint would double ~9 TB on VAST)
export TT_CKPT_DIR=${CKPT_ROOT:?}/$RUN_PREFIX.$SLURM_JOB_ID
case " ${ENTRIES} " in
    *-save*)
        if [ -n "$(ls -A "$CKPT_ROOT" 2>/dev/null)" ]; then
            echo "[$RUN_PREFIX] REFUSED: $CKPT_ROOT is not empty, delete what is there or wait for its job:" >&2
            ls -la "$CKPT_ROOT" >&2
            exit 3
        fi ;;
esac
ckpt_rm() {  # delete the checkpoint; fails if it is still there
    [ -d "$TT_CKPT_DIR" ] || return 0
    rm -rf "$TT_CKPT_DIR"
    if [ -e "$TT_CKPT_DIR" ]; then
        echo "[$RUN_PREFIX] ERROR: could not delete $TT_CKPT_DIR" >&2
        return 1
    fi
}
trap 'ckpt_rm || true; local_cleanup' EXIT

peak() {
    grep -rhoE "\"$1\": ?[0-9.]+" "$2" 2>/dev/null | grep -oE '[0-9.]+$' | sort -g | tail -1
}

summary=()
status_all=0
declare -A seen
experiment=${TT_EXPERIMENT:-}
for item in "${entries[@]}"; do
    entry=${item%@*}
    export TT_EXPERIMENT=$experiment
    [ "$item" != "$entry" ] && export TT_EXPERIMENT=${item#*@}
    # an entry that comes back in the job (F1's load) gets .2, .3, ... : one run name (and nodemon directory) per run
    seen[$entry]=$(( ${seen[$entry]:-0} + 1 ))
    run=$RUN_PREFIX.$SLURM_JOB_ID.$entry
    [ "${seen[$entry]}" -gt 1 ] && run=$run.${seen[$entry]}
    export MILABENCH_RUN_NAME=$run
    export BENCHMATE_NODEMON_DIR=${NODEMON_ROOT:?}/$run
    mkdir -p "$BENCHMATE_NODEMON_DIR" || echo "[$RUN_PREFIX] WARNING: could not create $BENCHMATE_NODEMON_DIR" >&2
    case $entry in
        *-save*)
            if ! ckpt_rm; then  # never a second checkpoint beside one that could not be deleted
                status_all=1
                summary+=("$entry: NOT RUN, the previous checkpoint is still there")
                continue
            fi
            mkdir -p "$TT_CKPT_DIR" ;;
    esac
    echo "[$RUN_PREFIX] === $entry   (run $run)  nodemon=$BENCHMATE_NODEMON_DIR"
    start=$(date +%s)
    export TT_VARIANT=${VARIANT:-$entry}
    milabench run --config "$CONFIG" --system "$system" --select "$entry" --run-name "$run"
    status=$?
    if [ $status != 0 ]; then
        status_all=$status
        # a segfault or a kill leaves other ranks and the checkpoint worker behind: none may write during the next steps
        srun --overlap --nodes="$SLURM_JOB_NUM_NODES" --ntasks="$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 \
            pkill -KILL -f '[m]egatron/main.py|[m]ultiprocessing.spawn|[m]ultiprocessing.resource_tracker' || true
    fi
    rundir=$MILABENCH_BASE/runs/$run
    line="$entry: $([ $status = 0 ] && echo PASS || echo "FAIL ($status)")  $(( $(date +%s) - start ))s"
    line+="  peak allocated $(peak peak_memory_allocated "$rundir") / reserved $(peak peak_memory_reserved "$rundir") GiB"
    echo "[$RUN_PREFIX] $line"
    summary+=("$line")
    case $entry in *-load*) ckpt_rm || status_all=1 ;; esac
done

echo "[$RUN_PREFIX] ===== summary (runs in $MILABENCH_BASE/runs/$RUN_PREFIX.$SLURM_JOB_ID.*)"
printf '  %s\n' "${summary[@]}"
[ "${WANDB_MODE:-}" = offline ] && echo "[$RUN_PREFIX] wandb, from a login node: wandb sync $WANDB_DIR/wandb/offline-run-*"
exit $status_all
