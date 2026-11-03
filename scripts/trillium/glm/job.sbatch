#!/usr/bin/env bash
# Job script of _common.sh `submit`: GLM-5 Megatron entries on Trillium, one `milabench run` each, in the order given.
# Needs MILABENCH_BASE, WORKDIR, MILABENCH_VENV, CONFIG, ENTRIES (space separated) and RUN_PREFIX (--export=ALL).
#   PREPARE=0         skip `milabench prepare`
#   CUDA_MODULE       loaded first (default cuda/12.9, empty: none). The benchmark venv brings its own CUDA (torch,
#                     TE and nvcc wheels): the module must not come before the venv on PATH (slime's make_env puts the
#                     wheels' nvcc first)
# F1 (checkpoints): every entry gets TT_CKPT_DIR=$CKPT_ROOT/<job name>.<job id>, the same for the whole job. A `-save`
# entry starts from an empty directory; after a `-load` entry the directory is deleted and the deletion timed
# ($MILABENCH_BASE/slurm/ckpt-rm.<job id>.jsonl): run them as save, load, save, load, ... so that only one checkpoint
# exists at a time. The directory is also deleted when the job ends, whatever happened.
# Prints one PASS / FAIL line per entry and the largest per-rank peak memory, then a summary.
set -uo pipefail
: "${MILABENCH_BASE:?}" "${WORKDIR:?}" "${MILABENCH_VENV:?}" "${CONFIG:?}" "${ENTRIES:?}" "${RUN_PREFIX:?}"

if [ -n "${CUDA_MODULE:-}" ]; then
    module load "$CUDA_MODULE" 2>/dev/null || echo "[$RUN_PREFIX] WARNING: module load $CUDA_MODULE failed" >&2
fi
export PATH=$MILABENCH_VENV/bin:$PATH
export MILABENCH_GPU_ARCH=cuda
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# caches on scratch: $HOME is read-only from Trillium's compute nodes
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$MILABENCH_BASE/cache}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$XDG_CACHE_HOME/triton}
export TILELANG_CACHE_DIR=${TILELANG_CACHE_DIR:-$XDG_CACHE_HOME/tilelang}
# wandb: through the cluster's HTTP proxy if there is one, else offline (`wandb sync` from a login node afterwards)
if [ "${BENCHMATE_WANDB:-0}" = 1 ]; then
    if command -v module >/dev/null && module load httpproxy 2>/dev/null; then
        export WANDB_BASE_URL=${WANDB_BASE_URL:-https://api.wandb.ai}
    else
        export WANDB_MODE=offline
    fi
fi
export WANDB_DIR=${WANDB_DIR:-$MILABENCH_BASE/wandb}/$RUN_PREFIX.$SLURM_JOB_ID
mkdir -p "$WANDB_DIR"
echo "[$RUN_PREFIX] wandb: BENCHMATE_WANDB=${BENCHMATE_WANDB:-0} mode=${WANDB_MODE:-online} proxy=${https_proxy:-${HTTPS_PROXY:-none}} dir=$WANDB_DIR"
read -ra entries <<< "$ENTRIES"
select=$(IFS=,; echo "${entries[*]}")
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

# F1: one checkpoint directory for the job, deleted at the end whatever happens
export TT_CKPT_DIR=${CKPT_ROOT:?}/$RUN_PREFIX.$SLURM_JOB_ID
ckpt_rm() {  # ckpt_rm RUN: delete the checkpoint, timed, one JSON line per deletion
    [ -d "$TT_CKPT_DIR" ] || return 0
    local bytes start end
    bytes=$(du -sb "$TT_CKPT_DIR" 2>/dev/null | cut -f1)
    start=$(date +%s.%N)
    rm -rf "$TT_CKPT_DIR"
    end=$(date +%s.%N)
    printf '{"run": "%s", "path": "%s", "bytes": %s, "rm_s": %s, "start_epoch": %s, "end_epoch": %s}\n' \
        "$1" "$TT_CKPT_DIR" "${bytes:-null}" "$(python3 -c "print(round($end - $start, 3))")" "$start" "$end" \
        | tee -a "$MILABENCH_BASE/slurm/ckpt-rm.$SLURM_JOB_ID.jsonl" | sed "s/^/[$RUN_PREFIX] checkpoint deleted: /"
}
trap 'ckpt_rm end-of-job' EXIT

peak() {
    grep -rhoE "\"$1\": ?[0-9.]+" "$2" 2>/dev/null | grep -oE '[0-9.]+$' | sort -g | tail -1
}

summary=()
status_all=0
declare -A seen
for entry in "${entries[@]}"; do
    # an entry that comes back in the job (F1's load) gets .2, .3, ... : one run name (and nodemon directory) per run
    seen[$entry]=$(( ${seen[$entry]:-0} + 1 ))
    run=$RUN_PREFIX.$SLURM_JOB_ID.$entry
    [ "${seen[$entry]}" -gt 1 ] && run=$run.${seen[$entry]}
    export MILABENCH_RUN_NAME=$run
    export BENCHMATE_NODEMON_DIR=${NODEMON_ROOT:?}/$run
    mkdir -p "$BENCHMATE_NODEMON_DIR" || echo "[$RUN_PREFIX] WARNING: could not create $BENCHMATE_NODEMON_DIR" >&2
    case $entry in *-save*) ckpt_rm "before $entry"; mkdir -p "$TT_CKPT_DIR" ;; esac
    echo "[$RUN_PREFIX] === $entry   (run $run)  nodemon=$BENCHMATE_NODEMON_DIR"
    start=$(date +%s)
    export TT_VARIANT=${VARIANT:-$entry}
    milabench run --config "$CONFIG" --system "$system" --select "$entry" --run-name "$run"
    status=$?
    [ $status = 0 ] || status_all=$status
    rundir=$MILABENCH_BASE/runs/$run
    line="$entry: $([ $status = 0 ] && echo PASS || echo "FAIL ($status)")  $(( $(date +%s) - start ))s"
    line+="  peak allocated $(peak peak_memory_allocated "$rundir") / reserved $(peak peak_memory_reserved "$rundir") GiB"
    echo "[$RUN_PREFIX] $line"
    summary+=("$line")
    case $entry in *-load*) ckpt_rm "$run" ;; esac
done

echo "[$RUN_PREFIX] ===== summary (runs in $MILABENCH_BASE/runs/$RUN_PREFIX.$SLURM_JOB_ID.*)"
printf '  %s\n' "${summary[@]}"
[ "${WANDB_MODE:-}" = offline ] && echo "[$RUN_PREFIX] wandb, from a login node: wandb sync $WANDB_DIR/wandb/offline-run-*"
exit $status_all
