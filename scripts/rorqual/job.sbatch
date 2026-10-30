#!/usr/bin/env bash
# Job script of _common.sh `submit`: GLM-5 Megatron entries, one `milabench run` each, in the order given.
# Needs MILABENCH_BASE, WORKDIR, CONFIG, ENTRIES (space separated) and RUN_PREFIX in the environment (--export=ALL).
#   STOP_AT_FIRST=1   stop at the first entry that completes (the fallback ladders), else run them all
#   PREPARE=0         skip `milabench prepare`
#   RESULT_FILE       write "<first entry that completes | NONE> <job id>" there
#   LOCAL_BASE        node-local MILABENCH_BASE the runs use (default $SLURM_TMPDIR/milabench: same path on every node,
#                     wiped at the end of the job). On every node: data/{LOCAL_DATA} and cache/ are copied from the shared
#                     base (scratch, slow to read from every rank), venv/, runs/ and extra/ are symlinks to it: the
#                     installed venv is reused and the metrics land on scratch, watched live from the login node.
#                     LOCAL_BASE= (empty): run on the shared base directly
#   LOCAL_DATA        data folders copied (default "megatron fineweb")
#   Logs, system / topology files, wandb and nodemon stay on the shared base / $SCRATCH
#   nodemon (FORMAT.md §3): always on, one directory per entry
#       $SCRATCH/nodemon/<run name>
# Prints one PASS / FAIL line per entry and the largest per-rank peak memory, then a summary at the end.
set -uo pipefail
: "${MILABENCH_BASE:?}" "${WORKDIR:?}" "${CONFIG:?}" "${ENTRIES:?}" "${RUN_PREFIX:?}"

export PATH=$WORKDIR/.venv/bin:$PATH
export MILABENCH_GPU_ARCH=cuda
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# wandb from the compute nodes: through the cluster's HTTP proxy (its variables reach the other nodes through srun)
if [ "${BENCHMATE_WANDB:-0}" = 1 ]; then
    { command -v module >/dev/null && module load httpproxy; } \
        || echo "[$RUN_PREFIX] WARNING: module load httpproxy failed: wandb cannot reach its server" >&2
    # the proxy only lets api.wandb.ai through: wandb 0.31 defaults to api.forge.coreweave.com, refused (403), and each
    # rank then loses ~100 s at startup before training on without wandb. Same key and project (brdg-mila/milabench)
    export WANDB_BASE_URL=${WANDB_BASE_URL:-https://api.wandb.ai}
fi
# one wandb folder per job (every rank of every entry keeps its local run files there), on the shared filesystem
export WANDB_DIR=${WANDB_DIR:-$MILABENCH_BASE/wandb}/$RUN_PREFIX.$SLURM_JOB_ID
mkdir -p "$WANDB_DIR"
echo "[$RUN_PREFIX] fine-grained timings: BENCHMATE_RECORD_FINE=${BENCHMATE_RECORD_FINE:-0}  wandb: BENCHMATE_WANDB=${BENCHMATE_WANDB:-0}" \
     "all ranks=${BENCHMATE_WANDB_ALL_RANKS:-0} project=${BENCHMATE_WANDB_PROJECT:-} mode=${WANDB_MODE:-online}" \
     "proxy=${https_proxy:-${HTTPS_PROXY:-none}} base_url=${WANDB_BASE_URL:-default} dir=$WANDB_DIR"
read -ra entries <<< "$ENTRIES"
select=$(IFS=,; echo "${entries[*]}")
system=$MILABENCH_BASE/slurm/system.$SLURM_JOB_ID.yaml
echo "[$RUN_PREFIX] job=$SLURM_JOB_ID nodes=$SLURM_JOB_NODELIST ($SLURM_JOB_NUM_NODES)"

# run_start record (FORMAT.md): the topology at allocation time, read by the node placement AND recorded with the run
export TT_TOPOLOGY=$MILABENCH_BASE/slurm/topology.$SLURM_JOB_ID.txt
scontrol show topology > "$TT_TOPOLOGY" 2>/dev/null || { echo "[$RUN_PREFIX] WARNING: no topology snapshot" >&2; unset TT_TOPOLOGY; }
# excluded nodes {host: reason}: EXCLUDE_REASONS if given, else the EXCLUDE hostlist without a reason
if [ -z "${TT_EXCLUDED:-}" ]; then
    if [ -n "${EXCLUDE_REASONS:-}" ]; then
        TT_EXCLUDED=$EXCLUDE_REASONS
    else
        TT_EXCLUDED=$(scontrol show hostnames "${EXCLUDE:-}" 2>/dev/null \
            | python3 -c 'import json, sys; print(json.dumps({h: "excluded by the launcher (EXCLUDE), no reason given" for h in sys.stdin.read().split()}))')
    fi
fi
export TT_EXCLUDED
echo "[$RUN_PREFIX] campaign=${TT_CAMPAIGN:-} topology=${TT_TOPOLOGY:-none} excluded=$TT_EXCLUDED health=${TT_HEALTH_DIR:-none}"

# Node telemetry, always. The Megatron pack starts benchmate.nodemon from every rank when
# BENCHMATE_NODEMON_DIR is set. The directory name is the milabench run name (the join to
# the run data), and a second run in the same directory is refused, so each entry gets its own.
nodemon_root=${SCRATCH:?}/nodemon
echo "[$RUN_PREFIX] nodemon root=$nodemon_root"

milabench slurm system > "$system" || { echo "[$RUN_PREFIX] milabench slurm system failed" >&2; exit 1; }
cat "$system"

# node-local base: the same layout on every node (the srun steps of the other ranks inherit MILABENCH_BASE)
SHARED_BASE=$MILABENCH_BASE
LOCAL_BASE=${LOCAL_BASE-${SLURM_TMPDIR:?}/milabench}
if [ -n "$LOCAL_BASE" ]; then
    if srun --nodes="$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 bash -c '
            set -e; shared=$1 dst=$2; shift 2
            mkdir -p "$dst/data"
            for d in venv runs extra; do ln -sfn "$shared/$d" "$dst/$d"; done
            cp -r "$shared/cache" "$dst/cache"
            for d in "$@"; do cp -r "$shared/data/$d" "$dst/data/"; done
            echo "$(hostname):$(du -shL "$dst/data" | cut -f1)"' _ "$SHARED_BASE" "$LOCAL_BASE" ${LOCAL_DATA-megatron fineweb} \
            | sort | paste -sd' ' | sed "s/^/[$RUN_PREFIX] local base: /"; then
        export MILABENCH_BASE=$LOCAL_BASE
    else
        echo "[$RUN_PREFIX] WARNING: could not build $LOCAL_BASE on every node, running on $SHARED_BASE" >&2
    fi
fi
echo "[$RUN_PREFIX] base=$MILABENCH_BASE (data $(readlink -f "$MILABENCH_BASE/data"), runs $(readlink -f "$MILABENCH_BASE/runs"))"

# after the local base: prepare finds the copied data (no-op, the compute nodes are offline)
if [ "${PREPARE:-1}" = 1 ]; then
    srun --nodes=1 --ntasks=1 --nodelist="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)" \
        milabench prepare --config "$CONFIG" --select "$select" \
        || echo "[$RUN_PREFIX] prepare failed, see above" >&2
fi

# largest per-rank peak memory of a run (rank 0 in the run data, the other ranks in their rankstore files)
peak() {
    grep -rhoE "\"$1\": ?[0-9.]+" "$2" 2>/dev/null | grep -oE '[0-9.]+$' | sort -g | tail -1
}

summary=()
passed=
for entry in "${entries[@]}"; do
    run=$RUN_PREFIX.$SLURM_JOB_ID.$entry
    export MILABENCH_RUN_NAME=$run
    export BENCHMATE_NODEMON_DIR=$nodemon_root/$MILABENCH_RUN_NAME
    mkdir -p "$BENCHMATE_NODEMON_DIR" \
        || echo "[$RUN_PREFIX] WARNING: could not create $BENCHMATE_NODEMON_DIR" >&2
    echo "[$RUN_PREFIX] === $entry   (run $run)  nodemon=$BENCHMATE_NODEMON_DIR"
    start=$(date +%s)
    # run.variant: the G run (VARIANT, set by _g.sh) or the entry itself (the check ladders)
    export TT_VARIANT=${VARIANT:-$entry} TT_REPETITION=${REPETITION:-1}
    milabench run --config "$CONFIG" --system "$system" --select "$entry" --run-name "$run"
    status=$?
    rundir=$SHARED_BASE/runs/$run
    line="$entry: $([ $status = 0 ] && echo PASS || echo "FAIL ($status)")  $(( $(date +%s) - start ))s"
    line+="  peak allocated $(peak peak_memory_allocated "$rundir") / reserved $(peak peak_memory_reserved "$rundir") GiB"
    echo "[$RUN_PREFIX] $line"
    summary+=("$line")
    if [ $status = 0 ]; then
        passed=${passed:-$entry}
        [ "${STOP_AT_FIRST:-0}" = 1 ] && break
    fi
done

echo "[$RUN_PREFIX] ===== summary (runs in $SHARED_BASE/runs/$RUN_PREFIX.$SLURM_JOB_ID.*)"
printf '  %s\n' "${summary[@]}"
[ "${STOP_AT_FIRST:-0}" = 1 ] && echo "[$RUN_PREFIX] first entry that completes: ${passed:-NONE}"
[ "${WANDB_MODE:-}" = offline ] && echo "[$RUN_PREFIX] wandb, from a login node: wandb sync $WANDB_DIR/wandb/offline-run-*"
# the ladders: the steps after read the rung from there (_common.sh full_rung)
[ -n "${RESULT_FILE:-}" ] && echo "${passed:-NONE} $SLURM_JOB_ID" > "$RESULT_FILE"
[ -n "$passed" ]
