#!/usr/bin/env bash
# Job script of _common.sh `submit`: GLM-5 Megatron entries, one `milabench run` each, in the order given.
# Needs MILABENCH_BASE, WORKDIR, CONFIG, ENTRIES (space separated) and RUN_PREFIX in the environment (--export=ALL).
#   STOP_AT_FIRST=1   stop at the first entry that completes (the fallback ladders), else run them all
#   PREPARE=0         skip `milabench prepare`
#   RESULT_FILE       write "<first entry that completes | NONE> <job id>" there
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
fi
# one wandb folder per job (every rank of every entry keeps its local run files there), on the shared filesystem
export WANDB_DIR=${WANDB_DIR:-$MILABENCH_BASE/wandb}/$RUN_PREFIX.$SLURM_JOB_ID
mkdir -p "$WANDB_DIR"
echo "[$RUN_PREFIX] fine-grained timings: BENCHMATE_RECORD_FINE=${BENCHMATE_RECORD_FINE:-0}  wandb: BENCHMATE_WANDB=${BENCHMATE_WANDB:-0}" \
     "all ranks=${BENCHMATE_WANDB_ALL_RANKS:-0} project=${BENCHMATE_WANDB_PROJECT:-} mode=${WANDB_MODE:-online}" \
     "proxy=${https_proxy:-${HTTPS_PROXY:-none}} dir=$WANDB_DIR"
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

milabench slurm system > "$system" || { echo "[$RUN_PREFIX] milabench slurm system failed" >&2; exit 1; }
cat "$system"

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
    echo "[$RUN_PREFIX] === $entry   (run $run)"
    start=$(date +%s)
    # run.variant: the G run (VARIANT, set by _g.sh) or the entry itself (the check ladders)
    export TT_VARIANT=${VARIANT:-$entry} TT_REPETITION=${REPETITION:-1}
    milabench run --config "$CONFIG" --system "$system" --select "$entry" --run-name "$run"
    status=$?
    rundir=$MILABENCH_BASE/runs/$run
    line="$entry: $([ $status = 0 ] && echo PASS || echo "FAIL ($status)")  $(( $(date +%s) - start ))s"
    line+="  peak allocated $(peak peak_memory_allocated "$rundir") / reserved $(peak peak_memory_reserved "$rundir") GiB"
    echo "[$RUN_PREFIX] $line"
    summary+=("$line")
    if [ $status = 0 ]; then
        passed=${passed:-$entry}
        [ "${STOP_AT_FIRST:-0}" = 1 ] && break
    fi
done

echo "[$RUN_PREFIX] ===== summary (runs in $MILABENCH_BASE/runs/$RUN_PREFIX.$SLURM_JOB_ID.*)"
printf '  %s\n' "${summary[@]}"
[ "${STOP_AT_FIRST:-0}" = 1 ] && echo "[$RUN_PREFIX] first entry that completes: ${passed:-NONE}"
[ "${WANDB_MODE:-}" = offline ] && echo "[$RUN_PREFIX] wandb, from a login node: wandb sync $WANDB_DIR/wandb/offline-run-*"
# the ladders: the steps after read the rung from there (_common.sh full_rung)
[ -n "${RESULT_FILE:-}" ] && echo "${passed:-NONE} $SLURM_JOB_ID" > "$RESULT_FILE"
[ -n "$passed" ]
