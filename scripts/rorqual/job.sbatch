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
read -ra entries <<< "$ENTRIES"
select=$(IFS=,; echo "${entries[*]}")
system=$MILABENCH_BASE/slurm/system.$SLURM_JOB_ID.yaml
echo "[$RUN_PREFIX] job=$SLURM_JOB_ID nodes=$SLURM_JOB_NODELIST ($SLURM_JOB_NUM_NODES)"

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
# the ladders: the steps after read the rung from there (_common.sh full_rung)
[ -n "${RESULT_FILE:-}" ] && echo "${passed:-NONE} $SLURM_JOB_ID" > "$RESULT_FILE"
[ -n "$passed" ]
