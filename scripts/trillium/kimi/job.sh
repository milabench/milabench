#!/usr/bin/env bash
# Job of serve.sh: power check on every node, then one `milabench run` of $ENTRY (config/kimi_study.yaml) over the
# job's nodes. Needs MILABENCH_BASE and ENTRY (--export=ALL).
set -uo pipefail
: "${MILABENCH_BASE:?}" "${ENTRY:?}"
SRC=$MILABENCH_BASE/src
export PATH=$SRC/.venv/bin:$PATH MILABENCH_GPU_ARCH=cuda PYTHONUNBUFFERED=1
# compute nodes: no internet, $HOME read-only
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 XDG_CACHE_HOME=$MILABENCH_BASE/cache
# node-local files in RAM (/tmp holds 2 GB): TMPDIR and the kernel compile caches, removed at the end
LOCAL=/dev/shm/$USER/k3inf
srun --overlap --ntasks-per-node=1 mkdir -p "$LOCAL/tmp"
export TMPDIR=$LOCAL/tmp TRITON_CACHE_DIR=$LOCAL/triton FLASHINFER_WORKSPACE_BASE=$LOCAL
trap 'srun --overlap --ntasks-per-node=1 rm -rf "$LOCAL"' EXIT

run=$ENTRY.$SLURM_JOB_ID
echo "[k3inf] job=$SLURM_JOB_ID nodes=$SLURM_JOB_NODELIST entry=$ENTRY run=$run milabench=$(git -C "$SRC" rev-parse --short HEAD)" \
     "data=${MILABENCH_DIRS_DATA:-$MILABENCH_BASE/data}"
srun --overlap --ntasks-per-node=1 bash "$SRC/scripts/trillium/power_check.sh" || exit 4
system=$MILABENCH_BASE/slurm/system.$SLURM_JOB_ID.yaml
milabench slurm system > "$system" || exit 1
start=$(date +%s)
milabench run --config "$SRC/config/kimi_study.yaml" --system "$system" --select "$ENTRY" --run-name "$run"
status=$?
echo "[k3inf] $ENTRY: exit $status after $(( $(date +%s) - start )) s, run data $MILABENCH_BASE/runs/$run"
exit $status
