#!/usr/bin/env bash
# Job of serve.sh: power check on every node, then one `milabench run` of $ENTRY (config/kimi_study.yaml) over the
# job's nodes, then a clean end on every node (vLLM and Ray stopped, no process left on a GPU). Needs MILABENCH_BASE and
# ENTRY (--export=ALL).
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
# in the background: bash runs a trap only between commands, `wait` lets the time-limit signal in
milabench run --config "$SRC/config/kimi_study.yaml" --system "$system" --select "$ENTRY" --run-name "$run" &
pid=$!
# serve.sh asks Slurm for USR1 5 min before --time
trap 'echo "[k3inf] time limit in 5 min: stopping the run"; kill -TERM $pid' USR1
wait $pid; status=$?
kill -0 $pid 2>/dev/null && { wait $pid; status=$?; }
echo "[k3inf] $ENTRY: exit $status after $(( $(date +%s) - start )) s, run data $MILABENCH_BASE/runs/$run"

# A node drained right after a job whose processes still held its GPUs at teardown (another workload, 2026-11-07):
# stop vLLM and Ray on every node, then wait until no process holds a GPU (the nodes are exclusive: any is ours)
srun --overlap --ntasks-per-node=1 bash -c '
    "$1/ray" stop --force > /dev/null 2>&1
    pkill -u "$USER" -f "[v]llm serve|[r]ay::|[r]aylet|[g]cs_server"
    for i in $(seq 60); do
        [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && { echo "[k3inf] $(hostname -s): GPUs free"; exit 0; }
        sleep 5
    done
    echo "[k3inf] $(hostname -s): GPUs still held after 5 min:"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
    exit 1' _ "$MILABENCH_BASE/venv/torch/bin" || echo "[k3inf] WARNING: a node still has processes on its GPUs" >&2
exit $status
