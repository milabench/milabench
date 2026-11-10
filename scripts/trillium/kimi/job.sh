#!/usr/bin/env bash
# Job of serve.sh: power check on every node, then one `milabench run` of $ENTRY ($CONFIG, config/kimi_study.yaml by
# default) over the job's nodes with benchmate.nodemon sampling every node (nodemon/<run>/), then a clean end on every
# node (vLLM and Ray stopped, no process left on a GPU). Needs MILABENCH_BASE and ENTRY (--export=ALL).
set -uo pipefail
ulimit -c 0  # the compute nodes allow core dumps (a K3 crash left 681 GB of them, 2026-11-07); --propagate=CORE takes
# it to the other nodes' tasks (PropagateResourceLimits=NONE)
: "${MILABENCH_BASE:?}" "${ENTRY:?}"
SRC=$MILABENCH_BASE/src
# PYTHONDONTWRITEBYTECODE: K3-XN runs K3-INF's venv, whose files stay as they are (12 .pyc files written, review 2)
export PATH=$SRC/.venv/bin:$PATH MILABENCH_GPU_ARCH=cuda PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
# compute nodes: no internet, $HOME read-only
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 XDG_CACHE_HOME=$MILABENCH_BASE/cache
# node-local files in RAM (/tmp holds 2 GB): TMPDIR, FlashInfer's workspace and Triton's kernels, removed at the end.
# Triton's cache starts as a copy of $MILABENCH_BASE/cache/triton on each node: a kernel compiled during a measured pass
# stalls it (9.8 s, job 1262824), and 16 ranks of 2 nodes compiling into one cache on VAST raced (job 1269990). Kernels
# compiled in a job are not kept: the warm-ups compile them before the measured passes
LOCAL=/dev/shm/$USER/k3inf
srun --overlap --propagate=CORE --ntasks-per-node=1 bash -c 'mkdir -p "$1/tmp" "$1/triton" && cp -a "$2/." "$1/triton/"' \
    _ "$LOCAL" "$MILABENCH_BASE/cache/triton"
export TMPDIR=$LOCAL/tmp TRITON_CACHE_DIR=$LOCAL/triton FLASHINFER_WORKSPACE_BASE=$LOCAL
# DeepEP's JIT kernels (K3-XN): node-local, not in the read-only $HOME it defaults to, nor in one cache shared by 2 nodes
# (Triton's raced there, job 1269990)
export EP_JIT_CACHE_DIR=$LOCAL/deepep-jit
trap 'srun --overlap --propagate=CORE --ntasks-per-node=1 rm -rf "$LOCAL"' EXIT

run=$ENTRY.$SLURM_JOB_ID
echo "[k3inf] job=$SLURM_JOB_ID nodes=$SLURM_JOB_NODELIST entry=$ENTRY run=$run milabench=$(git -C "$SRC" rev-parse --short HEAD)" \
     "data=${MILABENCH_DIRS_DATA:-$MILABENCH_BASE/data}"
srun --overlap --propagate=CORE --ntasks-per-node=1 bash -c 'echo "[k3inf] $(hostname -s): ulimit -c $(ulimit -c)"; bash "$1"' \
    _ "$SRC/scripts/trillium/power_check.sh" || exit 4
system=$MILABENCH_BASE/slurm/system.$SLURM_JOB_ID.yaml
milabench slurm system > "$system" || exit 1
# node metrics (GPU power, clocks and throttle reasons, NVLink, InfiniBand, CPU, host memory), one sampler per node
nodemon=$MILABENCH_BASE/nodemon/$run
mkdir -p "$nodemon"
srun --overlap --propagate=CORE --ntasks-per-node=1 python -m benchmate.nodemon --out "$nodemon" --bench "$ENTRY" \
    2> "$nodemon/nodemon.log" &
monitor=$!
# stall investigation (README): py-spy stack dumps of one worker per node, every second
spy=
if [ -n "${PYSPY:-}" ]; then
    srun --overlap --propagate=CORE --ntasks-per-node=1 bash "$SRC/scripts/trillium/kimi/pyspy_dump.sh" "$PYSPY" "$nodemon" &
    spy=$!
fi
start=$(date +%s)
# in the background: bash runs a trap only between commands, `wait` lets the time-limit signal in
milabench run --config "${CONFIG:-$SRC/config/kimi_study.yaml}" --system "$system" --select "$ENTRY" --run-name "$run" &
pid=$!
# serve.sh asks Slurm for USR1 5 min before --time. milabench delays a first SIGTERM, a second one stops it (job 1262548)
trap 'echo "[k3inf] time limit in 5 min: stopping the run"; kill -TERM $pid; sleep 2; kill -TERM $pid' USR1
status=0
while kill -0 $pid 2>/dev/null; do wait $pid; status=$?; done
echo "[k3inf] $ENTRY: exit $status after $(( $(date +%s) - start )) s, run data $MILABENCH_BASE/runs/$run"
kill $monitor $spy; wait $monitor $spy
# each node's kernel log (Xid, NVLink, InfiniBand events), to line up with the run
srun --overlap --propagate=CORE --ntasks-per-node=1 bash -c 'dmesg -T | tail -n 5000 > "$1/$(hostname -s).dmesg.txt"' _ "$nodemon"

# A node drained right after a job whose processes still held its GPUs at teardown (another workload, 2026-11-07):
# stop vLLM and Ray on every node, then wait until no process holds a GPU (the nodes are exclusive: any is ours)
srun --overlap --propagate=CORE --ntasks-per-node=1 bash -c '
    "$1/ray" stop --force > /dev/null 2>&1
    pkill -u "$USER" -f "[v]llm serve|[r]ay::|[r]aylet|[g]cs_server"
    for i in $(seq 60); do
        [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && { echo "[k3inf] $(hostname -s): GPUs free"; exit 0; }
        sleep 5
    done
    echo "[k3inf] $(hostname -s): GPUs still held after 5 min:"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
    exit 1' _ "$MILABENCH_BASE/venv/torch/bin" || echo "[k3inf] WARNING: a node still has processes on its GPUs" >&2
exit $status
