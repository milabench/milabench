#!/bin/bash
# QW-CP bring-up job: one `milabench run` per entry of benchmarks/megatron/dev.yaml (Qwen3 32B, context parallel), in
# the order given, as scripts/trillium/glm-fp4/job.sbatch. Code from this workload's checkout $Q/src (its Megatron-LM
# copied from GLM-FP4's); runs, nodemon, caches and Slurm logs under $Q; the venv and the data (Qwen3's config and
# tokenizer, FineWeb) are GLM-FP4's, read only (MILABENCH_DIRS_VENV / _DATA). The venv's benchmate is GLM-FP4's checkout
# (editable install). Core dumps off (ulimit here, --propagate=CORE on srun), wandb disabled. exp:NAME sets
# run.experiment for the entries after it (default QW-CP-bringup).
# Submit from a login node, the lane's nodes in --nodelist, a job name nw-qw-cp-...:
#   /opt/slurm/bin/.sbatch --export=NONE --get-user-env --job-name=nw-qw-cp-s1 --account=rrg-bengioy-ad \
#     --partition=compute_b200_full_node --reservation=b200-friendly-users --qos=bump --nodes=1 --ntasks-per-node=1 \
#     --exclusive --gpus-per-node=b200:8 --time=00:30:00 --nodelist=... \
#     --output=$SCRATCH/trillium/new/qw-cp/slurm/%x-%j.out job.sbatch ENTRY...
ulimit -c 0
set -uo pipefail
unset PYTHONPATH PIP_CONFIG_FILE
# the srun steps milabench starts on the other nodes (benchmarks/megatron SrunOneNode) get this job's environment
export SLURM_EXPORT_ENV=ALL
Q=$SCRATCH/trillium/new/qw-cp
G=$SCRATCH/trillium/new/glm-fp4
SRC=$Q/src
export MILABENCH_BASE=$Q MILABENCH_DIRS_VENV=$G/venv/torch MILABENCH_DIRS_DATA=$G/data MILABENCH_GPU_ARCH=cuda \
    PYTHONUNBUFFERED=1 PATH=$SRC/.venv/bin:$PATH
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 XDG_CACHE_HOME=$Q/cache
export WANDB_MODE=disabled TORCHINDUCTOR_COMPILE_THREADS=2
export TT_CAMPAIGN=trillium-new-$(date +%m%d) TT_EXPERIMENT=QW-CP-bringup TT_REPETITION=1 TT_FAMILY=slurm
CONFIG=$SRC/benchmarks/megatron/dev.yaml
LOCAL=/dev/shm/$USER
run_all() { srun --export=ALL --overlap --propagate=CORE --nodes="$SLURM_JOB_NUM_NODES" --ntasks="$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 "$@"; }
run_all mkdir -p "$LOCAL/tmp"
trap 'run_all rm -rf "$LOCAL"' EXIT
export TMPDIR=$LOCAL/tmp TRITON_CACHE_DIR=$LOCAL/triton
echo "[qw-cp] job=$SLURM_JOB_ID nodes=$SLURM_JOB_NODELIST $SRC $(git -C $SRC rev-parse --short HEAD) ulimit -c $(ulimit -c)"
run_all bash "$SRC/scripts/trillium/power_check.sh" || exit 4
system=$Q/slurm/system.$SLURM_JOB_ID.yaml
milabench slurm system > "$system" || exit 1
status_all=0
for item in "$@"; do
    if [ "${item#exp:}" != "$item" ]; then
        export TT_EXPERIMENT=${item#exp:}
        continue
    fi
    entry=$item
    run=qw-cp.$SLURM_JOB_ID.$entry
    export MILABENCH_RUN_NAME=$run TT_VARIANT=$entry BENCHMATE_NODEMON_DIR=$Q/nodemon/$run
    mkdir -p "$BENCHMATE_NODEMON_DIR"
    echo "[qw-cp] === $entry (run $run)"
    start=$(date +%s)
    milabench run --config "$CONFIG" --system "$system" --select "$entry" --run-name "$run"
    status=$?
    [ $status != 0 ] && status_all=$status && run_all pkill -KILL -f '[m]egatron/main.py' || true
    peak=$(grep -rhoE '"mem_alloc_peak_gib": ?[0-9.]+' "$Q/runs/$run" 2>/dev/null | grep -oE '[0-9.]+$' | sort -g | tail -1)
    echo "[qw-cp] $entry: $([ $status = 0 ] && echo PASS || echo "FAIL ($status)") $(( $(date +%s) - start )) s, peak allocated ${peak:-?} GiB"
done
exit $status_all
