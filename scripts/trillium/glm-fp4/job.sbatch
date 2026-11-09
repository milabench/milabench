#!/bin/bash
# GLM-FP4 bring-up job: one `milabench run` per entry of benchmarks/megatron/dev.yaml, in the order given, as the
# campaign's scripts/trillium/glm/job.sbatch, with the GLM-FP4 base ($G: venv, data, runs, nodemon, caches), core dumps
# off (ulimit here, --propagate=CORE on its srun steps), wandb disabled. An entry given as lt:ENTRY runs with cuBLASLt's
# trace log in $G/logs (the operand types of every GEMM; its times are not used). exp:NAME sets run.experiment for the
# entries after it (default GLM-FP4-bringup). check:MODEL runs nvfp4_check.py MODEL
# (glm5, qwen3, moe) on one GPU, with the venv's CUDA headers for TE's runtime-compiled kernels (milabench sets them for runs).
# Submit from a login node, the lane's nodes in --nodelist, a job name nw-glm-fp4-...:
#   /opt/slurm/bin/.sbatch --export=NONE --get-user-env --job-name=nw-glm-fp4-s2 --account=rrg-bengioy-ad \
#     --partition=compute_b200_full_node --reservation=b200-friendly-users --qos=bump --nodes=1 --ntasks-per-node=1 \
#     --exclusive --gpus-per-node=b200:8 --time=00:45:00 --nodelist=... --output=$G/slurm/%x-%j.out job.sbatch [src:DIR] ENTRY...
ulimit -c 0
set -uo pipefail
unset PYTHONPATH PIP_CONFIG_FILE
# the srun steps milabench starts on the other nodes (benchmarks/megatron SrunOneNode) get this job's environment
export SLURM_EXPORT_ENV=ALL
G=$SCRATCH/trillium/new/glm-fp4
# src:DIR as the first item: run that milabench checkout (with its own .venv) instead of $G/src
SRC=$G/src
if [ "${1#src:}" != "${1:-}" ]; then SRC=${1#src:}; shift; fi
export MILABENCH_BASE=$G MILABENCH_GPU_ARCH=cuda PYTHONUNBUFFERED=1 PATH=$SRC/.venv/bin:$PATH
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 XDG_CACHE_HOME=$G/cache
export WANDB_MODE=disabled TORCHINDUCTOR_COMPILE_THREADS=2
export TT_CAMPAIGN=trillium-new-$(date +%m%d) TT_EXPERIMENT=GLM-FP4-bringup TT_REPETITION=1 TT_FAMILY=slurm
CONFIG=$SRC/benchmarks/megatron/dev.yaml
LOCAL=/dev/shm/$USER
run_all() { srun --export=ALL --overlap --propagate=CORE --nodes="$SLURM_JOB_NUM_NODES" --ntasks="$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 "$@"; }
run_all mkdir -p "$LOCAL/tmp"
trap 'run_all rm -rf "$LOCAL"' EXIT
export TMPDIR=$LOCAL/tmp TRITON_CACHE_DIR=$LOCAL/triton TILELANG_CACHE_DIR=$LOCAL/tilelang FLASHINFER_WORKSPACE_BASE=$LOCAL \
    SGLANG_JIT_CACHE_DIR=$LOCAL/sglang-jit
echo "[glm-fp4] job=$SLURM_JOB_ID nodes=$SLURM_JOB_NODELIST $SRC $(git -C $SRC rev-parse --short HEAD) ulimit -c $(ulimit -c)"
run_all bash "$SRC/scripts/trillium/power_check.sh" || exit 4
system=$G/slurm/system.$SLURM_JOB_ID.yaml
milabench slurm system > "$system" || exit 1
mkdir -p $G/logs $G/prof
status_all=0
SP=$(echo $G/venv/torch/lib/python3.*/site-packages)
for item in "$@"; do
    if [ "${item#exp:}" != "$item" ]; then
        export TT_EXPERIMENT=${item#exp:}
        continue
    fi
    if [ "${item#check:}" != "$item" ]; then
        args=${item#check:}
        # check:moe[:EP[:MODEL...]]: the models' config.json, staged from a login node
        [ "${args%%:*}" = moe ] && args="moe $G/data/moe-configs $(echo ${args#moe} | tr ':' ' ')"
        srun --export=ALL --overlap --propagate=CORE --ntasks=1 env CUDA_VISIBLE_DEVICES=0 CUDA_HOME=$SP/nvidia/cu13 \
            NVTE_CUDA_INCLUDE_DIR=$SP/nvidia/cu13/include LD_LIBRARY_PATH=$SP/nvidia/cu13/lib:${LD_LIBRARY_PATH:-} \
            $G/venv/torch/bin/python $SRC/scripts/trillium/glm-fp4/nvfp4_check.py $args
        echo "[glm-fp4] $item rc=$?"
        continue
    fi
    entry=${item#lt:}
    run=glm-fp4.$SLURM_JOB_ID.$entry
    [ "$item" != "$entry" ] && run=$run.lt
    export MILABENCH_RUN_NAME=$run TT_VARIANT=$entry BENCHMATE_NODEMON_DIR=$G/nodemon/$run
    mkdir -p "$BENCHMATE_NODEMON_DIR"
    echo "[glm-fp4] === $item (run $run)"
    start=$(date +%s)
    if [ "$item" != "$entry" ]; then
        CUBLASLT_LOG_LEVEL=2 CUBLASLT_LOG_FILE=$G/logs/$run.%i.log \
            milabench run --config "$CONFIG" --system "$system" --select "$entry" --run-name "$run"
    else
        milabench run --config "$CONFIG" --system "$system" --select "$entry" --run-name "$run"
    fi
    status=$?
    # an entry with Megatron's profiler leaves its trace in $G/prof/torch_profile (dev.yaml _qwen3_prof)
    [ -d $G/prof/torch_profile ] && mv $G/prof/torch_profile $G/prof/$run
    [ $status != 0 ] && status_all=$status && run_all pkill -KILL -f '[m]egatron/main.py' || true
    peak=$(grep -rhoE '"mem_alloc_peak_gib": ?[0-9.]+' "$G/runs/$run" 2>/dev/null | grep -oE '[0-9.]+$' | sort -g | tail -1)
    echo "[glm-fp4] $entry: $([ $status = 0 ] && echo PASS || echo "FAIL ($status)") $(( $(date +%s) - start )) s, peak allocated ${peak:-?} GiB"
done
exit $status_all
