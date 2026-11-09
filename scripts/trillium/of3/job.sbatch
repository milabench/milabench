#!/bin/bash
# OF3 bring-up job: OpenFold3 0.5.1 training on the job's nodes, one run per VARIANT of STEPS steps each (entries of
# variants.yml, comma-separated to combine them),
# VARIANT+tf32 the same with run_openfold's --use_tf32 True (TF32 for FP32 matmuls; cuEquivariance's triangle attention
# backward needs it);
# 8 ranks per node (DDP, one GPU and 12 cores each, Lightning's SLURM environment). The runner yaml is
# initial_training.yml (the published first stage on the data.sh subset) merged with the variant (merge.py), run by
# train.py (run_openfold train plus a per-step log). Per run in $O/runs/of3.<job>.<variant>: runner.yml, train.log,
# steps-r<rank>.jsonl, nodemon/ (benchmate's node monitor: GPU power and clocks, CPU clocks, InfiniBand bytes), nccl/
# (NCCL's INIT and NET log per process), samples/ (featurization time per sample, per data-loader worker); with variant
# prof, prof/ (each node's local rank 0 trace), prof.txt (prof.py on the first node's trace) and the gradient-sync timing
# of train.py (OF3_SYNC_TIMING; both slow the steps: timing runs go without prof). Core dumps off.
# Submit from a login node, the lane's nodes in --nodelist, a job name nw-of3-...:
#   /opt/slurm/bin/.sbatch --export=NONE --get-user-env --job-name=nw-of3-s1 --account=rrg-bengioy-ad \
#     --partition=compute_b200_full_node --reservation=b200-friendly-users --qos=bump --nodes=1 --ntasks-per-node=8 \
#     --cpus-per-task=12 --exclusive --gpus-per-node=b200:8 --time=00:30:00 --nodelist=... \
#     --output=$SCRATCH/trillium/new/of3/slurm/%x-%j.out job.sbatch STEPS VARIANT...
ulimit -c 0
set -uo pipefail
unset PYTHONPATH PIP_CONFIG_FILE
O=$SCRATCH/trillium/new/of3
S=$O/src/scripts/trillium/of3
PY=$O/venv/bin/python
# nvcc for DeepSpeed's kernel, built at first use (variant ds) into TORCH_EXTENSIONS_DIR
export CUDA_HOME=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Core/cudacore/13.2.0
export PATH=$CUDA_HOME/bin:$PATH PYTHONUNBUFFERED=1 WANDB_MODE=disabled
LOCAL=/dev/shm/$USER-of3
export TRITON_CACHE_DIR=$LOCAL/triton TORCH_EXTENSIONS_DIR=$O/cache/torch_extensions XDG_CACHE_HOME=$O/cache \
    OPENFOLD_CACHE=$O/cache/openfold3
NODES=$SLURM_JOB_NUM_NODES
run_nodes() { srun --export=ALL --overlap --propagate=CORE --nodes=$NODES --ntasks=$NODES --ntasks-per-node=1 --cpus-per-task=1 "$@"; }
run_nodes mkdir -p "$LOCAL/tmp"
trap 'run_nodes rm -rf "$LOCAL"' EXIT
export TMPDIR=$LOCAL/tmp
STEPS=$1
shift
echo "[of3] job=$SLURM_JOB_ID nodes=$SLURM_JOB_NODELIST steps=$STEPS $(git -C $O/src rev-parse --short HEAD) ulimit -c $(ulimit -c)"
run_nodes bash "$O/src/scripts/trillium/power_check.sh" || exit 4
run_nodes bash -c 'echo "[of3] $(hostname -s): $(nvidia-smi --query-gpu=name,driver_version,enforced.power.limit --format=csv,noheader | sort | uniq -c | xargs)"'
status_all=0
for variant in "$@"; do
    run=$O/runs/of3.$SLURM_JOB_ID.$variant
    mkdir -p "$run"
    name=${variant%+tf32}
    tf32=False
    [ "$name" != "$variant" ] && tf32=True
    $PY "$S/merge.py" "$S/initial_training.yml" "$S/variants.yml" "$name" "$O/data" "$run" "$NODES" "$STEPS" > "$run/runner.yml" || exit 1
    run_nodes $PY -m benchmate.nodemon --out "$run/nodemon" --bench "of3-$variant" 2> "$run/nodemon.log" &
    monitor=$!
    echo "[of3] === $variant ($run) $(date +%T)"
    start=$(date +%s)
    mkdir -p "$run/nccl" "$run/samples"
    timing=0
    [[ $name == *prof* ]] && timing=1
    OF3_SAMPLELOG="$run/samples/{pid}.jsonl" OF3_SYNC_TIMING=$timing NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET NCCL_DEBUG_FILE="$run/nccl/%h.%p.log" OF3_STEPLOG="$run/steps-r{rank}.jsonl" srun --export=ALL --propagate=CORE --kill-on-bad-exit=1 --cpus-per-task=12 \
        $PY "$S/train.py" train --runner-yaml "$run/runner.yml" --seed 42 --use_tf32 $tf32 > "$run/train.log" 2>&1
    status=$?
    kill $monitor; wait $monitor
    [ $status != 0 ] && status_all=$status
    trace=$(ls "$run"/prof/*.pt.trace.json 2>/dev/null | head -1)
    [ -n "$trace" ] && srun --export=ALL --overlap --propagate=CORE --nodes=1 --ntasks=1 $PY "$S/prof.py" "$trace" > "$run/prof.txt" 2>&1
    echo "[of3] $variant: $([ $status = 0 ] && echo PASS || echo "FAIL ($status)") $(( $(date +%s) - start )) s, $(cat "$run"/steps-r*.jsonl 2>/dev/null | grep -c '"step"') step lines"
done
exit $status_all
