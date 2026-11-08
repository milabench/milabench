#!/bin/bash
# GLM-FP4 step 1, one B200 node: builds TransformerEngine 2.18.0 into the GLM-FP4 venv offline, as the campaign did
# ($SCRATCH/trillium/glm/te_build.sbatch: TE from the wheelhouse, the rest from the uv cache, NVTE_CUDA_ARCHS=100),
# with the GLM-FP4 uv cache, copy mode and the campaign versions as constraints (install-login.sh). Then runs
# nvfp4_check.py on one GPU: TE's NVFP4 recipe at GLM-5's GEMM shapes (quantizers, time, memory, error against bf16).
# Submit from a login node (lane nodes in NODELIST):
#   /opt/slurm/bin/.sbatch --export=NONE --get-user-env --job-name=nw-glm-fp4-te --account=rrg-bengioy-ad \
#     --partition=compute_b200_full_node --reservation=b200-friendly-users --qos=bump --nodes=1 --exclusive \
#     --gpus-per-node=b200:8 --time=00:40:00 --nodelist=$NODELIST \
#     --output=$SCRATCH/trillium/new/glm-fp4/slurm/%x-%j.out te_build.sbatch
ulimit -c 0
unset PYTHONPATH PIP_CONFIG_FILE
G=$SCRATCH/trillium/new/glm-fp4
export PATH=$G/src/.venv/bin:$HOME/.local/bin:$PATH
export UV_CACHE_DIR=$G/uv-cache UV_LINK_MODE=copy UV_CONSTRAINT=$G/pins-campaign.txt
export UV_OFFLINE=1 UV_NO_INDEX=1 UV_FIND_LINKS=$G/wheelhouse
export NVTE_CUDA_ARCHS=100 NVTE_PYTORCH_FORCE_BUILD=TRUE MILABENCH_GPU_ARCH=cuda MILABENCH_BASE=$G
export TMPDIR=/dev/shm/$USER/tmp XDG_CACHE_HOME=$G/cache; mkdir -p $TMPDIR
trap "rm -rf /dev/shm/$USER" EXIT
echo "[glm-fp4] $(hostname -s) ulimit -c $(ulimit -c)"
nvidia-smi --query-gpu=name,driver_version,power.limit --format=csv,noheader | sort | uniq -c
start=$(date +%s)
milabench install --config $G/src/benchmarks/megatron/dev.yaml --select megatron-glm5-tri-test-n2-gbs16
echo "[glm-fp4] install rc=$? in $(( $(date +%s) - start )) s"
cat $G/venv/torch/.slime_installed
srun --export=ALL --propagate=CORE --overlap --ntasks=1 bash -c 'echo "[glm-fp4] step ulimit -c $(ulimit -c)"; \
    CUDA_VISIBLE_DEVICES=0 '"$G"'/venv/torch/bin/python '"$G"'/src/scripts/trillium/glm-fp4/nvfp4_check.py'
echo "[glm-fp4] check rc=$?"
