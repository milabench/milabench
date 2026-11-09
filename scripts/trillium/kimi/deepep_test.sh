#!/usr/bin/env bash
# K3-XN: DeepEP's own EP test (tests/elastic/test_ep.py, correctness and bandwidth) across the job's nodes, 8 GPUs each, at
# K3's shape (hidden 3,584, top-16 of 896 experts), at 128 tokens per rank (decode) and 4,096 (prefill), with the venv
# deepep_build.sh made. Across nodes it goes over RDMA (NCCL Gin), before vLLM uses it.
#
#   /opt/slurm/bin/.sbatch --export=NONE --get-user-env --nodes=2 --nodelist=<nodes> scripts/trillium/kimi/deepep_test.sh
#SBATCH --job-name=nw-k3-xn-deepep-test
#SBATCH --account=rrg-bengioy-ad --partition=compute_b200_full_node --reservation=b200-friendly-users --qos=bump
#SBATCH --gpus-per-node=b200:8 --ntasks-per-node=1 --exclusive --mem=0 --time=0:20:00
#SBATCH --exclude=trig0125,trig0132,trig0150
#SBATCH --output=/scratch/%u/trillium/new/k3-xn/slurm/%x.%j.out
set -uo pipefail
ulimit -c 0
X=$SCRATCH/trillium/new/k3-xn
export CUDA_HOME=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Core/cudacore/13.2.0
export PATH=$CUDA_HOME/bin:$PATH EP_JIT_CACHE_DIR=/dev/shm/$USER/deepep-jit
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1) MASTER_PORT=8361 WORLD_SIZE=$SLURM_NNODES
unset PYTHONPATH
echo "[deepep] job=$SLURM_JOB_ID nodes=$SLURM_JOB_NODELIST master=$MASTER_ADDR $(git -C "$X/deepep-src" log -1 --oneline)"
for tokens in 128 4096; do
    echo "[deepep] --num-tokens $tokens"
    srun --propagate=CORE --ntasks-per-node=1 bash -c 'set -o pipefail; RANK=$SLURM_NODEID timeout 420 "$1/venv/deepep/bin/python" \
        "$1/deepep-src/tests/elastic/test_ep.py" --num-processes 8 --num-tokens "$2" --hidden 3584 --num-topk 16 \
        --num-experts 896 --test-first-only 2>&1 | sed "s/^/[$(hostname -s)] /"' _ "$X" "$tokens"
    echo "[deepep] --num-tokens $tokens: exit $?"
done
srun --propagate=CORE --ntasks-per-node=1 rm -rf "$EP_JIT_CACHE_DIR"
