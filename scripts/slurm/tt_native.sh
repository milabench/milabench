#!/bin/bash
# Run one torchtitan_native bench under Slurm (DRAC clusters).
#
#   sbatch --nodes=2 --time=00:45:00 --account=rrg-bengioy-ad_gpu \
#       scripts/slurm/tt_native.sh tt-qwen38-n2 [extra milabench run args]
#
# Expects (see journals/cluster-guidelines.md):
#   $TT38/env        venv with milabench installed (uv venv --python 3.12; uv pip install -e)
#   $TT38/milabench  this repository (branch torchtitan_qwen38)
#   $TT38/output     MILABENCH_BASE, with the bench already installed
#
#SBATCH --job-name=tt-native
#SBATCH --gpus-per-node=h100:4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --output=%x-%j.out

set -euo pipefail

BENCH=${1:?usage: tt_native.sh <bench-name> [milabench run args]}
shift

TT38=${TT38:-$SCRATCH/tt38}
# DRAC login profile points Python at cvmfs site-packages and the DRAC wheelhouse.
unset PYTHONPATH PIP_CONFIG_FILE
export PATH=$TT38/env/bin:$HOME/.local/bin:$PATH

export MILABENCH_BASE=$TT38/output
export MILABENCH_CONFIG=$TT38/milabench/config/torchtitan_native.yaml
export MILABENCH_GPU_ARCH=cuda
export PYTHONUNBUFFERED=1

echo "[tt-native] job_start_epoch=$(date +%s.%N) job=$SLURM_JOB_ID nodes=$SLURM_JOB_NODELIST"
squeue -h -j "$SLURM_JOB_ID" -o "[tt-native] submit=%V start=%S" || true
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | sort | uniq -c | sed 's/^/[tt-native] gpu /'
ip -br addr | grep -v "^lo" | sed 's/^/[tt-native] iface /'

SYSTEM=$MILABENCH_BASE/system.$SLURM_JOB_ID.yaml
milabench slurm system > "$SYSTEM"
echo "[tt-native] system_ready_epoch=$(date +%s.%N)"

milabench run --config "$MILABENCH_CONFIG" --system "$SYSTEM" --select "$BENCH" "$@"
echo "[tt-native] job_end_epoch=$(date +%s.%N)"
