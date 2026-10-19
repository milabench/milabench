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

# Pre-warmed torch.compile caches (Inductor + Triton): unpack an archive onto
# every node's local disk (same path as benchfile.py's cache dirs). Per-GPU
# shapes are identical at every scale, so a cache built on a few nodes is
# valid for large runs. TT_COMPILE_CACHE= (empty) disables it.
LOCAL_CACHE=${SLURM_TMPDIR:-/tmp}/tt_native_cache
CACHE_ARCHIVE=${TT_COMPILE_CACHE-$TT38/compile-cache.tar}
if [ -n "$CACHE_ARCHIVE" ] && [ -f "$CACHE_ARCHIVE" ]; then
    srun --ntasks-per-node=1 bash -c "mkdir -p $LOCAL_CACHE && tar -xf $CACHE_ARCHIVE -C $LOCAL_CACHE"
    echo "[tt-native] compile_cache_restored_epoch=$(date +%s.%N) archive=$CACHE_ARCHIVE"
fi

SYSTEM=$MILABENCH_BASE/system.$SLURM_JOB_ID.yaml
milabench slurm system > "$SYSTEM"
echo "[tt-native] system_ready_epoch=$(date +%s.%N)"

rc=0
milabench run --config "$MILABENCH_CONFIG" --system "$SYSTEM" --select "$BENCH" "$@" || rc=$?

if [ -n "$CACHE_ARCHIVE" ] && [ ! -f "$CACHE_ARCHIVE" ] && [ -d "$LOCAL_CACHE" ]; then
    tar -cf "$CACHE_ARCHIVE.tmp" -C "$LOCAL_CACHE" . && mv "$CACHE_ARCHIVE.tmp" "$CACHE_ARCHIVE"
    echo "[tt-native] compile_cache_saved archive=$CACHE_ARCHIVE size=$(du -h "$CACHE_ARCHIVE" | cut -f1)"
fi
echo "[tt-native] job_end_epoch=$(date +%s.%N)"
exit $rc
