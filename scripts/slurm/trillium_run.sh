#!/bin/bash
#
# Trillium - run step of a pipeline (see scripts/pipeline/TrilliumRun.json)
#
# Runs on the GPU compute nodes: NO internet access, $HOME is read only, no local disk.
# Everything (python, venvs, milabench, datasets, weights) was built on scratch by
# trillium_prepare.sh on a data mover node, in the folder shared by the steps of the
# pipeline run ($JR_PIPELINE_ID). Nothing is downloaded or installed here.
#
# Several run steps can share one prepare (parallel jobs on different GPUs): each one writes
# its results to its own run folder.
#
# KEEP IN SYNC with trillium_prepare.sh
export MILABENCH_BRANCH=amd_mi355
export PYTHON_VERSION=3.12
export MILABENCH_GPU_ARCH=cuda
export MILABENCH_REPO=https://github.com/milabench/milabench.git
export CUDA_VERSION="132"
export PYTORCH_VERSION="2.15.0"
export CONFIG_PATH=benchmarks/torchtitan/dev.yaml
export MILABENCH_ARGS="--select torchtitan-qwen3-4b-pretrain-smoke,torchtitan-qwen3-4b-sft-smoke"
# ---

export HF_TOKEN="{{ secrets.HF_TOKEN }}"
export MILABENCH_HF_TOKEN="$HF_TOKEN"

# Set it AFTER exporting the secrets :)
set -ex

# ===
OUTPUT_DIRECTORY=$(scontrol show job "$SLURM_JOB_ID" --json | jq -r '.jobs[0].standard_output' | xargs dirname)
mkdir -p $OUTPUT_DIRECTORY/meta
scontrol show job --json $SLURM_JOB_ID | jq '.jobs[0]' > $OUTPUT_DIRECTORY/meta/info.json
# ===

SCRATCH="${SCRATCH:-$HOME/links/scratch}"
export MILABENCH_WORDIR="$SCRATCH/milabench/pipelines/${JR_PIPELINE_ID:?run this script as a pipeline step}/$MILABENCH_GPU_ARCH"
export MILABENCH_BASE="$MILABENCH_WORDIR/results"
export MILABENCH_ENV="$MILABENCH_WORDIR/.env/$PYTHON_VERSION/"
export BENCHMARK_VENV="$MILABENCH_WORDIR/results/venv/torch"
export MILABENCH_SOURCE="$MILABENCH_WORDIR/milabench"
export MILABENCH_CONFIG="$MILABENCH_SOURCE/$CONFIG_PATH"
export MILABENCH_SHARED="$SCRATCH/shared"
export MILABENCH_USE_TOML_DEPS=1

if [ ! -f "$MILABENCH_WORDIR/.prepared" ]; then
    echo "error: $MILABENCH_WORDIR was not prepared (trillium_prepare.sh did not complete)"
    exit 1
fi
cat $MILABENCH_WORDIR/.prepared

# Same locations as the prepare step (interpreter, weights, datasets)
export UV_PYTHON_INSTALL_DIR="$MILABENCH_WORDIR/uv/python"
export UV_CACHE_DIR="$MILABENCH_WORDIR/uv/cache"
export HF_HOME="$MILABENCH_WORDIR/cache/huggingface"
export TORCH_HOME="$MILABENCH_WORDIR/cache/torch"

# Per job caches: compiled kernels are written while running, $HOME is read only
JOB_CACHE="$MILABENCH_WORDIR/jobs/$SLURM_JOB_ID"
mkdir -p $JOB_CACHE
export XDG_CACHE_HOME="$JOB_CACHE/cache"
export TRITON_CACHE_DIR="$JOB_CACHE/triton"
export TORCHINDUCTOR_CACHE_DIR="$JOB_CACHE/inductor"
export MILABENCH_SIZER_SAVE="$JOB_CACHE/scaling.yaml"
# no local disk, $SLURM_TMPDIR is a RAM disk
export TMPDIR="${SLURM_TMPDIR:-$JOB_CACHE/tmp}"
mkdir -p $TMPDIR

# No internet: fail fast instead of hanging on a download
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export UV_OFFLINE=1
export PIP_NO_INDEX=1
export WANDB_MODE=offline

. $MILABENCH_ENV/bin/activate

module load cuda 2>/dev/null || echo "no cuda module"

# On the GPU node(s) of this job
milabench slurm system > $JOB_CACHE/system.yaml

RUN_NAME="trillium.$SLURM_JOB_ID"
RC=0
milabench run --config $MILABENCH_CONFIG --system $JOB_CACHE/system.yaml --run-name $RUN_NAME $MILABENCH_ARGS || RC=$?

# Copy the results of every node of the job, not only the main one.
# Files should not clash; just in case, identical files are skipped (--checksum) and a
# different version is not lost (--backup renames it <file>.replaced-by.<host>)
RUNS_OUT="$OUTPUT_DIRECTORY/runs"
mkdir -p $RUNS_OUT

srun --overlap --nodes=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 bash -c \
    "rsync -a --checksum --backup --suffix=.replaced-by.\$SLURMD_NODENAME $MILABENCH_BASE/runs/$RUN_NAME $RUNS_OUT/" \
    || echo "collect failed on some nodes"

# ===
scontrol show job --json $SLURM_JOB_ID | jq '.jobs[0]' > $OUTPUT_DIRECTORY/meta/info.json
# ===

exit $RC
