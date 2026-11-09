#!/bin/bash
#
# Trillium - install + prepare step of a pipeline (see scripts/pipeline/TrilliumRun.json)
#
# Runs on a data mover node (tri-dm*) over ssh, OUTSIDE of Slurm: no SLURM_JOB_ID, no GPU,
# but internet access. It builds everything the run step needs on scratch, in a folder shared
# by all the steps of the pipeline run ($JR_PIPELINE_ID):
#   - python + venvs, milabench checkout, benchmark venvs
#   - datasets / model weights (milabench prepare)
#
# The run step (trillium_run.sh) runs on the compute nodes which have NO internet and cannot
# write to $HOME, it only reuses this folder.
#
# KEEP IN SYNC with trillium_run.sh
export MILABENCH_BRANCH=amd_mi355
export PYTHON_VERSION=3.12
export MILABENCH_GPU_ARCH=cuda
export MILABENCH_REPO=https://github.com/milabench/milabench.git
# torch 2.15 wheels are on the cu132 index (platforms.toml defaults)
export CUDA_VERSION="132"
export PYTORCH_VERSION="2.15.0"
# torchtitan is not in all.yaml: it lives in its own config, in the milabench checkout
export CONFIG_PATH=benchmarks/torchtitan/dev.yaml
# quick test: smoke entries only
export MILABENCH_ARGS="--select torchtitan-qwen3-4b-pretrain-smoke,torchtitan-qwen3-4b-sft-smoke"
# ---

export HF_TOKEN="{{ secrets.HF_TOKEN }}"
export MILABENCH_HF_TOKEN="$HF_TOKEN"

# No GPU here: tell the CUDA builds (TE, sglang, ...) which GPUs to compile for
#   H100 / H200 = 9.0, B200 = 10.0
export TORCH_CUDA_ARCH_LIST="9.0;10.0"

# Set it AFTER exporting the secrets :)
set -ex

# The ssh runner exports JR_JOB_DIR (job folder) and JR_PIPELINE_ID (shared by the run steps)
OUTPUT_DIRECTORY="${JR_JOB_DIR:-$PWD}"
mkdir -p $OUTPUT_DIRECTORY/meta

SCRATCH="${SCRATCH:-$HOME/links/scratch}"
export MILABENCH_WORDIR="$SCRATCH/milabench/pipelines/${JR_PIPELINE_ID:?run this script as a pipeline step}/$MILABENCH_GPU_ARCH"
export MILABENCH_BASE="$MILABENCH_WORDIR/results"
export MILABENCH_ENV="$MILABENCH_WORDIR/.env/$PYTHON_VERSION/"
export BENCHMARK_VENV="$MILABENCH_WORDIR/results/venv/torch"
export MILABENCH_SOURCE="$MILABENCH_WORDIR/milabench"
export MILABENCH_CONFIG="$MILABENCH_SOURCE/$CONFIG_PATH"
export MILABENCH_SHARED="$SCRATCH/shared"
export MILABENCH_USE_TOML_DEPS=1

# Everything that would land in $HOME goes to scratch: the run step cannot write to $HOME
# and it needs the python interpreter and the caches filled here
export UV_PYTHON_INSTALL_DIR="$MILABENCH_WORDIR/uv/python"
export UV_CACHE_DIR="$MILABENCH_WORDIR/uv/cache"
export XDG_CACHE_HOME="$MILABENCH_WORDIR/cache"
export HF_HOME="$MILABENCH_WORDIR/cache/huggingface"
export TORCH_HOME="$MILABENCH_WORDIR/cache/torch"

mkdir -p $MILABENCH_WORDIR $MILABENCH_BASE/runs
cd $MILABENCH_WORDIR

# uv
export UV=$HOME/.local/bin/uv
if [ ! -x "$UV" ]; then
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$MILABENCH_WORDIR/bin" UV_NO_MODIFY_PATH=1 sh
    export UV="$MILABENCH_WORDIR/bin/uv"
fi

# milabench (a retry of the step reuses the checkout)
if [ -d "$MILABENCH_SOURCE/.git" ]; then
    git -C $MILABENCH_SOURCE fetch origin $MILABENCH_BRANCH
    git -C $MILABENCH_SOURCE checkout -B $MILABENCH_BRANCH origin/$MILABENCH_BRANCH
else
    git clone $MILABENCH_REPO -b $MILABENCH_BRANCH $MILABENCH_SOURCE
fi
git -C $MILABENCH_SOURCE log -1 --pretty=format:"%H %s%n" | tee $OUTPUT_DIRECTORY/meta/milabench_commit.txt

$UV venv --allow-existing --python=$PYTHON_VERSION $MILABENCH_ENV
. $MILABENCH_ENV/bin/activate
$UV pip install -e $MILABENCH_SOURCE[$MILABENCH_GPU_ARCH]

# CUDA toolkit for the builds that compile kernels, if the module system has one
module load cuda 2>/dev/null || echo "no cuda module, builds use the pip CUDA toolkit"

# Single node, no GPU: only used for install/prepare, the run step makes its own on the GPU node
milabench slurm system > $MILABENCH_WORDIR/system.prepare.yaml

# data mover nodes are shared, do not take every core
export MAX_JOBS=${MAX_JOBS:-16}

# No pin file for this combination yet (.pin/constraints.cuda<ver>.torch<ver>.txt): generate it
# for the selected benchmarks, otherwise install stops with MissingPinError.
# Saved with the job so it can be committed to the repo.
# (with or without the arch suffix: constraints.cuda132.torch2150[.x86_64].txt)
PIN_GLOB="$MILABENCH_SOURCE/.pin/constraints.cuda$CUDA_VERSION.torch${PYTORCH_VERSION//./}*.txt"
# The pin covers EVERY benchmark of the repo (--select does not apply), so one incompatible
# benchmark blocks the whole combination.
if ! compgen -G "$PIN_GLOB" > /dev/null; then
    milabench tools pin --config $MILABENCH_CONFIG --set cuda=$CUDA_VERSION torch=$PYTORCH_VERSION $MILABENCH_ARGS
fi
# `milabench tools pin` exits 0 even when the pin fails: check its output
if ! compgen -G "$PIN_GLOB" > /dev/null; then
    echo "error: no pin file for cuda=$CUDA_VERSION torch=$PYTORCH_VERSION, the pin failed (see log.stdout)" >&2
    exit 1
fi
cp $PIN_GLOB $OUTPUT_DIRECTORY/meta/

# A failure here fails the step: the run jobs are cancelled instead of running on a broken env
milabench install --config $MILABENCH_CONFIG --set cuda=$CUDA_VERSION torch=$PYTORCH_VERSION --system $MILABENCH_WORDIR/system.prepare.yaml $MILABENCH_ARGS

milabench prepare --config $MILABENCH_CONFIG --system $MILABENCH_WORDIR/system.prepare.yaml $MILABENCH_ARGS

# Tells the run step the environment is complete
{
    echo "pipeline=$JR_PIPELINE_ID"
    echo "prepare_job=$JR_JOB_ID"
    echo "host=$(hostname)"
    echo "time=$(date -Is)"
} > $MILABENCH_WORDIR/.prepared
