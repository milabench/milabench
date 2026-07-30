#!/bin/bash

export MILABENCH_BRANCH=main
export CONFIG=all.yaml
export PYTHON_VERSION=3.12
export MILABENCH_GPU_ARCH=cuda
export HF_TOKEN=""
export MILABENCH_ARGS="--select torchsrun"
export MILABENCH_REPO=https://github.com/milabench/milabench.git
export CUDA_VERSION="130"
export PYTORCH_VERSION="2.10.0"

set -ex

# ===
OUTPUT_DIRECTORY=$(scontrol show job "$SLURM_JOB_ID" --json | jq -r '.jobs[0].standard_output' | xargs dirname)
mkdir -p $OUTPUT_DIRECTORY/meta
scontrol show job --json $SLURM_JOB_ID | jq '.jobs[0]' > $OUTPUT_DIRECTORY/meta/info.json
# ===

module load cuda/12.6.0

export MILABENCH_WORDIR="/tmp/$SLURM_JOB_ID/$MILABENCH_GPU_ARCH"
export MILABENCH_BASE="$MILABENCH_WORDIR/results"
export MILABENCH_ENV="$MILABENCH_WORDIR/.env/$PYTHON_VERSION/"
export BENCHMARK_VENV="$MILABENCH_WORDIR/results/venv/torch"
export MILABENCH_SIZER_SAVE="$MILABENCH_WORDIR/scaling.yaml"
export MILABENCH_HF_TOKEN="$HF_TOKEN"
export MILABENCH_SHARED="$HOME/scratch/shared"
export MILABENCH_SOURCE="$MILABENCH_WORDIR/milabench"

mkdir -p $MILABENCH_WORDIR
cd $MILABENCH_WORDIR
git clone $MILABENCH_REPO -b $MILABENCH_BRANCH

export UV=$HOME/.local/bin/uv
$UV venv --python=$PYTHON_VERSION $MILABENCH_ENV
. $MILABENCH_ENV/bin/activate
mkdir -p $MILABENCH_WORDIR/results/runs
$UV pip install -e $MILABENCH_SOURCE[$MILABENCH_GPU_ARCH]

milabench slurm system > $MILABENCH_WORDIR/system.yaml

export MILABENCH_USE_TOML_DEPS=1 

milabench install --set cuda=$CUDA_VERSION torch=$PYTORCH_VERSION --system $MILABENCH_WORDIR/system.yaml $MILABENCH_ARGS

milabench prepare --system $MILABENCH_WORDIR/system.yaml $MILABENCH_ARGS

milabench run --system $MILABENCH_WORDIR/system.yaml $MILABENCH_ARGS

rsync -az $MILABENCH_WORDIR/results/runs $OUTPUT_DIRECTORY

# ===
scontrol show job --json $SLURM_JOB_ID | jq '.jobs[0]' > $OUTPUT_DIRECTORY/meta/info.json
# ===
