#!/bin/bash

export MILABENCH_BRANCH=amd_mi355
export CONFIG=all.yaml
export PYTHON_VERSION=3.12
export MILABENCH_GPU_ARCH=rocm
export MILABENCH_ARGS="${MILABENCH_ARGS:-}"

export MILABENCH_PUBLISH_KEY="{{ secrets.MILABENCH_PUBLISH_KEY }}"
export HF_TOKEN="{{ secrets.HF_TOKEN }}"
export MILABENCH_HF_TOKEN=$HF_TOKEN

set -ex

export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"

export MILABENCH_WORDIR="$(pwd)/$MILABENCH_GPU_ARCH"
export MILABENCH_BASE="$MILABENCH_WORDIR/results"
export MILABENCH_ENV="$MILABENCH_WORDIR/.env/$PYTHON_VERSION/"
export BENCHMARK_VENV="$MILABENCH_WORDIR/results/venv/torch"
export MILABENCH_SIZER_SAVE="$MILABENCH_WORDIR/scaling.yaml"

if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

UV="${UV:-$HOME/.local/bin/uv}"

ARGS="$@"

mkdir -p $MILABENCH_WORDIR
cd $MILABENCH_WORDIR

$UV venv $MILABENCH_ENV --python=$PYTHON_VERSION
git clone https://github.com/milabench/milabench.git -b $MILABENCH_BRANCH
export MILABENCH_SOURCE="$MILABENCH_WORDIR/milabench"

source $MILABENCH_ENV/bin/activate

$UV pip install -e $MILABENCH_SOURCE[$MILABENCH_GPU_ARCH]

export MILABENCH_CONFIG="$MILABENCH_SOURCE/config/$CONFIG"

milabench install --variant rocm $MILABENCH_ARGS

milabench prepare $MILABENCH_ARGS

milabench run $MILABENCH_ARGS

milabench report --runs $MILABENCH_WORDIR/results/runs
