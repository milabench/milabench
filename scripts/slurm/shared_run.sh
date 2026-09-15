#!/bin/bash


export MILABENCH_BRANCH=main
export PYTHON_VERSION=3.12
export MILABENCH_GPU_ARCH=cuda
export PYTHONUNBUFFERED=0
export MILABENCH_ARGS=""
export MILABENCH_CONFIG_NAME=all
export MILABENCH_REPO=https://github.com/milabench/milabench.git
export CHERRYBIN_REPO=https://github.com/Delaunay/cherrybin.git
export CHERRYBIN_BRANCH=main
export CUDA_VERSION="130"
export PYTORCH_VERSION="2.10.0"

# Stream I/O chunk in bytes; unset keeps cherrybin's default (16MiB).
export CHERRYBIN_IO_CHUNK=""

export MILABENCH_PUBLISH_KEY="{{ secrets.MILABENCH_PUBLISH_KEY }}"
export HF_TOKEN="{{ secrets.HF_TOKEN }}"
export MILABENCH_HF_TOKEN=$HF_TOKEN

# Set it AFTER exporting the secrets :)
set -ex

# Fix
export PATH="$HOME/.bin/:$PATH"

# ===
OUTPUT_DIRECTORY=$(scontrol show job "$SLURM_JOB_ID" --json | jq -r '.jobs[0].standard_output' | xargs dirname)
export JR_JOB_ID=$(basename "$OUTPUT_DIRECTORY")
mkdir -p $OUTPUT_DIRECTORY/meta
scontrol show job --json $SLURM_JOB_ID | jq '.jobs[0]' > $OUTPUT_DIRECTORY/meta/info.json
touch $SLURM_SUBMIT_DIR/.no_report
# ===

export UV=$HOME/.local/bin/uv

export MILABENCH_SHARED="$HOME/scratch/shared"
export MILABENCH_WORDIR="/tmp/$SLURM_JOB_ID/$MILABENCH_GPU_ARCH"
export MILABENCH_ENV="$MILABENCH_WORDIR/.env/$PYTHON_VERSION/"
export MILABENCH_BASE="$MILABENCH_WORDIR/results"
export MILABENCH_SIZER_SAVE="$MILABENCH_WORDIR/results/runs/scaling.yaml"
export BENCHMARK_VENV="$MILABENCH_WORDIR/results/venv/torch"
export MILABENCH_SOURCE="$MILABENCH_WORDIR/milabench"
export MILABENCH_CONFIG="$MILABENCH_WORDIR/milabench/config/$MILABENCH_CONFIG_NAME.yaml"
export CHERRYBIN_SOURCE="$MILABENCH_WORDIR/cherrybin"
export CHERRYBIN_DB="${CHERRYBIN_DB:-$MILABENCH_SHARED/archive.db}"

mkdir -p $MILABENCH_WORDIR
cd $MILABENCH_WORDIR
git clone $MILABENCH_REPO -b $MILABENCH_BRANCH
git clone $CHERRYBIN_REPO -b $CHERRYBIN_BRANCH

$UV venv --python=$PYTHON_VERSION $MILABENCH_ENV
. $MILABENCH_ENV/bin/activate

mkdir -p $MILABENCH_WORDIR/results/runs

$UV pip install -e $MILABENCH_SOURCE[$MILABENCH_GPU_ARCH]
$UV pip install -e $CHERRYBIN_SOURCE

milabench slurm system > $MILABENCH_WORDIR/system.yaml
rm -rf $MILABENCH_WORDIR/results/venv

module load cuda/12.6.0

milabench install --system $MILABENCH_WORDIR/system.yaml --set cuda=$CUDA_VERSION torch=$PYTORCH_VERSION $MILABENCH_ARGS

if [ ! -f "$CHERRYBIN_DB" ]; then
    echo "error: shared archive not found at $CHERRYBIN_DB (run shared_cherrybin.sh's update step first)"
    exit 1
fi

PREPARE_FLAGS=(--shared "$CHERRYBIN_DB" --system "$MILABENCH_WORDIR/system.yaml")
if [ -n "$CHERRYBIN_IO_CHUNK" ]; then
    PREPARE_FLAGS+=(--io-chunk "$CHERRYBIN_IO_CHUNK")
fi

milabench cherrybin prepare "${PREPARE_FLAGS[@]}" $MILABENCH_ARGS

milabench run --system $MILABENCH_WORDIR/system.yaml $MILABENCH_ARGS $MILABENCH_DB || :

rsync -az $MILABENCH_WORDIR/results/runs $OUTPUT_DIRECTORY

# ===
scontrol show job --json $SLURM_JOB_ID | jq '.jobs[0]' > $OUTPUT_DIRECTORY/meta/info.json
# ===
