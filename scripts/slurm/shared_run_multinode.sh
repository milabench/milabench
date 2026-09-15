#!/bin/bash

export MILABENCH_BRANCH=stacc
export PYTHON_VERSION=3.12
export MILABENCH_GPU_ARCH=cuda
export PYTHONUNBUFFERED=1
export MILABENCH_ARGS=""
export CHERRYBIN_REPO=https://github.com/Delaunay/cherrybin.git
export CHERRYBIN_BRANCH=main

# Stream I/O chunk in bytes; unset keeps cherrybin's default (16MiB).
export CHERRYBIN_IO_CHUNK=""

set -ex

# ===
OUTPUT_DIRECTORY=$(scontrol show job "$SLURM_JOB_ID" --json | jq -r '.jobs[0].standard_output' | xargs dirname)
export JR_JOB_ID=$(basename "$OUTPUT_DIRECTORY")
mkdir -p $OUTPUT_DIRECTORY/meta
scontrol show job --json $SLURM_JOB_ID | jq '.jobs[0]' > $OUTPUT_DIRECTORY/meta/info.json
touch $SLURM_SUBMIT_DIR/.no_report
# ===

CONDA_EXEC="$(which conda)"
CONDA_BASE=$(dirname $CONDA_EXEC)
source $CONDA_BASE/../etc/profile.d/conda.sh

export MILABENCH_SHARED="$HOME/scratch/shared"
export MILABENCH_WORDIR="/tmp/$SLURM_JOB_ID/$MILABENCH_GPU_ARCH"  

export MILABENCH_ENV="$MILABENCH_WORDIR/.env/$PYTHON_VERSION/"
export MILABENCH_SIZER_SAVE="$MILABENCH_WORDIR/results/runs/scaling.yaml"
export MILABENCH_SYSTEM="$MILABENCH_WORDIR/results/runs/system.yaml"
export MILABENCH_BASE="$MILABENCH_WORDIR/results"
export BENCHMARK_VENV="$MILABENCH_WORDIR/results/venv/torch"
export MILABENCH_SOURCE="$MILABENCH_WORDIR/milabench"
export CHERRYBIN_SOURCE="$MILABENCH_WORDIR/cherrybin"
export CHERRYBIN_DB="${CHERRYBIN_DB:-$MILABENCH_SHARED/archive.db}"

cd /tmp
srun --ntasks-per-node=1 mkdir -p $MILABENCH_BASE

cd $MILABENCH_WORDIR
conda create --prefix $MILABENCH_ENV python=$PYTHON_VERSION -y
conda activate $MILABENCH_ENV

git clone https://github.com/mila-iqia/milabench.git -b $MILABENCH_BRANCH
git clone $CHERRYBIN_REPO -b $CHERRYBIN_BRANCH
pip install -e $MILABENCH_SOURCE[$MILABENCH_GPU_ARCH]
pip install -e $CHERRYBIN_SOURCE

pip install -e $MILABENCH_SOURCE

cd $MILABENCH_WORDIR

milabench slurm system > $MILABENCH_SYSTEM

if [ ! -f "$CHERRYBIN_DB" ]; then
    echo "error: shared archive not found at $CHERRYBIN_DB (run shared_cherrybin.sh's update step first)"
    exit 1
fi

PREPARE_CMD="$(which milabench) cherrybin prepare --shared \"$CHERRYBIN_DB\" --system \"$MILABENCH_SYSTEM\""
if [ -n "$CHERRYBIN_IO_CHUNK" ]; then
    PREPARE_CMD="$PREPARE_CMD --io-chunk $CHERRYBIN_IO_CHUNK"
fi
srun --ntasks-per-node=1 bash -c "$PREPARE_CMD $MILABENCH_ARGS"

milabench run --select multinode --system $MILABENCH_SYSTEM $MILABENCH_ARGS || :

rsync -az $MILABENCH_WORDIR/results/runs $OUTPUT_DIRECTORY

# ===
scontrol show job --json $SLURM_JOB_ID | jq '.jobs[0]' > $OUTPUT_DIRECTORY/meta/info.json
# ===