#!/bin/bash


export MILABENCH_BRANCH=main
export PYTHON_VERSION=3.12
export MILABENCH_GPU_ARCH=cuda
export PYTHONUNBUFFERED=0
export MILABENCH_ARGS=""
export MILABENCH_CONFIG_NAME=all
export MILABENCH_REPO=git@github.com:milabench/milabench.git

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

export MILABENCH_WORDIR="/tmp/$SLURM_JOB_ID/$MILABENCH_GPU_ARCH"  
export MILABENCH_ENV="$MILABENCH_WORDIR/.env/$PYTHON_VERSION/"
export MILABENCH_BASE="$MILABENCH_WORDIR/results"
export MILABENCH_SOURCE="$MILABENCH_WORDIR/milabench"
export MILABENCH_CONFIG="$MILABENCH_WORDIR/milabench/config/$MILABENCH_CONFIG_NAME.yaml"


mkdir -p $MILABENCH_WORDIR
cd $MILABENCH_WORDIR
git clone $MILABENCH_REPO -b $MILABENCH_BRANCH

$UV venv --python=$PYTHON_VERSION $MILABENCH_ENV
. $MILABENCH_ENV/bin/activate

module load cuda/12.6.0
mkdir -p $MILABENCH_WORDIR/results/runs

$UV pip install -e $MILABENCH_SOURCE[$MILABENCH_GPU_ARCH]
$UV pip install torch

export MILABENCH_USE_TOML_DEPS=1 
milabench tools pin --from-scratch

(
    cd $MILABENCH_SOURCE
    git checkout -b "update_pins_${SLURM_JOB_ID}"
    git add --all
    git commit -m "Pin Dependencies"
    git push origin "update_pins_${SLURM_JOB_ID}"
)

# ===
scontrol show job --json $SLURM_JOB_ID | jq '.jobs[0]' > $OUTPUT_DIRECTORY/meta/info.json
# ===