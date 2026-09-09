#!/bin/bash
#
# Test the "checkout" path: pull all benchmarks' data/checkpoints from the
# network-hosted cherrybin archive ($MILABENCH_SHARED/archive.db) down to
# node-local /tmp storage, and measure how long it takes / how much lands
# on disk. This is a read-only drill against the shared archive built by
# shared_cherrybin.sh (its "update" step) -- it does not modify it.
#
# Example submission:
#   sbatch --job-name=checkout-test --time=01:00:00 --cpus-per-task=8 \
#       --mem=32G --output=logs/%j/log.out test_cherrybin_checkout.sh

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

# Stream I/O chunk in bytes; unset keeps cherrybin's default (4MiB).
export CHERRYBIN_IO_CHUNK=""

set -ex

export PATH="$HOME/.bin/:$PATH"

# ===
OUTPUT_DIRECTORY=$(scontrol show job "$SLURM_JOB_ID" --json | jq -r '.jobs[0].standard_output' | xargs dirname)
export JR_JOB_ID=$(basename "$OUTPUT_DIRECTORY")
mkdir -p $OUTPUT_DIRECTORY/meta
scontrol show job --json $SLURM_JOB_ID | jq '.jobs[0]' > $OUTPUT_DIRECTORY/meta/info.json
touch $SLURM_SUBMIT_DIR/.no_report
# ===

export UV=$HOME/.local/bin/uv

# Network archive to checkout FROM (built ahead of time by shared_cherrybin.sh).
export MILABENCH_SHARED="$HOME/scratch/shared"
export CHERRYBIN_DB="${CHERRYBIN_DB:-$MILABENCH_SHARED/archive.db}"

# Node-local destination we are checking out TO -- /tmp is local disk on the
# compute node, as opposed to $MILABENCH_SHARED which sits on the network fs.
export MILABENCH_WORDIR="/tmp/$SLURM_JOB_ID/$MILABENCH_GPU_ARCH"
export MILABENCH_ENV="$MILABENCH_WORDIR/.env/$PYTHON_VERSION/"
export MILABENCH_BASE="$MILABENCH_WORDIR/results"
export BENCHMARK_VENV="$MILABENCH_WORDIR/results/venv/torch"
export MILABENCH_SOURCE="$MILABENCH_WORDIR/milabench"
export MILABENCH_CONFIG="$MILABENCH_WORDIR/milabench/config/$MILABENCH_CONFIG_NAME.yaml"
export CHERRYBIN_SOURCE="$MILABENCH_WORDIR/cherrybin"

if [ ! -f "$CHERRYBIN_DB" ]; then
    echo "error: shared archive not found at $CHERRYBIN_DB (run shared_cherrybin.sh's update step first)"
    exit 1
fi

mkdir -p $MILABENCH_WORDIR
cd $MILABENCH_WORDIR
git clone $MILABENCH_REPO -b $MILABENCH_BRANCH
git clone $CHERRYBIN_REPO -b $CHERRYBIN_BRANCH

$UV venv --python=$PYTHON_VERSION $MILABENCH_ENV
. $MILABENCH_ENV/bin/activate

mkdir -p $MILABENCH_WORDIR/results/runs

$UV pip install -e $MILABENCH_SOURCE[$MILABENCH_GPU_ARCH]
$UV pip install -e $CHERRYBIN_SOURCE

module load cuda/12.6.0

milabench slurm system > $MILABENCH_WORDIR/system.yaml

milabench install --system $MILABENCH_WORDIR/system.yaml --set cuda=$CUDA_VERSION torch=$PYTORCH_VERSION $MILABENCH_ARGS

CHECKOUT_FLAGS=(--shared "$CHERRYBIN_DB" --system "$MILABENCH_WORDIR/system.yaml")
if [ -n "$CHERRYBIN_IO_CHUNK" ]; then
    CHECKOUT_FLAGS+=(--io-chunk "$CHERRYBIN_IO_CHUNK")
fi

# ---- the actual test: checkout ALL benchmarks from the network archive
# ---- into node-local storage, timed end to end.
CHECKOUT_LOG="$MILABENCH_WORDIR/results/runs/checkout.log"
CHECKOUT_STATS="$OUTPUT_DIRECTORY/meta/checkout_stats.json"

CHECKOUT_START=$(date +%s)
CHECKOUT_RC=0
milabench cherrybin prepare "${CHECKOUT_FLAGS[@]}" $MILABENCH_ARGS "$@" 2>&1 | tee "$CHECKOUT_LOG" || CHECKOUT_RC=$?
CHECKOUT_END=$(date +%s)
CHECKOUT_SECONDS=$((CHECKOUT_END - CHECKOUT_START))

DATA_BYTES=$(du -sb "$MILABENCH_BASE/data" 2>/dev/null | cut -f1)
CACHE_BYTES=$(du -sb "$MILABENCH_BASE/cache" 2>/dev/null | cut -f1)

jq -n \
    --arg source_db "$CHERRYBIN_DB" \
    --arg dest "$MILABENCH_BASE" \
    --argjson seconds "$CHECKOUT_SECONDS" \
    --argjson rc "$CHECKOUT_RC" \
    --argjson data_bytes "${DATA_BYTES:-0}" \
    --argjson cache_bytes "${CACHE_BYTES:-0}" \
    '{source_db: $source_db, dest: $dest, seconds: $seconds, exit_code: $rc, data_bytes: $data_bytes, cache_bytes: $cache_bytes}' \
    > "$CHECKOUT_STATS"

cat "$CHECKOUT_STATS"

rsync -az $MILABENCH_WORDIR/results/runs $OUTPUT_DIRECTORY

# ===
scontrol show job --json $SLURM_JOB_ID | jq '.jobs[0]' > $OUTPUT_DIRECTORY/meta/info.json
# ===

exit $CHECKOUT_RC
