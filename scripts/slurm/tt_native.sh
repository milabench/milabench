#!/bin/bash
# Run one torchtitan_native bench under Slurm (DRAC clusters).
#
#   sbatch --nodes=2 --time=00:45:00 --account=rrg-bengioy-ad_gpu \
#       scripts/slurm/tt_native.sh tt-qwen38-n2 [extra milabench run args]
#
#   Window configs: sbatch --export=ALL,TT_CONFIG=window ... tt_native.sh tt-w-leaf3
#   Per-job environment as leading arguments: tt_native.sh TT_LOCAL_VENV=1 tt-w-leaf3
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

# Leading NAME=value arguments are exported for the job (for launchers that
# cannot pass per-job environment, e.g. cluv: `-- NCCL_DEBUG=INFO <bench>`).
while [[ "${1:-}" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; do
    export "$1"
    shift
done

BENCH=${1:?usage: tt_native.sh [NAME=value ...] <bench-name> [milabench run args]}
shift

TT38=${TT38:-$SCRATCH/tt38}
# DRAC login profile points Python at cvmfs site-packages and the DRAC wheelhouse.
unset PYTHONPATH PIP_CONFIG_FILE
export PATH=$TT38/env/bin:$HOME/.local/bin:$PATH

export MILABENCH_BASE=$TT38/output
# TT_CONFIG=window selects config/torchtitan_native_window.yaml.
if [ "${TT_CONFIG:-}" = window ]; then
    export MILABENCH_CONFIG=$TT38/milabench/config/torchtitan_native_window.yaml
else
    export MILABENCH_CONFIG=$TT38/milabench/config/torchtitan_native.yaml
fi
export MILABENCH_GPU_ARCH=cuda
export PYTHONUNBUFFERED=1

echo "[tt-native] job_start_epoch=$(date +%s.%N) job=$SLURM_JOB_ID nodes=$SLURM_JOB_NODELIST"
squeue -h -j "$SLURM_JOB_ID" -o "[tt-native] submit=%V start=%S" || true
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | sort | uniq -c | sed 's/^/[tt-native] gpu /'
ip -br addr | grep -v "^lo" | sed 's/^/[tt-native] iface /'
# InfiniBand adapters and link rates (inter-node bandwidth per node).
for ca in /sys/class/infiniband/*; do
    [ -e "$ca" ] || continue
    echo "[tt-native] hca $(basename "$ca") $(cat "$ca"/ports/1/rate 2>/dev/null) $(cat "$ca"/ports/1/link_layer 2>/dev/null)"
done

# Nodes that get the compile cache / local venv: the run's subset (TT_NODES) or
# the whole job. Explicit, because inside a job step a plain srun inherits the
# step's node count (e.g. 1 node when launched by tt_variants.sh).
STAGE_NODES=${TT_NODES:-$SLURM_JOB_NODELIST}
STAGE_N=$(scontrol show hostnames "$STAGE_NODES" | wc -l)

# Pre-warmed torch.compile caches (Inductor + Triton): unpack an archive onto
# every node's local disk (same path as benchfile.py's cache dirs; it must be
# job-independent because cache entries store absolute paths). Per-GPU shapes
# are identical at every scale, so a cache built on a few nodes is valid for
# large runs. TT_COMPILE_CACHE= (empty) disables it.
LOCAL_CACHE=${TT_NATIVE_CACHE_DIR:-/tmp/$USER/tt_native_cache}
export TT_NATIVE_CACHE_DIR=$LOCAL_CACHE
# Archive named by torch/Triton version and GPU model: a cache built with another
# stack is never loaded. (Different model configs can share one archive:
# Inductor keys each entry by graph and settings.)
# Package metadata only: importing torch from the shared filesystem took ~100 s.
CACHE_KEY=$("$TT38/output/venv/torchtitan_native/bin/python" -c "
import importlib.metadata as m
print(f'torch{m.version(\"torch\")}-triton{m.version(\"triton\")}'.replace('+', '_'))" 2>/dev/null || echo unknown)
GPU_KEY=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr -cd '[:alnum:]')
CACHE_ARCHIVE=${TT_COMPILE_CACHE-$TT38/compile-cache/$CACHE_KEY-$GPU_KEY.tar}
mkdir -p "$(dirname "$CACHE_ARCHIVE")" 2>/dev/null || true
if [ -n "$CACHE_ARCHIVE" ] && [ -f "$CACHE_ARCHIVE" ]; then
    srun --overlap --nodes="$STAGE_N" --ntasks="$STAGE_N" --ntasks-per-node=1 -w "$STAGE_NODES" bash -c "mkdir -p $LOCAL_CACHE && tar -xf $CACHE_ARCHIVE -C $LOCAL_CACHE"
    echo "[tt-native] compile_cache_restored_epoch=$(date +%s.%N) archive=$CACHE_ARCHIVE"
fi
CACHE_FILES_BEFORE=$(find "$LOCAL_CACHE" -type f 2>/dev/null | wc -l)

# TT_LOCAL_VENV=1: run the bench venv (and its Python) from node-local disk
# instead of the shared filesystem, like the window's node-local image. The
# venv + interpreter are packed once into a single archive on scratch (one
# large sequential read per node), extracted on every node, and their
# absolute paths rewritten.
if [ "${TT_LOCAL_VENV:-0}" = 1 ]; then
    SRC_VENV=$MILABENCH_BASE/venv/torchtitan_native
    PY_REAL=$(readlink -f "$SRC_VENV/bin/python")              # .../cpython-3.12.13-.../bin/python3.12
    PY_ROOT=$(dirname "$(dirname "$PY_REAL")")
    PY_LINK_ROOT=$(dirname "$(dirname "$(readlink "$SRC_VENV/bin/python")")")  # cpython-3.12-... (symlinked dir)
    LOCAL_SW=/tmp/$USER/tt_native_sw
    VENV_ARCHIVE=$TT38/venv-archive/torchtitan_native-$CACHE_KEY.tar
    mkdir -p "$(dirname "$VENV_ARCHIVE")"
    t0=$(date +%s.%N)
    # flock: concurrent jobs (e.g. one per leaf switch) must not build it twice.
    (
        flock 9
        if [ ! -f "$VENV_ARCHIVE" ]; then
            tar -cf "$VENV_ARCHIVE.tmp" -C / "${SRC_VENV#/}" "${PY_ROOT#/}" && mv "$VENV_ARCHIVE.tmp" "$VENV_ARCHIVE"
        fi
    ) 9>"$VENV_ARCHIVE.lock"
    t1=$(date +%s.%N)
    LOCAL_VENV=$LOCAL_SW$SRC_VENV
    LOCAL_PY=$LOCAL_SW$PY_ROOT
    srun --overlap --nodes="$STAGE_N" --ntasks="$STAGE_N" --ntasks-per-node=1 -w "$STAGE_NODES" bash -c "
        set -e
        rm -rf $LOCAL_SW && mkdir -p $LOCAL_SW && tar -xf $VENV_ARCHIVE -C $LOCAL_SW
        for p in $LOCAL_VENV/bin/python $LOCAL_VENV/bin/python3 $LOCAL_VENV/bin/python3.12; do
            [ -L \$p ] && ln -sfn $LOCAL_PY/bin/python3.12 \$p
        done
        sed -i -e 's|$PY_LINK_ROOT|$LOCAL_PY|g' -e 's|$PY_ROOT|$LOCAL_PY|g' -e 's|$SRC_VENV|$LOCAL_VENV|g' $LOCAL_VENV/pyvenv.cfg
        grep -rlI --include='*' '^#!$SRC_VENV' $LOCAL_VENV/bin | xargs -r sed -i '1s|^#!$SRC_VENV|#!$LOCAL_VENV|'
    "
    t2=$(date +%s.%N)
    export MILABENCH_DIRS_VENV=$LOCAL_VENV
    echo "[tt-native] local_venv pack_s=$(echo "$t1 - $t0" | bc) stage_s=$(echo "$t2 - $t1" | bc) archive_size=$(du -h "$VENV_ARCHIVE" | cut -f1) venv=$LOCAL_VENV"
    "$LOCAL_VENV/bin/python" -c "import sys, torch; print('[tt-native] local_venv check', sys.prefix, sys.base_prefix, torch.__file__)"
fi

# Startup probe: where does `milabench slurm system` spend its time?
FIRST_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
t0=$(date +%s.%N); ssh -oStrictHostKeyChecking=no -oPasswordAuthentication=no "$FIRST_NODE" cat /etc/hostname >/dev/null 2>&1 || true
t1=$(date +%s.%N); python -c "import socket,sys; socket.gethostbyaddr(sys.argv[1])" "$FIRST_NODE" 2>/dev/null || true
t2=$(date +%s.%N)
echo "[tt-native] tmp $(df -h /tmp | tail -1)"
echo "[tt-native] probe ssh_s=$(echo "$t1 - $t0" | bc) dns_s=$(echo "$t2 - $t1" | bc) node=$FIRST_NODE"

SYSTEM=$MILABENCH_BASE/system.$SLURM_JOB_ID${SLURM_STEP_ID:+.$SLURM_STEP_ID}.yaml
# TT_NODES (hostlist): run on this subset of the allocation (this script must
# then run on its first node, e.g. via `srun --overlap -N1 -n1 -w <first>`).
SLURM_JOB_NODELIST=${TT_NODES:-$SLURM_JOB_NODELIST} milabench slurm system > "$SYSTEM"
echo "[tt-native] system_ready_epoch=$(date +%s.%N)"

rc=0
milabench run --config "$MILABENCH_CONFIG" --system "$SYSTEM" --select "$BENCH" "$@" || rc=$?

# Save (or refresh) the archive when this run compiled something new, e.g. a
# config with new shapes: the main node's cache = restored entries + new ones.
CACHE_FILES_AFTER=$(find "$LOCAL_CACHE" -type f 2>/dev/null | wc -l)
if [ -n "$CACHE_ARCHIVE" ] && [ -d "$LOCAL_CACHE" ] \
        && { [ ! -f "$CACHE_ARCHIVE" ] || [ "$CACHE_FILES_AFTER" -gt "$CACHE_FILES_BEFORE" ]; }; then
    (
        flock 9
        tar -cf "$CACHE_ARCHIVE.tmp" -C "$LOCAL_CACHE" . && mv "$CACHE_ARCHIVE.tmp" "$CACHE_ARCHIVE"
    ) 9>"$CACHE_ARCHIVE.lock"
    echo "[tt-native] compile_cache_saved archive=$CACHE_ARCHIVE files=$CACHE_FILES_BEFORE->$CACHE_FILES_AFTER size=$(du -h "$CACHE_ARCHIVE" | cut -f1)"
fi
echo "[tt-native] job_end_epoch=$(date +%s.%N)"
exit $rc
