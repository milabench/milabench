#!/bin/bash
#
# Trillium - run step of a pipeline (see scripts/pipeline/TrilliumRunBS.json)
#
# Same as trillium_run.sh, but the batch sizes are picked by the sizer (sizer.auto) to fill the
# B200 memory, and the observations are saved to build config/scaling/B200.yaml:
#   - predictions come from config/scaling/B200.yaml (default batch runs + the first resize run,
#     with OOM markers: the sizer bisects below a batch size that ran out of memory)
#   - new observations (and OOM markers) are added to a copy of it
# The result is in $OUTPUT_DIRECTORY/meta/B200.yaml (also $JOB_CACHE/B200.yaml): copy it to
# config/scaling/B200.yaml in the repo.
#
# Runs on the GPU compute nodes: NO internet access, $HOME is read only, no local disk.
# python, venvs and milabench were built on scratch by trillium_prepare.sh on a data mover
# node, in a cached environment linked from the pipeline run folder ($JR_PIPELINE_ID).
# Datasets / weights are checked out of the cherrybin archive onto each node (RAM disk).
# Nothing is downloaded or installed here.
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
# the full benchmark suite (relative to the milabench checkout)
export CONFIG_PATH=config/all.yaml
# extra milabench arguments, e.g. --select <bench>,<bench> for a subset
export MILABENCH_ARGS=""
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
# pipelines/<id>/cuda is a link to the cached environment (envs/<key>/cuda) made by the prepare step
export MILABENCH_WORDIR="$(readlink -f "$SCRATCH/milabench/pipelines/${JR_PIPELINE_ID:?run this script as a pipeline step}/$MILABENCH_GPU_ARCH")"
export MILABENCH_BASE="$MILABENCH_WORDIR/results"
export MILABENCH_ENV="$MILABENCH_WORDIR/.env/$PYTHON_VERSION/"
export BENCHMARK_VENV="$MILABENCH_WORDIR/results/venv/torch"
export MILABENCH_SOURCE="$MILABENCH_WORDIR/milabench"
export MILABENCH_CONFIG="$MILABENCH_SOURCE/$CONFIG_PATH"
export MILABENCH_SHARED="$SCRATCH/shared"
export CHERRYBIN_DB="${CHERRYBIN_DB:-$MILABENCH_SHARED/archive.db}"
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
# Copy of the job's output on scratch (the dashboard cleans up the job folder), unbuffered so
# nothing is lost if the job is killed
export PYTHONUNBUFFERED=1
exec > >(tee -a $JOB_CACHE/job.stdout) 2> >(tee -a $JOB_CACHE/job.stderr >&2)
export XDG_CACHE_HOME="$JOB_CACHE/cache"
export TRITON_CACHE_DIR="$JOB_CACHE/triton"
export TORCHINDUCTOR_CACHE_DIR="$JOB_CACHE/inductor"
# Batch size: predicted from the B200 observations of the milabench checkout, scaled to this
# GPU's capacity (system.yaml). Benchmarks with a single B200 batch size and no OOM marker keep
# their default batch size. Each job writes its own copy, seeded from the checkout's file, so
# observations accumulate across runs: copy it back to config/scaling/B200.yaml.
# (the first resize run used MI355X.yaml: it had no B200 data)
export MILABENCH_SIZER_AUTO=1
# predictions: B200 curves, with the H100 curve for the benches that have a single B200 point
# (scripts/b200_h100_scaling.py). Observations are still saved to B200.yaml (SAVE below)
export MILABENCH_SIZER_CONFIG="$MILABENCH_SOURCE/config/scaling/B200-H100.yaml"
export MILABENCH_SIZER_SAVE="$JOB_CACHE/B200.yaml"
# Reserve 10% of the B200 memory instead of the measured fixed overhead (NVML - torchmem):
# the first resize run ran out of memory on 5 benchmarks with only that margin
export MILABENCH_SIZER_OVERHEAD_OVERRIDE="10%"
if [ -s "$MILABENCH_SOURCE/config/scaling/B200.yaml" ]; then
    cp "$MILABENCH_SOURCE/config/scaling/B200.yaml" "$MILABENCH_SIZER_SAVE"
fi
# no local disk: $SLURM_TMPDIR is a RAM disk (/dev/shm, 1.5T on the B200 nodes)
export TMPDIR="${SLURM_TMPDIR:-$JOB_CACHE/tmp}"
mkdir -p $TMPDIR

# No internet: fail fast instead of hanging on a download
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export UV_OFFLINE=1
export PIP_NO_INDEX=1
export WANDB_MODE=offline
# The jobrunner prologue sets a SOCKS proxy to the login node: huggingface_hub builds its http
# client from it before checking offline mode and fails (socksio is not installed). Offline anyway.
unset ALL_PROXY all_proxy HTTPS_PROXY https_proxy HTTP_PROXY http_proxy
# SDPA prefers cuDNN attention on B200 (torch 2.15), which builds a plan for every new sequence
# length: ~1.6 s each (GPU idle, llm-lora at 1/4 of H100). Put it last: flash / memory efficient
# attention are used, cuDNN only for inputs they do not support. Read by torch in every process.
export TORCH_CUDNN_SDPA_DEPRIORITIZED=1
# Report: drop the benchmarks that did not run (--select subset) instead of listing them as empty rows
export MILABENCH_REPORT_LEAN=1

. $MILABENCH_ENV/bin/activate

module load cuda 2>/dev/null || echo "no cuda module"

# ffmpeg for torchcodec (whisper-transcribe). The ffmpeg modules (CVMFS 2023 stack) need glibc 2.35+,
# our python runs on the host glibc (2.34): use ffmpeg 4 of the 2020 stack (glibc 2.30) instead.
# Only ffmpeg and the dependencies the host does not have are linked into a folder put on
# LD_LIBRARY_PATH, so no host library (libstdc++, zlib, ...) is shadowed.
setup_ffmpeg_shim() {
    local shim="$1" src=/cvmfs/soft.computecanada.ca/gentoo/2020/usr/lib64 l m missing i
    mkdir -p "$shim"
    for l in libavutil.so.56 libavcodec.so.58 libavformat.so.58 libavdevice.so.58 libavfilter.so.7 libswscale.so.5 libswresample.so.3; do
        ln -sfn "$src/$l" "$shim/$l"
    done
    for i in $(seq 15); do
        missing=$(LD_LIBRARY_PATH="$shim" /usr/bin/ldd "$shim"/*.so* 2>/dev/null | awk '$2 == "=>" && $3 == "not" {print $1}' | sort -u)
        [ -z "$missing" ] && return 0
        for m in $missing; do
            if [ -e "$src/$m" ]; then ln -sfn "$src/$m" "$shim/$m"; else echo "ffmpeg shim: $m not found"; return 1; fi
        done
    done
}
if setup_ffmpeg_shim "$JOB_CACHE/ffmpeg"; then
    export LD_LIBRARY_PATH="$JOB_CACHE/ffmpeg${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
else
    echo "no ffmpeg for torchcodec, whisper-transcribe will fail"
fi

# On the GPU node(s) of this job
milabench slurm system > $JOB_CACHE/system.yaml

# Datasets / weights: checked out of the cherrybin archive onto every node of the job, in
# $SLURM_TMPDIR (same path on each node). all.yaml is ~580G without the vllm benchmarks, it is RAM!
# Generated datasets (FakeImageNet, ...) are built there as well (no internet needed).
LOCAL_DIR="${SLURM_TMPDIR:?}/milabench"
export MILABENCH_DIRS_DATA="$LOCAL_DIR/data"
export MILABENCH_DIRS_CACHE="$LOCAL_DIR/cache"
# its output is also kept on scratch ($JOB_CACHE), the dashboard cleans up the job folder
set -o pipefail
srun --overlap --nodes=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 \
    milabench cherrybin prepare --config $MILABENCH_CONFIG --system $JOB_CACHE/system.yaml \
        --shared $CHERRYBIN_DB --cache $LOCAL_DIR/blobs $MILABENCH_ARGS 2>&1 | tee $JOB_CACHE/cherrybin_prepare.log \
    || echo "cherrybin checkout incomplete: the benchmarks missing from the archive will fail"
du -sh $LOCAL_DIR || true

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

# B200 scaling observations of this run, next to the job's data
cp "$MILABENCH_SIZER_SAVE" $OUTPUT_DIRECTORY/meta/B200.yaml || echo "no B200 scaling data was saved"

# ===
scontrol show job --json $SLURM_JOB_ID | jq '.jobs[0]' > $OUTPUT_DIRECTORY/meta/info.json
# ===

exit $RC
