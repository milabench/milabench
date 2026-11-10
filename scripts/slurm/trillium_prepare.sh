#!/bin/bash
#
# Trillium - install + prepare step of a pipeline (see scripts/pipeline/TrilliumRun.json)
#
# Runs on a data mover node (tri-dm*) over ssh, OUTSIDE of Slurm: no SLURM_JOB_ID, no GPU,
# but internet access. It builds everything the run step needs on scratch:
#   - python + venvs, milabench checkout, benchmark venvs
#   - cherrybin (the run step checks the datasets / weights out of $CHERRYBIN_DB)
#
# The environment is cached in $SCRATCH/milabench/envs/<key>, the key being the milabench and
# cherrybin commits + the settings below. A pipeline with the same key reuses it and skips the
# install (MILABENCH_FORCE_PREPARE=1 to redo it). The pipeline run ($JR_PIPELINE_ID) gets a link
# to it: $SCRATCH/milabench/pipelines/$JR_PIPELINE_ID/cuda -> envs/<key>/cuda
#
# The run step (trillium_run.sh) runs on the compute nodes which have NO internet and cannot
# write to $HOME, it only reuses this folder.
#
# KEEP IN SYNC with trillium_run.sh
export MILABENCH_BRANCH=amd_mi355
export PYTHON_VERSION=3.12
export MILABENCH_GPU_ARCH=cuda
export MILABENCH_REPO=https://github.com/milabench/milabench.git
export CHERRYBIN_REPO=https://github.com/Delaunay/cherrybin.git
export CHERRYBIN_BRANCH=main
# torch 2.15 wheels are on the cu132 index (platforms.toml defaults)
export CUDA_VERSION="132"
export PYTORCH_VERSION="2.15.0"
# the full benchmark suite (relative to the milabench checkout)
export CONFIG_PATH=config/all.yaml
# extra milabench arguments, e.g. --select <bench>,<bench> for a subset
export MILABENCH_ARGS=""
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
PIPELINE_DIR="$SCRATCH/milabench/pipelines/${JR_PIPELINE_ID:?run this script as a pipeline step}"

# Cache key of the environment: exact commits + everything that changes what gets installed
MILABENCH_SHA=$(git ls-remote $MILABENCH_REPO refs/heads/$MILABENCH_BRANCH | cut -f1)
CHERRYBIN_SHA=$(git ls-remote $CHERRYBIN_REPO refs/heads/$CHERRYBIN_BRANCH | cut -f1)
: "${MILABENCH_SHA:?could not resolve $MILABENCH_BRANCH}" "${CHERRYBIN_SHA:?could not resolve $CHERRYBIN_BRANCH}"
ENV_DESC="milabench=$MILABENCH_SHA cherrybin=$CHERRYBIN_SHA python=$PYTHON_VERSION arch=$MILABENCH_GPU_ARCH cuda=$CUDA_VERSION torch=$PYTORCH_VERSION config=$CONFIG_PATH args=$MILABENCH_ARGS cuda_archs=$TORCH_CUDA_ARCH_LIST"
ENV_KEY=$(echo "$ENV_DESC" | sha1sum | cut -c1-12)
# Keep the path short: nox (milabench's venv creation) renames a venv to a hash when its path
# is longer than 87 chars, then milabench does not find it and installs nothing (exit 0 !)
ENV_DIR="$SCRATCH/milabench/envs/$ENV_KEY"

export MILABENCH_WORDIR="$ENV_DIR/$MILABENCH_GPU_ARCH"
export MILABENCH_BASE="$MILABENCH_WORDIR/results"
export MILABENCH_ENV="$MILABENCH_WORDIR/.env/$PYTHON_VERSION/"
export BENCHMARK_VENV="$MILABENCH_WORDIR/results/venv/torch"
export MILABENCH_SOURCE="$MILABENCH_WORDIR/milabench"
export MILABENCH_CONFIG="$MILABENCH_SOURCE/$CONFIG_PATH"
export MILABENCH_SHARED="$SCRATCH/shared"
# datasets / weights already downloaded (cherrybin archive), checked out by the run step
export CHERRYBIN_DB="${CHERRYBIN_DB:-$MILABENCH_SHARED/archive.db}"
export CHERRYBIN_SOURCE="$MILABENCH_WORDIR/cherrybin"
export MILABENCH_USE_TOML_DEPS=1

if [ $(( ${#BENCHMARK_VENV} + ${#PYTHON_VERSION} + 11 )) -gt 100 ]; then
    echo "error: $BENCHMARK_VENV is too long for nox (venv would be renamed), use a shorter path" >&2
    exit 1
fi

# Everything that would land in $HOME goes to scratch: the run step cannot write to $HOME
# and it needs the python interpreter and the caches filled here
export UV_PYTHON_INSTALL_DIR="$MILABENCH_WORDIR/uv/python"
export UV_CACHE_DIR="$MILABENCH_WORDIR/uv/cache"
export XDG_CACHE_HOME="$MILABENCH_WORDIR/cache"
export HF_HOME="$MILABENCH_WORDIR/cache/huggingface"
export TORCH_HOME="$MILABENCH_WORDIR/cache/torch"

mkdir -p $MILABENCH_WORDIR $MILABENCH_BASE/runs $PIPELINE_DIR
echo "$ENV_DESC" > $ENV_DIR/key.txt
# the run step finds the environment through the pipeline folder
ln -sfn $MILABENCH_WORDIR $PIPELINE_DIR/$MILABENCH_GPU_ARCH
echo "$ENV_DIR" > $OUTPUT_DIRECTORY/meta/env_dir.txt
cd $MILABENCH_WORDIR

PIN_GLOB="$MILABENCH_SOURCE/.pin/constraints.cuda$CUDA_VERSION.torch${PYTORCH_VERSION//./}*.txt"

# Two pipelines preparing the same environment: the second one waits, then reuses it
exec 9> $ENV_DIR/.lock
flock -w 43200 9 || echo "warning: could not lock $ENV_DIR/.lock"

# uv
export UV=$HOME/.local/bin/uv
if [ ! -x "$UV" ]; then
    export UV="$MILABENCH_WORDIR/bin/uv"
    [ -x "$UV" ] || curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$MILABENCH_WORDIR/bin" UV_NO_MODIFY_PATH=1 sh
fi

# data mover nodes are shared, do not take every core
export MAX_JOBS=${MAX_JOBS:-16}

# Fix-ups of the installed environment, also applied to an environment already prepared
# (they are not part of the cache key). Idempotent.
#
# xformers: PyPI has the same version as the milabench wheel (0.0.35) and wins the resolution,
# but it is built for another torch (no C++/CUDA ops, no Blackwell): replace it with the
# milabench wheel built for this torch / cuda.
XFORMERS_FIND_LINKS="https://github.com/milabench/wheels/releases/expanded_assets/torch${PYTORCH_VERSION%.*}-cu${CUDA_VERSION}"
xformers_ok() {
    $BENCHMARK_VENV/bin/python - <<'EOF'
import importlib.util, json, pathlib, sys
spec = importlib.util.find_spec("xformers")
if spec is None:
    sys.exit(0)
import torch
info = json.loads((pathlib.Path(spec.origin).parent / "cpp_lib.json").read_text())
print(f"xformers built for torch {info['version']['torch']}, cuda archs {info['env'].get('TORCH_CUDA_ARCH_LIST')}; torch is {torch.__version__}")
sys.exit(0 if info["version"]["torch"] == torch.__version__ else 1)
EOF
}
fix_xformers() {
    xformers_ok && return 0
    echo "xformers is built for another torch, installing the wheel from $XFORMERS_FIND_LINKS"
    $UV pip install --python $BENCHMARK_VENV/bin/python --no-deps --reinstall \
        --no-index --find-links "$XFORMERS_FIND_LINKS" xformers
    xformers_ok
}

post_install() {
    fix_xformers || { echo "error: xformers is still not built for this torch" >&2; return 1; }
}

if [ -f "$MILABENCH_WORDIR/.prepared" ] && [ -z "$MILABENCH_FORCE_PREPARE" ]; then
    echo "environment already prepared, reusing $ENV_DIR"
    cat $MILABENCH_WORDIR/.prepared
    cp $PIN_GLOB $OUTPUT_DIRECTORY/meta/ || true
    post_install
    exit $?
fi
rm -f $MILABENCH_WORDIR/.prepared

# milabench, at the commit of the cache key (a failed prepare resumes in the same folder)
if [ -d "$MILABENCH_SOURCE/.git" ]; then
    git -C $MILABENCH_SOURCE fetch origin $MILABENCH_BRANCH
else
    git clone $MILABENCH_REPO -b $MILABENCH_BRANCH $MILABENCH_SOURCE
fi
git -C $MILABENCH_SOURCE checkout -B $MILABENCH_BRANCH $MILABENCH_SHA
git -C $MILABENCH_SOURCE log -1 --pretty=format:"%H %s%n" | tee $OUTPUT_DIRECTORY/meta/milabench_commit.txt

if [ -d "$CHERRYBIN_SOURCE/.git" ]; then
    git -C $CHERRYBIN_SOURCE fetch origin $CHERRYBIN_BRANCH
else
    git clone $CHERRYBIN_REPO -b $CHERRYBIN_BRANCH $CHERRYBIN_SOURCE
fi
git -C $CHERRYBIN_SOURCE checkout -B $CHERRYBIN_BRANCH $CHERRYBIN_SHA

$UV venv --allow-existing --python=$PYTHON_VERSION $MILABENCH_ENV
. $MILABENCH_ENV/bin/activate
$UV pip install -e $MILABENCH_SOURCE[$MILABENCH_GPU_ARCH]
$UV pip install -e $CHERRYBIN_SOURCE

# CUDA toolkit for the builds that compile kernels, if the module system has one
module load cuda 2>/dev/null || echo "no cuda module, builds use the pip CUDA toolkit"

# Single node, no GPU: only used for install/prepare, the run step makes its own on the GPU node
milabench slurm system > $MILABENCH_WORDIR/system.prepare.yaml

# No pin file for this combination yet (.pin/constraints.cuda<ver>.torch<ver>.txt): generate it
# for the selected benchmarks, otherwise install stops with MissingPinError.
# Saved with the job so it can be committed to the repo.
# (with or without the arch suffix: constraints.cuda132.torch2150[.x86_64].txt)
# The pin only covers the benchmarks selected by the config + $MILABENCH_ARGS
if ! compgen -G "$PIN_GLOB" > /dev/null; then
    milabench tools pin --config $MILABENCH_CONFIG --set cuda=$CUDA_VERSION torch=$PYTORCH_VERSION $MILABENCH_ARGS
fi
# `milabench tools pin` exits 0 even when the pin fails: check its output
if ! compgen -G "$PIN_GLOB" > /dev/null; then
    echo "error: no pin file for cuda=$CUDA_VERSION torch=$PYTORCH_VERSION, the pin failed (see log.stdout)" >&2
    exit 1
fi
cp $PIN_GLOB $OUTPUT_DIRECTORY/meta/

# Git dependencies (torchtune, ...) cannot be built in the uv cache on scratch: setuptools fails
# with "[Errno 39] Directory not empty" on NFS (.nfs* files). They are built once in a uv cache on
# the data mover's local RAM disk (kept between pipeline runs) and installed into the venv from
# their git URL: milabench install then sees them as already satisfied and does not rebuild them.
# (a plain wheel would not work: uv reinstalls a `pkg @ git+...` requirement installed from a file)
GIT_UV_CACHE="/dev/shm/$USER/uv-git"
install_git_deps() {
    cat $PIN_GLOB | grep -E '^[A-Za-z0-9_.-]+ @ git\+' | sort -u | while read -r req; do
        echo "building $req in $GIT_UV_CACHE"
        UV_CACHE_DIR=$GIT_UV_CACHE $UV pip install --link-mode=copy --python $BENCHMARK_VENV/bin/python --no-deps "$req"
    done
}

# A failure here fails the step: the run jobs are cancelled instead of running on a broken env.
# The first pass creates the venv and installs everything else, the git dependencies fail to build;
# they are then installed from the local cache and the second pass completes the install.
INSTALL=(milabench install --config $MILABENCH_CONFIG --set cuda=$CUDA_VERSION torch=$PYTORCH_VERSION --system $MILABENCH_WORDIR/system.prepare.yaml $MILABENCH_ARGS)
if ! "${INSTALL[@]}"; then
    echo "install failed, installing the git dependencies from a local cache and retrying"
    install_git_deps
    "${INSTALL[@]}"
fi
# milabench install can exit 0 when a benchmark's install crashed: check the venv is there
if ! $BENCHMARK_VENV/bin/python -c "import torch, benchmate, voir"; then
    echo "error: $BENCHMARK_VENV is incomplete, the install failed (see log.stderr)" >&2
    exit 1
fi
post_install

# No `milabench prepare` here: the run step checks the datasets / weights out of the cherrybin
# archive onto each compute node (cherrybin is installed in the venv above)
if [ ! -f "$CHERRYBIN_DB" ]; then
    echo "error: cherrybin archive not found at $CHERRYBIN_DB" >&2
    exit 1
fi

# Tells the run step (and the next prepares) the environment is complete
{
    echo "env=$ENV_DIR"
    echo "milabench=$MILABENCH_SHA"
    echo "cherrybin=$CHERRYBIN_SHA"
    echo "pipeline=$JR_PIPELINE_ID"
    echo "prepare_job=$JR_JOB_ID"
    echo "host=$(hostname)"
    echo "time=$(date -Is)"
} > $MILABENCH_WORDIR/.prepared
