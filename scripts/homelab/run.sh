#!/usr/bin/env bash
# Run the homelab suite (config/homelab/) on this machine, with the node sampler and the memory guard.
#
#   nohup scripts/homelab/run.sh > $MILABENCH_BASE/runs/homelab.<stamp>.log 2>&1 &
#   CONFIG=coding-agent.yaml scripts/homelab/run.sh ...                     # another homelab config
#   RESUME=<run name> SELECT=a,b scripts/homelab/run.sh >> <log> 2>&1      # rerun benches into a run
#   STEPS="install prepare" scripts/homelab/run.sh                          # only some steps
#
# Per run folder ($MILABENCH_BASE/runs/<run name>):
#   *.data                 milabench's events (throughput, accuracy, GPU power/temperature per bench)
#   nodemon/               node samples every second: GPU power, energy counter, temperature, clocks,
#                          host memory used and its peak (= GPU memory on unified-memory machines), OOM kills
#   memguard.log           benchmarks stopped before they ran the machine out of memory
#
# Unified memory (DGX Spark: the GPU allocates from system RAM, NVML reports no memory): the servers'
# memory fraction is capped (MILABENCH_GPU_MEMORY_FRACTION_MAX, default 0.75) and memguard kills a
# benchmark when the machine runs out of memory, instead of letting it swap until it freezes.
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MILABENCH_SRC=${MILABENCH_SRC:-$(cd "$HERE/../.." && pwd)}
export MILABENCH_BASE=${MILABENCH_BASE:-/opt/milabench}
# milabench CLI venv: this checkout and benchmate editable, released dependencies
#   uv venv -p 3.12 $MILABENCH_BASE/venv/homelab-cli && uv pip install --python ... -e . -e ./benchmate nvidia-ml-py
MILABENCH_VENV=${MILABENCH_VENV:-$MILABENCH_BASE/venv/homelab-cli}
PY="$MILABENCH_VENV/bin/python"
MB="$MILABENCH_VENV/bin/milabench"

export MILABENCH_GPU_ARCH=${MILABENCH_GPU_ARCH:-cuda}
export MILABENCH_CONFIG=$MILABENCH_SRC/config/homelab/${CONFIG:-all.yaml}
export MILABENCH_SYSTEM=${MILABENCH_SYSTEM:-$MILABENCH_BASE/data/system.yaml}
export MILABENCH_SIZER_CONFIG=${MILABENCH_SIZER_CONFIG:-$MILABENCH_BASE/runs/scaling.yaml}
export MILABENCH_SIZER_SAVE=${MILABENCH_SIZER_SAVE:-$MILABENCH_BASE/runs/scaling.yaml}
export MILABENCH_DASH=no
export MILABENCH_NOTERM=1
export UV_HTTP_RETRIES=${UV_HTTP_RETRIES:-10}

# engines provisioned outside milabench (see config/homelab/_base.yaml)
[[ -d $MILABENCH_BASE/extra/comfyui/ComfyUI ]] && export MILABENCH_COMFY_ROOT=${MILABENCH_COMFY_ROOT:-$MILABENCH_BASE/extra/comfyui/ComfyUI}
[[ -x $MILABENCH_BASE/venv/comfyui/bin/python ]] && export MILABENCH_COMFY_PYTHON=${MILABENCH_COMFY_PYTHON:-$MILABENCH_BASE/venv/comfyui/bin/python}
[[ -d $MILABENCH_BASE/extra/llamacpp/llama.cpp ]] && export MILABENCH_LLAMACPP_ROOT=${MILABENCH_LLAMACPP_ROOT:-$MILABENCH_BASE/extra/llamacpp/llama.cpp}

# container runtime of the SWE-bench / QualityBench code tasks: podman when it answers
if [[ -z ${MILABENCH_CONTAINER:-} ]] && podman info >/dev/null 2>&1; then export MILABENCH_CONTAINER=podman; fi

# unified memory: NVML has no memory total for the GPU
if nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | grep -q "N/A"; then
    export MILABENCH_GPU_MEMORY_FRACTION_MAX=${MILABENCH_GPU_MEMORY_FRACTION_MAX:-0.75}
    # the model caches' page cache counts as used memory: dropped before each server starts
    export MILABENCH_EVICT_PAGE_CACHE=${MILABENCH_EVICT_PAGE_CACHE:-${HF_HUB_CACHE:-$MILABENCH_BASE/data/hub}:$MILABENCH_BASE/data}
    UNIFIED=1
else
    UNIFIED=0
fi

# install stack: aarch64 + CUDA has no vLLM wheel for the default torch 2.15 + cu132 (platforms.toml):
# torch 2.14 + cu130 + vLLM 0.33 from PyPI, unpinned (no lockfile for that pair; the platform constraints apply)
if [[ -z ${INSTALL_ARGS+x} ]]; then
    INSTALL_ARGS=()
    if [[ $(uname -m) == aarch64 && $MILABENCH_GPU_ARCH == cuda ]]; then
        INSTALL_ARGS=(--variant unpinned --set torch=2.14.0 cuda=130)
    fi
else
    read -r -a INSTALL_ARGS <<< "$INSTALL_ARGS"
fi

if [[ -z ${MILABENCH_HF_TOKEN:-${HF_TOKEN:-}} ]]; then
    echo "HF_TOKEN / MILABENCH_HF_TOKEN is not set (gated models: Llama, FLUX)" >&2
    exit 1
fi
export HF_TOKEN=${HF_TOKEN:-$MILABENCH_HF_TOKEN}
export MILABENCH_HF_TOKEN=${MILABENCH_HF_TOKEN:-$HF_TOKEN}

STAMP=$(date +%Y-%m-%d_%H-%M-%S)
RUN_NAME=${RESUME:-${RUN_NAME:-homelab.$STAMP}}
RUN_DIR=$MILABENCH_BASE/runs/$RUN_NAME
STEPS=${STEPS:-install prepare run report}
[[ -n ${RESUME:-} ]] && STEPS=${STEPS_RESUME:-run report}
ARGS=(--base "$MILABENCH_BASE" --config "$MILABENCH_CONFIG" --system "$MILABENCH_SYSTEM")
SEL=()
[[ -n ${SELECT:-} ]] && SEL=(--select "$SELECT")
[[ -n ${EXCLUDE:-} ]] && SEL+=(--exclude "$EXCLUDE")

step() { echo; echo "===== [$(date '+%F %T')] $* ====="; }
has() { [[ " $STEPS " == *" $1 "* ]]; }

step "homelab on $(hostname) | run=$RUN_NAME | config=$MILABENCH_CONFIG | src=$MILABENCH_SRC@$(git -C "$MILABENCH_SRC" rev-parse --short HEAD 2>/dev/null)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
echo "unified memory: $UNIFIED | memory fraction cap: ${MILABENCH_GPU_MEMORY_FRACTION_MAX:-none} | page cache eviction: ${MILABENCH_EVICT_PAGE_CACHE:-none} | containers: ${MILABENCH_CONTAINER:-auto}"
df -h "$MILABENCH_BASE" | tail -1

rc_install=- rc_prepare=- rc_run=-
if has install; then
    for try in 1 2 3; do
        step "install (try $try) ${INSTALL_ARGS[*]}"; "$MB" install "${ARGS[@]}" "${SEL[@]}" "${INSTALL_ARGS[@]}"; rc_install=$?
        [[ $rc_install == 0 ]] && break
        sleep 120
    done
fi
has prepare && { step "prepare"; "$MB" prepare "${ARGS[@]}" "${SEL[@]}"; rc_prepare=$?; }

if has run; then
    mkdir -p "$RUN_DIR"
    # the node sampler: one process for the whole run, its own folder per attempt
    NODEMON_DIR=$RUN_DIR/nodemon/$STAMP
    "$PY" -m benchmate.nodemon --out "$NODEMON_DIR" --bench "$RUN_NAME" > "$RUN_DIR/nodemon.$STAMP.log" 2>&1 &
    NODEMON=$!

    step "run ${SEL[*]}"
    "$MB" run "${ARGS[@]}" "${SEL[@]}" --run-name "$RUN_NAME" &
    MBPID=$!
    "$PY" -m benchmate.memguard --pid "$MBPID" --log "$RUN_DIR/memguard.log" > "$RUN_DIR/memguard.$STAMP.log" 2>&1 &
    GUARD=$!
    wait "$MBPID"; rc_run=$?
    kill "$GUARD" "$NODEMON" 2>/dev/null
    wait "$GUARD" "$NODEMON" 2>/dev/null
    [[ -s $RUN_DIR/memguard.log ]] && { echo "memguard events:"; grep '"kill"' "$RUN_DIR/memguard.log" | cut -c1-400; }
fi

has report && { step "report"; "$MB" report --runs "$RUN_DIR" || true; }

step "done: install=$rc_install prepare=$rc_prepare run=$rc_run -> $RUN_DIR"
echo "$rc_install $rc_prepare $rc_run" > "$MILABENCH_BASE/runs/$RUN_NAME.status"
[[ $rc_run == - ]] && exit 0 || exit $rc_run
