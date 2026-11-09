#!/usr/bin/env bash
# Login-node part of the slime install on Trillium (downloads only: compute nodes have no internet), all under
# $SCRATCH/trillium/new/glm-rl: the milabench CLI venv, the pack's venv without TransformerEngine / torch_memory_saver
# (the first job builds them on a GPU node, job.sbatch), TE's files (wheelhouse), torch_memory_saver's git source (uv
# cache), the entry's HF checkpoint and dataset (no conversion: GPUs).
#   git clone -b trillium-new https://github.com/milabench/milabench.git $SCRATCH/trillium/new/glm-rl/src
#   $SCRATCH/trillium/new/glm-rl/src/scripts/trillium/slime/install.sh [ENTRY]    (default slime-qwen3-30b-a3b-grpo)
set -euo pipefail
E=${1:-slime-qwen3-30b-a3b-grpo}
B=${SCRATCH:?}/trillium/new/glm-rl
C=benchmarks/slime/dev.yaml
unset PYTHONPATH PIP_CONFIG_FILE
# own uv cache, files copied: patch_sglang rewrites sglang's installed files, which with uv's default hardlinks are the
# cache's too (the campaign's venv/torch shares its inodes with $SCRATCH/uv-cache, whose sglang 0.5.21 is patched)
export UV_CACHE_DIR=$B/uv-cache UV_LINK_MODE=copy MILABENCH_BASE=$B MILABENCH_GPU_ARCH=cuda
# 30 s (the default) timed out on torch's 400 MB nvidia-cublas beside a checkpoint download (2026-11-07)
export UV_HTTP_TIMEOUT=300
export PATH=$B/src/.venv/bin:$HOME/.local/bin:$PATH
cd "$B/src"
if [ ! -x .venv/bin/milabench ]; then
    uv venv --python-preference only-managed --python 3.12 --seed .venv
    VIRTUAL_ENV=$B/src/.venv uv pip install -e . pybind11 nvidia-ml-py
fi
SLIME_DOWNLOAD_ONLY=1 nice -n 10 milabench install --config $C --select "$E"

# TransformerEngine 2.18.0 (benchfile.TE_REQUIREMENT), found offline through UV_FIND_LINKS
mkdir -p "$B/wheelhouse"
for p in transformer-engine transformer-engine-cu13 transformer-engine-torch; do
    curl -sSf "https://pypi.org/pypi/$p/2.18.0/json" | python3 -c 'import json, sys
for u in json.load(sys.stdin)["urls"]:
    if u["filename"].endswith(("-none-any.whl", "manylinux_2_28_x86_64.whl", ".tar.gz")): print(u["url"])' \
        | while read -r url; do [ -f "$B/wheelhouse/${url##*/}" ] || curl -sSfL -o "$B/wheelhouse/${url##*/}" "$url"; done
done
# and TE's own dependencies (the wheels' metadata): offline, uv finds nothing but the wheelhouse
uv pip install --python "$B/venv/torch/bin/python" pydantic importlib-metadata einops onnxscript onnx nvdlfw-inspect \
    "nvidia-cudnn-frontend>=1.25.0"
# slime's sglang-router fork, as slime's Dockerfile installs it (--no-deps): PyPI's 0.3.2 drops the routed experts that
# --use-rollout-routing-replay needs (job 1261523). This workload's venv only, pinned by sha256 (the user's approval,
# 2026-11-07; README.md)
router=$B/wheelhouse/sglang_router-0.3.2-cp38-abi3-manylinux_2_28_x86_64.whl
[ -f "$router" ] || curl -sSfL -o "$router" "https://github.com/zhuzilin/sgl-router/releases/download/v0.3.2-9daabcd/${router##*/}"
echo "c26d31b9decd4bef04c5c2bf72470e7660c2802b9599b885eb7c1dfbfbd9a4b5  $router" | sha256sum -c
uv pip install --python "$B/venv/torch/bin/python" --no-deps --reinstall-package sglang-router "$router"
"$B/venv/torch/bin/python" -c 'import sglang_router; assert "slime" in sglang_router.__version__, sglang_router.__version__'
# FlashInfer's cubins (B200's trtllm attention, MoE and FP8 GEMM kernels, which the nodes cannot download) at sglang
# 0.5.21's flashinfer_python==0.6.18, for the *-trtllm entries (the user's choice, 2026-11-09; README.md). 1.57 GB from
# GitHub, 0.2 MB/s from the login node (2026-11-07): curl it into the wheelhouse on tri-dm2 first
cubin=$B/wheelhouse/flashinfer_cubin-0.6.18-py3-none-any.whl
[ -f "$cubin" ] || curl -sSfL -o "$cubin" "https://github.com/flashinfer-ai/flashinfer/releases/download/v0.6.18/${cubin##*/}"
echo "2dd65c0fcfc6bc44c67f148530de5372979c2e3d260e47935730f94156d4d873  $cubin" | sha256sum -c
uv pip install --python "$B/venv/torch/bin/python" --no-deps "$cubin"
# torch_memory_saver (colocated GRPO offloads the trainer with it): offline, uv takes a pinned git commit from its
# cache. The dry run fetches it there (its build needs a GPU node: a failure here is expected)
tms=$(sed -n 's/^TMS_REQUIREMENT = "\(.*\)"$/\1/p' benchmarks/slime/benchfile.py)
uv pip install --dry-run --no-build-isolation --python "$B/venv/torch/bin/python" "$tms" || true

SLIME_DOWNLOAD_ONLY=1 nice -n 10 milabench prepare --config $C --select "$E"
