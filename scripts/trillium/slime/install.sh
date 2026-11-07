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
# torch_memory_saver (colocated GRPO offloads the trainer with it): offline, uv takes a pinned git commit from its
# cache. The dry run fetches it there (its build needs a GPU node: a failure here is expected)
tms=$(sed -n 's/^TMS_REQUIREMENT = "\(.*\)"$/\1/p' benchmarks/slime/benchfile.py)
uv pip install --dry-run --no-build-isolation --python "$B/venv/torch/bin/python" "$tms" || true

SLIME_DOWNLOAD_ONLY=1 nice -n 10 milabench prepare --config $C --select "$E"
