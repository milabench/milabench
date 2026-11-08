#!/usr/bin/env bash
# login node, after milabench install: FlashInfer 0.7.0.post1 kernels for the offline compute nodes (what
# `vllm download-kernels` installs: the matching flashinfer-cubin and flashinfer-jit-cache wheels). flashinfer.ai's wheels
# are GitHub release assets, 0.2 MB/s from here on 2026-11-07: the cubins from Compute Canada's wheelhouse (same
# release), the sm100a-only jit-cache (273 MB) from upstream. milabench points the vLLM server at
# FLASHINFER_CUBIN_DIR=$B/cache/flashinfer/cubins (milabench/commands/serving.py): link it to the cubin wheel's directory
set -ex
B=${MILABENCH_BASE:-$SCRATCH/trillium/new/k3-inf}
unset PYTHONPATH PIP_CONFIG_FILE
export PATH=$B/venv/torch/bin:$PATH
cd /tmp
python -m pip install --no-deps /cvmfs/soft.computecanada.ca/custom/python/wheelhouse/generic/flashinfer_cubin-0.7.0.post1+computecanada-py3-none-any.whl
# the jit-cache shim (4.5 kB, lists the providers) and the sm100a provider
python -m pip install --no-deps "https://github.com/flashinfer-ai/flashinfer/releases/download/v0.7.0.post1/flashinfer_jit_cache-0.7.0.post1+cu130-cp39-abi3-manylinux_2_28_x86_64.whl#sha256=9630eeb32396adb89ce8ffa8465fedd14549eb2e2ad3aff182d9e69be4a12075" \
    "https://github.com/flashinfer-ai/flashinfer/releases/download/v0.7.0.post1/flashinfer_jit_cache_sm100a-0.7.0.post1+cu130-cp39-abi3-manylinux_2_28_x86_64.whl#sha256=3654aad2bbaebf7ede45590e5fc241035fec92219e505ea2d2d8b77b0d88ee91"
mkdir -p $B/cache/flashinfer
ln -sfn "$(python -c 'import flashinfer_cubin; print(flashinfer_cubin.get_cubin_dir())')" $B/cache/flashinfer/cubins
FLASHINFER_CUBIN_DIR=$B/cache/flashinfer/cubins python -m flashinfer show-config | head -40
