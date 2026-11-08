#!/bin/bash
# GLM-FP4 venv, login-node part: the downloads (TE and torch_memory_saver are built on a B200 node, te_build.sbatch).
# The campaign's steps ($SCRATCH/trillium/glm/install-login.sh, scripts/trillium/glm/README.md "Install on Trillium") with
# their own uv cache, copy mode (the sglang patch must not reach shared files) and the campaign venv's versions as
# constraints: pins-campaign.txt, name==version of every dist-info directory in $SCRATCH/trillium/glm/venv/torch (read
# only), except benchmate (editable, this checkout), megatron-core and slime (git commits, requirements.toml).
# Needs $G/src (git clone -b trillium-new) and $G/src/.venv (milabench), $G/wheelhouse (the campaign's TE files).
unset PYTHONPATH PIP_CONFIG_FILE
G=$SCRATCH/trillium/new/glm-fp4
ls -d $SCRATCH/trillium/glm/venv/torch/lib/python3.12/site-packages/*.dist-info | xargs -n1 basename | sed 's/\.dist-info$//' \
    | grep -v -E '^(benchmate|megatron_core|slime)-' | sed -E 's/^([^-]+)-(.*)$/\1==\2/' > $G/pins-campaign.txt
export PATH=$G/src/.venv/bin:$HOME/.local/bin:$PATH
export UV_CACHE_DIR=$G/uv-cache UV_LINK_MODE=copy UV_CONSTRAINT=$G/pins-campaign.txt
export MILABENCH_BASE=$G MILABENCH_GPU_ARCH=cuda
cd $G/src
export SLIME_DOWNLOAD_ONLY=1
nice -n 10 milabench install --config benchmarks/megatron/dev.yaml --select megatron-glm5-tri-test-n2-gbs16; echo "install rc=$?"
# milabench gives the requirements install its own -c (platform pins), which replaces UV_CONSTRAINT there: 13 packages
# released since the campaign install came in newer (2026-11-08). Put every package at the campaign version (TE: te_build.sbatch)
grep -v '^transformer_engine' $G/pins-campaign.txt > $G/pins-campaign-no-te.txt
nice -n 10 uv pip install --python $G/venv/torch/bin/python --no-deps --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://flashinfer.ai/whl/cu130 \
    -r $G/pins-campaign-no-te.txt; echo "pins rc=$?"
# TE's dependencies, installed by hand in the campaign venv
nice -n 10 uv pip install --python $G/venv/torch/bin/python onnxscript onnx nvdlfw-inspect nvidia-cudnn-frontend importlib-metadata; echo "te deps rc=$?"
