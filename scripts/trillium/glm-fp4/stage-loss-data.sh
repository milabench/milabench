#!/bin/bash
# GLM-FP4, Qwen3 32B C-loss data: 600,000 FineWeb documents (sample-10BT) into $G/data/fineweb, and their Arrow cache in
# $G/cache (the jobs' XDG_CACHE_HOME), with the venv's datasets. What prepare.py does for the FineWeb part, run on a
# data-transfer node (internet, $SCRATCH; not the login node): ssh tri-dm1 bash -l $G/src2/scripts/trillium/glm-fp4/stage-loss-data.sh
# Qwen3's tokenizer is already in $G/data/megatron/qwen3-32b (milabench prepare of a Qwen3 entry).
unset PYTHONPATH PIP_CONFIG_FILE
G=$SCRATCH/trillium/new/glm-fp4
export XDG_CACHE_HOME=$G/cache
nice -n 10 $G/venv/torch/bin/python -c "
from pathlib import Path
from benchmate.fineweb import prepare_fineweb
print(prepare_fineweb(Path('$G/data/fineweb/sample-10BT'), 600000, subset='sample-10BT'))
"
echo "stage rc=$?"
