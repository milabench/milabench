#!/usr/bin/env bash
# GLM-5.3 (FP8, 755.6 GB, 141 shards) into $SCRATCH/trillium/new/glm-rl/data/slime/models/glm5/hf, on a data-transfer
# node (transfers above 10 GB, the user's decision of 2026-11-07; the login node stops a process after 1 h of CPU):
#   ssh tri-dm1 'nohup $SCRATCH/trillium/new/glm-rl/src/scripts/trillium/slime/download_glm5.sh > $SCRATCH/trillium/new/glm-rl/download-glm5.log 2>&1 &'
# First the configuration and the 20 shards that hold layers 0-9, the embeddings, the final norm and the LM head
# (106.6 GB: truncate_hf.py's input), then the rest. Resumable: hf download skips the files that are there.
set -euo pipefail
B=${SCRATCH:?}/trillium/new/glm-rl
H=$B/data/slime/models/glm5/hf
export HF_HOME=$B/cache/huggingface HF_HUB_DISABLE_PROGRESS_BARS=1
hf=(nice "$B/venv/torch/bin/hf" download zai-org/GLM-5.3 --local-dir "$H")
t0=$(date +%s)
stamp() { echo "[download-glm5] $(date '+%F %T') +$(( $(date +%s) - t0 )) s: $*"; }
stamp "start"
"${hf[@]}" --include "*.json" --include "*.jinja"
shards=$("$B/venv/torch/bin/python" -c '
import json, re, sys
wm = json.load(open(sys.argv[1]))["weight_map"]
print(" ".join(sorted({f for k, f in wm.items() if not (m := re.match(r"model\.layers\.(\d+)\.", k)) or int(m[1]) < 10})))
' "$H/model.safetensors.index.json")
stamp "$(wc -w <<< "$shards") shards for the truncated test"
"${hf[@]}" $shards
stamp "truncated-test shards done, $(du -sb "$H" | cut -f1) bytes in $H"
"${hf[@]}" --include "*.json" --include "*.jinja" --include "*.safetensors"
stamp "done, $(ls "$H"/*.safetensors | wc -l) shards, $(du -sb "$H" | cut -f1) bytes"
