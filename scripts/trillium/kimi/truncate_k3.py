"""Kimi-K3 cut to its first N decoder layers, for bring-up tests of the real kernels in a few minutes.

    python truncate_k3.py <snapshot dir> <output dir> [N=4]

N=4 keeps layer 1 (KDA, dense MLP), 2-3 (KDA, MoE: 896 MXFP4 experts) and 4 (MLA, MoE), 1-indexed as in
linear_attn_config. The output holds symlinks to the snapshot's files (shards 1..N, the embeddings and lm_head in
shard 94, the vision tower) and its own config.json and model.safetensors.index.json. Give it the same layout as the
HF cache (<out>/hub/models--moonshotai--Kimi-K3/snapshots/<sha>, refs/main) to run a config entry on it through
MILABENCH_DIRS_DATA=<out>.
"""

import json
import os
import sys

src, dst = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 4
os.makedirs(dst, exist_ok=True)

config = json.load(open(os.path.join(src, "config.json")))
text = config["text_config"]
text["num_hidden_layers"] = n
attn = text["linear_attn_config"]
attn["full_attn_layers"] = [layer for layer in attn["full_attn_layers"] if layer <= n]
attn["kda_layers"] = [layer for layer in attn["kda_layers"] if layer <= n]


def keep(name):
    parts = name.split(".")
    return "vision_tower" in name or "layers" not in parts or int(parts[parts.index("layers") + 1]) < n


index = json.load(open(os.path.join(src, "model.safetensors.index.json")))
index["weight_map"] = {name: f for name, f in index["weight_map"].items() if keep(name)}
shards = set(index["weight_map"].values())

for name in os.listdir(src):
    if name in ("config.json", "model.safetensors.index.json"):
        continue
    if name.endswith(".safetensors") and name not in shards:
        continue
    link = os.path.join(dst, name)
    if not os.path.lexists(link):
        os.symlink(os.path.realpath(os.path.join(src, name)), link)

json.dump(config, open(os.path.join(dst, "config.json"), "w"), indent=2)
json.dump(index, open(os.path.join(dst, "model.safetensors.index.json"), "w"), indent=2)
print(f"{dst}: {n} layers, shards {sorted(shards)}, MLA {attn['full_attn_layers']}, KDA {attn['kda_layers']}")
