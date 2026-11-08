"""Truncated copy of a HF safetensors checkpoint (GLM-5.3, FP8 as it is): its first N decoder layers, the embeddings,
the final norm and the LM head; no MTP layer (num_nextn_predict_layers = 0). Reads only the shards that hold them.

    venv/torch/bin/python truncate_hf.py SRC DST N
"""
import json
import shutil
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

src, dst, n = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
index = json.loads((src / "model.safetensors.index.json").read_text())


def kept(name):
    return not name.startswith("model.layers.") or int(name.split(".")[2]) < n


shards = {}
for name, file in index["weight_map"].items():
    if kept(name):
        shards.setdefault(file, []).append(name)

dst.mkdir(parents=True, exist_ok=True)
weight_map = {}
for i, (file, names) in enumerate(sorted(shards.items()), 1):
    out = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
    with safe_open(src / file, framework="pt") as tensors:
        save_file({name: tensors.get_tensor(name) for name in names}, dst / out, metadata={"format": "pt"})
    weight_map.update(dict.fromkeys(names, out))
    print(f"{out}: {len(names)} tensors from {file}", flush=True)
(dst / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}, indent=2))

config = json.loads((src / "config.json").read_text())
config["num_hidden_layers"] = n
config["num_nextn_predict_layers"] = 0
(dst / "config.json").write_text(json.dumps(config, indent=2))
for path in src.iterdir():
    if path.is_file() and path.suffix in (".json", ".jinja", ".txt", ".model", ".py") and not path.name.startswith(
            ("config.json", "model.safetensors")):
        shutil.copy(path, dst / path.name)
