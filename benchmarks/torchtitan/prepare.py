#!/usr/bin/env python
"""Stage tokenizers / HF weights and tiny datasets for torchtitan benches."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download


# Bench name → (hf_repo_id, mode)
BENCHES = {
    "torchtitan-qwen3-4b-pretrain": ("Qwen/Qwen3-4B-Instruct-2507", "pretrain"),
    "torchtitan-qwen3-4b-pretrain-smoke": ("Qwen/Qwen3-4B-Instruct-2507", "pretrain"),
    "torchtitan-qwen3-4b-sft": ("Qwen/Qwen3-4B-Instruct-2507", "sft"),
    "torchtitan-qwen3-4b-sft-smoke": ("Qwen/Qwen3-4B-Instruct-2507", "sft"),
    "torchtitan-qwen3-30b-pretrain": ("Qwen/Qwen3-30B-A3B", "pretrain"),
    "torchtitan-qwen3-30b-sft": ("Qwen/Qwen3-30B-A3B", "sft"),
    "torchtitan-glm5-pretrain": ("zai-org/GLM-5", "pretrain"),
    "torchtitan-glm5-sft": ("zai-org/GLM-5", "sft"),
}

TOKENIZER_PATTERNS = [
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "vocab.txt",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "chat_template.jinja",
    "*.jinja",
]

CONFIG_PATTERNS = [
    "config.json",
    "generation_config.json",
]

WEIGHT_PATTERNS = [
    "*.safetensors",
    "model.safetensors.index.json",
]


def _config():
    raw = os.environ.get("MILABENCH_CONFIG")
    if not raw:
        return {}
    return json.loads(raw)


def _code_dir() -> Path:
    return Path(os.environ.get("MILABENCH_DIR_CODE", Path(__file__).resolve().parent))


def _data_dir() -> Path:
    return Path(os.environ["MILABENCH_DIR_DATA"])


def _hf_token():
    return (
        os.environ.get("MILABENCH_HF_TOKEN")
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )


def stage_datasets(data: Path, code: Path) -> None:
    for name in ("c4_test", "sft_test"):
        src = code / "assets" / name
        dst = data / name
        if not src.exists():
            continue
        dst.mkdir(parents=True, exist_ok=True)
        for item in src.iterdir():
            target = dst / item.name
            if not target.exists():
                shutil.copy2(item, target)
        print(f"Staged dataset {name} → {dst}")


def download_assets(repo_id: str, mode: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    patterns = list(TOKENIZER_PATTERNS) + list(CONFIG_PATTERNS)
    if mode == "sft":
        patterns.extend(WEIGHT_PATTERNS)

    token = _hf_token()
    print(f"Downloading {repo_id} ({mode}) → {dest}")
    print(f"  allow_patterns: {patterns}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(dest),
        allow_patterns=patterns,
        token=token,
    )
    print(f"Assets ready at {dest}")


def main():
    cfg = _config()
    name = cfg.get("name", "")
    code = _code_dir()
    data = _data_dir()
    data.mkdir(parents=True, exist_ok=True)

    stage_datasets(data, code)

    if name not in BENCHES:
        # Fall back: prepare all tokenizers for every known model (dev convenience).
        print(f"Unknown/empty bench name {name!r}; preparing all pretrain tokenizers.")
        seen = set()
        for repo_id, mode in BENCHES.values():
            if mode != "pretrain" or repo_id in seen:
                continue
            seen.add(repo_id)
            download_assets(repo_id, "pretrain", data / "hf" / repo_id.split("/")[-1])
        return

    repo_id, mode = BENCHES[name]
    dest = data / "hf" / repo_id.split("/")[-1]
    download_assets(repo_id, mode, dest)


if __name__ == "__main__":
    main()
