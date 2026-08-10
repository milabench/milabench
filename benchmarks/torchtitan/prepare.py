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
    "torchtitan-mistral-7b-pretrain": ("mistralai/Mistral-7B-v0.1", "pretrain"),
    "torchtitan-mistral-7b-sft": ("mistralai/Mistral-7B-v0.1", "sft"),
    "torchtitan-olmo-7b-pretrain": ("allenai/OLMo-7B-hf", "pretrain"),
    "torchtitan-olmo-7b-sft": ("allenai/OLMo-7B-hf", "sft"),
    "torchtitan-mixtral-8x7b-pretrain": ("mistralai/Mixtral-8x7B-v0.1", "pretrain"),
    "torchtitan-mixtral-8x7b-sft": ("mistralai/Mixtral-8x7B-v0.1", "sft"),
    "torchtitan-olmoe-7b-pretrain": ("allenai/OLMoE-1B-7B-0924", "pretrain"),
    "torchtitan-olmoe-7b-sft": ("allenai/OLMoE-1B-7B-0924", "sft"),
    "torchtitan-gemma4-26b-pretrain": ("google/gemma-4-26B-A4B", "pretrain"),
    "torchtitan-gemma4-26b-sft": ("google/gemma-4-26B-A4B", "sft"),
    "torchtitan-deepseek-v2-lite-pretrain": ("deepseek-ai/DeepSeek-V2-Lite", "pretrain"),
    "torchtitan-deepseek-v2-lite-sft": ("deepseek-ai/DeepSeek-V2-Lite", "sft"),
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



DEFAULT_SFT_CHAT_TEMPLATE = """{% for message in messages %}{% if message['role'] == 'user' %}{{ message['content'] }}
{% elif message['role'] == 'assistant' %}{{ message['content'] }}{% endif %}{% endfor %}"""


def ensure_sft_chat_template(dest: Path) -> None:
    """Models like Mistral-7B-v0.1 ship without chat templates; SFT needs one."""
    jinja = dest / "chat_template.jinja"
    if jinja.exists():
        return
    cfg_path = dest / "tokenizer_config.json"
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text())
        except json.JSONDecodeError:
            cfg = {}
        if cfg.get("chat_template"):
            return
    jinja.write_text(DEFAULT_SFT_CHAT_TEMPLATE)
    print(f"Wrote default SFT chat template → {jinja}")

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
    if mode == "sft":
        ensure_sft_chat_template(dest)
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
