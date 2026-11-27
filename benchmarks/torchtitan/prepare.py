#!/usr/bin/env python
"""Download what a run reads: tokenizer/config (weights for SFT), FineWeb, SFT data.

Called by ``milabench prepare`` with the bench argv (same flags as main.py);
only ``--recipe``, ``--dataset`` and ``--samples`` matter here. Needs internet
access (run on a login node); the runs themselves are offline.
"""

import argparse
import json
import os
from pathlib import Path

# make_env sets the runs offline; prepare is the step that downloads.
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["HF_DATASETS_OFFLINE"] = "0"

from tt_native.assets import (  # noqa: E402
    DEFAULT_FINEWEB_SAMPLES,
    TOKENIZER_FILES,
    TOKENIZER_REPO,
    TOKENIZER_REVISION,
    data_dir,
    fineweb_dir,
    tokenizer_dir,
)

# HF backend configs (tt_configs) -> Hugging Face repo. ``*_sft`` configs
# also need the weights; pre-training starts from scratch.
HF_MODELS = {
    "qwen3_4b": "Qwen/Qwen3-4B-Instruct-2507",
    "qwen3_30b": "Qwen/Qwen3-30B-A3B",
    "mistral_7b": "mistralai/Mistral-7B-v0.1",
    "olmo_7b": "allenai/OLMo-7B-hf",
    "mixtral_8x7b": "mistralai/Mixtral-8x7B-v0.1",
    "olmoe_7b": "allenai/OLMoE-1B-7B-0924",
    "gemma4_26b": "google/gemma-4-26B-A4B",
    "deepseek_v2_lite": "deepseek-ai/DeepSeek-V2-Lite",
    "glm5": "zai-org/GLM-5",
}

HF_PATTERNS = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "vocab.txt",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "*.jinja",
]

WEIGHT_PATTERNS = ["*.safetensors", "model.safetensors.index.json"]

DEFAULT_SFT_CHAT_TEMPLATE = """{% for message in messages %}{% if message['role'] == 'user' %}{{ message['content'] }}
{% elif message['role'] == 'assistant' %}{{ message['content'] }}{% endif %}{% endfor %}"""

_LOREM = (
    "The quick brown fox jumps over the lazy dog while the benchmark measures "
    "throughput across several training steps on a small synthetic corpus. "
)


def _token():
    return os.environ.get("MILABENCH_HF_TOKEN") or os.environ.get("HF_TOKEN")


def _hf_model(recipe: str):
    """(repo, sft) for an HF backend recipe, None for native ones."""
    base, _, mode = recipe.rpartition("_")
    if base in HF_MODELS and mode in ("pretrain", "sft"):
        return HF_MODELS[base], mode == "sft"
    return None


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
    print(f"Wrote default SFT chat template -> {jinja}")


def stage_sft_dataset() -> None:
    """Synthetic single-turn chat samples (question/answer) for the SFT benches."""
    dst = data_dir() / "sft_test"
    target = dst / "data.json"
    if target.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    with open(target, "w") as fp:
        for i in range(2000):
            sample = {
                "question": f"Question {i}: what is {i} + {i}?",
                "answer": f"{i} + {i} = {2 * i}. " + _LOREM,
            }
            fp.write(json.dumps(sample) + "\n")
    print(f"Staged SFT dataset -> {target}")


def download_hf_model(repo_id: str, sft: bool) -> None:
    from huggingface_hub import snapshot_download

    dest = data_dir() / "hf" / repo_id.split("/")[-1]
    patterns = HF_PATTERNS + (WEIGHT_PATTERNS if sft else [])
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(dest),
        allow_patterns=patterns,
        token=_token(),
    )
    if sft:
        ensure_sft_chat_template(dest)
    print(f"{repo_id} ({'sft' if sft else 'pretrain'}) at {dest}")


def download_native_tokenizer() -> None:
    from huggingface_hub import snapshot_download

    dest = tokenizer_dir()
    snapshot_download(
        repo_id=TOKENIZER_REPO,
        revision=TOKENIZER_REVISION,
        local_dir=str(dest),
        allow_patterns=TOKENIZER_FILES,
        token=_token(),
    )
    print(f"Tokenizer {TOKENIZER_REPO}@{TOKENIZER_REVISION[:8]} at {dest}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", default="qwen38")
    parser.add_argument("--dataset", default="fineweb")
    parser.add_argument("--samples", type=int, default=DEFAULT_FINEWEB_SAMPLES)
    args, _ = parser.parse_known_args()

    hf = _hf_model(args.recipe)
    if hf is not None:
        repo_id, sft = hf
        # Pre-training with random tokens still builds the model from its HF config.
        download_hf_model(repo_id, sft)
        if sft:
            stage_sft_dataset()
            return
    elif args.dataset != "random":
        download_native_tokenizer()

    if args.dataset == "random":
        print("Random tokens: no dataset to prepare")
        return

    from benchmate.fineweb import prepare_fineweb

    prepare_fineweb(fineweb_dir(), args.samples, token=_token())


if __name__ == "__main__":
    main()
