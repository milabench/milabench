"""Locations of the assets prepared by ``prepare.py`` and read by the runs.

Kept free of torch/torchtitan imports so ``prepare.py`` stays light on login
nodes.
"""

from __future__ import annotations

import os
from pathlib import Path

# Tokenizer of the benchmarked model, pinned for reproducible tokenization.
TOKENIZER_REPO = "Qwen/Qwen3.8-27B"
TOKENIZER_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "config.json",
    "generation_config.json",
]

# Kimi K3 ships a tiktoken tokenizer (tiktoken.model + remote code), which
# TorchTitan's HuggingFaceTokenizer cannot read. Its runs tokenize FineWeb with
# DeepSeek-V3.1's tokenizer.json instead, as TorchTitan's DeepSeek recipes do:
# its ids (< 129,280) index K3's 163,840-entry embedding.
TOKENIZERS = {
    "qwen3_8": (TOKENIZER_REPO, TOKENIZER_REVISION, TOKENIZER_FILES),
    "kimi_k3": (
        "deepseek-ai/DeepSeek-V3.1-Base",
        "d3d4eafdc470de44bbf6f0a74f852eb522357be8",
        ["tokenizer.json", "tokenizer_config.json"],
    ),
}

# Shared with the HF torchtitan pack (benchmate.fineweb grows the subset to
# the largest request, and every bench reads its own prefix of it).
FINEWEB_SUBSET = "sample-10BT"

# Documents read by the runs. A rank cycles over its 1/world share, so the
# count only has to cover the largest run without repeating: 264 GPUs x
# ~90 steps x ~19 docs (12,288 tokens/GPU, ~650 tokens/doc) ~ 450k.
DEFAULT_FINEWEB_SAMPLES = 500_000


def data_dir() -> Path:
    return Path(os.environ["MILABENCH_DIR_DATA"])


def tokenizer_dir(model: str = "qwen3_8") -> Path:
    return data_dir() / "hf" / TOKENIZERS[model][0].split("/")[-1]


def fineweb_dir() -> Path:
    return data_dir() / "fineweb" / FINEWEB_SUBSET
