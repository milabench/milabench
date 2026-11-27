#!/usr/bin/env python
"""Fetch what a fine-tuning run loads, so the run itself never downloads: the base model's weights and
tokenizer, and the pinned dataset slice. CPU only (no GPU, no model load)."""

import os

from argklass import ArgumentParser

from ftargs import Arguments


def main():
    parser = ArgumentParser()
    parser.add_arguments(Arguments)
    args, _ = parser.parse_known_args()
    token = os.environ.get("HF_TOKEN")

    from huggingface_hub import snapshot_download

    path = snapshot_download(
        args.model,
        revision=args.model_revision,
        token=token,
        allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja", "*.py", "LICENSE*"],
        ignore_patterns=["original/*", "metal/*", "*.gguf"],
    )
    print(f"model {args.model} -> {path}", flush=True)

    from datasets import load_dataset

    ds = load_dataset(args.dataset, revision=args.dataset_revision, split=f"train[:{args.num_samples}]")
    print(f"dataset {args.dataset}@{args.dataset_revision}: {len(ds)} rows", flush=True)


if __name__ == "__main__":
    main()
