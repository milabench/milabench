#!/usr/bin/env python
"""Download open-instruct model + dataset into MILABENCH HF_HOME (dirs.data)."""

from __future__ import annotations

import os
import sys
from argparse import ArgumentParser


def arguments():
    parser = ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument(
        "--dataset_mixer_list",
        nargs="+",
        default=None,
        help="Alternating dataset name and weight, e.g. name 1.0",
    )
    args, _ = parser.parse_known_args([a for a in sys.argv[1:] if a != "--"])
    return args


def _hf_token():
    token = (
        os.environ.get("MILABENCH_HF_TOKEN")
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    token = (token or "").strip()
    return token or None


def main() -> None:
    from benchmate.hugginface import download_hf_dataset, download_hf_model

    args = arguments()
    token = _hf_token()

    models = {
        m
        for m in (args.model_name_or_path, args.tokenizer_name)
        if m and not os.path.isdir(m)
    }
    for model in sorted(models):
        download_hf_model(model, token=token)

    mixer = args.dataset_mixer_list or []
    # open-instruct style: [dataset_a, weight_a, dataset_b, weight_b, ...]
    for i in range(0, len(mixer), 2):
        ds = mixer[i]
        if ds and not os.path.isdir(ds):
            download_hf_dataset(ds, token=token)

    print(f"HF_HOME={os.environ.get('HF_HOME')}")
    print(f"MILABENCH_DIR_DATA={os.environ.get('MILABENCH_DIR_DATA')}")
    print("=" * 60)
    print("Prepare script completed successfully")
    print("=" * 60)


if __name__ == "__main__":
    main()
