#!/usr/bin/env python
"""Download the tokenizer and the FineWeb subset used by the runs.

Called by ``milabench prepare`` with the bench argv (same flags as main.py);
only ``--dataset`` and ``--samples`` matter here. Needs internet access (run
on a login node); the runs themselves are offline.
"""

import argparse
import os

# make_env sets the runs offline; prepare is the step that downloads.
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["HF_DATASETS_OFFLINE"] = "0"

from tt_native.assets import (  # noqa: E402
    DEFAULT_FINEWEB_SAMPLES,
    TOKENIZER_FILES,
    TOKENIZER_REPO,
    TOKENIZER_REVISION,
    fineweb_dir,
    tokenizer_dir,
)


def _token():
    return os.environ.get("MILABENCH_HF_TOKEN") or os.environ.get("HF_TOKEN")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="fineweb")
    parser.add_argument("--samples", type=int, default=DEFAULT_FINEWEB_SAMPLES)
    args, _ = parser.parse_known_args()

    if args.dataset == "random":
        print("Random tokens: nothing to prepare")
        return

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

    from benchmate.fineweb import prepare_fineweb

    prepare_fineweb(fineweb_dir(), args.samples, token=_token())


if __name__ == "__main__":
    main()
