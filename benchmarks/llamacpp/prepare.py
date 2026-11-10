#!/usr/bin/env python
"""Download the GGUF checkpoint into the HF cache (XDG_CACHE_HOME is set by
the milabench activator, so this lands under the pack cache dir)."""
import os
import sys

from argparse import ArgumentParser


def main(argv):
    parser = ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--file", required=True)
    parser.add_argument("--model", default=None, help="HF id of the client's tokenizer")
    args, _ = parser.parse_known_args(argv)

    from huggingface_hub import snapshot_download
    from main import download_gguf  # every shard of a split GGUF

    path = download_gguf(args.repo, args.file)
    print(f"prepared {args.repo}/{args.file} -> {path}")

    # the bench client tokenizes with the original model's tokenizer
    tokenizer = args.model or args.repo
    snapshot_download(tokenizer, allow_patterns=["*.json", "*.txt", "*.model", "*.tiktoken", "*.jinja"])
    print(f"prepared tokenizer {tokenizer}")

    # --real-text in the client argv (benchmate.benchserve)
    from benchmate.benchserve import prepare_text_corpus
    prepare_text_corpus(argv, passes=[])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
