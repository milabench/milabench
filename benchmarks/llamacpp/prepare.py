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
    parser.add_argument("--mode", default="host")
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

    if args.mode == "quality":
        # QualityBench, its harness environments, datasets and code-task images (the runs are offline)
        from benchmate.benchserve import quality
        from main import quality_spec  # the run's own spec from the same options
        q = ArgumentParser()
        q.add_argument("--quality-tier", default="quick")
        q.add_argument("--quality-thinking", default="off")
        q.add_argument("--quality-tasks", default="gsm8k,ifeval,mmlu_pro,humaneval,mbpp,lcb")
        q.add_argument("--quality-thinking-max-tokens", type=int, default=None)
        q.add_argument("--quality-max-tokens", type=int, default=None)
        quality.prepare(quality_spec(q.parse_known_args(argv)[0]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
