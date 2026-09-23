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
    args, _ = parser.parse_known_args(argv)

    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=args.repo, filename=args.file)
    print(f"prepared {args.repo}/{args.file} -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
