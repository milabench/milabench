#!/usr/bin/env python
"""Everything the offline runs need: the model, the real-text corpus of the speed passes, the retrieval datasets.

argv: <server argv> -- <client argv> (benchmarks/serving's build_prepare_plan)."""

import json
import os
import sys
from argparse import ArgumentParser

from benchmate.hugginface import download_hf_model

import retrieval_eval


def arguments():
    argv = [arg for arg in sys.argv[1:] if arg != "--"]
    parser = ArgumentParser()
    parser.add_argument("server_model", nargs="?", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--retrieval-text", default="real")
    parser.add_argument("--chat-template", default=None)
    args, _ = parser.parse_known_args(argv)
    return args


def client_passes() -> list[dict]:
    try:
        cfg = json.loads(os.environ.get("MILABENCH_CONFIG", "{}"))
    except json.JSONDecodeError:
        return []
    arch = cfg.get("system", {}).get("arch", "cuda")
    variant = ((cfg.get("variants") or {}).get(arch) or {}).get("client") or {}
    return variant.get("passes") or (cfg.get("client") or {}).get("passes") or []


def main():
    args = arguments()
    passes = client_passes()

    if args.retrieval_text == "real":
        from benchmate.benchserve import text_corpus
        text_corpus.build_corpus()

    for spec in passes:
        if "retrieval" in spec:
            retrieval_eval.prepare(spec["retrieval"] or {})

    model = args.server_model or args.model
    if model is None:
        raise SystemExit("prepare: no model found in server positional or --model")
    download_hf_model(model)

    # a server --chat-template file that does not exist: the model's own template (e.g. Nemotron-3.5-Rerank's
    # model card serves it with the tokenizer's template written to a file). Relative to this folder, the
    # server's working directory.
    if args.chat_template and not os.path.exists(args.chat_template):
        from transformers import AutoTokenizer

        template = AutoTokenizer.from_pretrained(model).get_chat_template()
        os.makedirs(os.path.dirname(os.path.abspath(args.chat_template)), exist_ok=True)
        with open(args.chat_template, "w") as f:
            f.write(template)
        print(f"chat template of {model} -> {args.chat_template}")
    print("Prepare script completed successfully")


if __name__ == "__main__":
    main()
