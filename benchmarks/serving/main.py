"""Serving benchmark client: milabench runs the inference server (milabench.commands.serving) and waits for it.

The client (vLLM's serving benchmark, lm-eval passes, ``--mode`` / ``--pass`` selection) is benchmate.benchserve;
this pack adds real-text requests (``--real-text``) and its own lm-eval tasks.
"""

import os
import sys

from benchmate import benchserve

import text_corpus


# milabench's zero-shot tasks for reasoning models
EVAL_TASKS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_tasks")


def add_options(parser):
    # real text instead of random token ids, same lengths (--random-input-len / --random-output-len)
    parser.add_argument("--real-text", action="store_true")


def get_samples(args, tokenizer, datasets):
    if args.real_text:
        return text_corpus.make_real_text_dataset(datasets)(
            dataset_path=args.dataset_path, random_seed=args.seed,
        ).sample(
            tokenizer=tokenizer,
            num_requests=args.num_prompts,
            input_len=args.random_input_len,
            output_len=args.random_output_len,
            request_id_prefix=args.request_id_prefix,
        )
    if args.dataset_name == "sharegpt" and args.dataset_path is None:
        args.dataset_path = text_corpus.corpus_path()
    return None


def main(argv):
    benchserve.main(argv[1:], eval_tasks=EVAL_TASKS, samples_hook=get_samples, parser_hook=add_options)


if __name__ == "__main__":
    main(sys.argv)
