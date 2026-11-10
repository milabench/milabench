"""Serving benchmark client: milabench runs the inference server (milabench.commands.serving) and waits for it.

The client (vLLM's serving benchmark, --real-text, lm-eval passes, ``--mode`` / ``--pass`` selection) is
benchmate.benchserve; this pack adds its own lm-eval tasks.
"""

import os
import sys

from benchmate import benchserve


# milabench's zero-shot tasks for reasoning models
EVAL_TASKS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_tasks")


def main(argv):
    benchserve.main(argv[1:], eval_tasks=EVAL_TASKS)


if __name__ == "__main__":
    main(sys.argv)
