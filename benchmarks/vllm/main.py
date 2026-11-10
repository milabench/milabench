"""vLLM / ATOM / sglang serving benchmark: starts the inference server, then runs every client pass against it.

The client (vLLM's serving benchmark, lm-eval passes, per-pass recording) is benchmate.benchserve.
"""

import os
import sys

from benchmate.benchserve import (
    InferenceServer,
    client_passes,
    make_observer,
    run_benchmark_watched,
    split_args,
)
from server_backends import (
    build_server_command,
    resolved_server_backend,
    resolved_server_command,
)
from atom_aiter_compat import ensure_atom_aiter_compat


# milabench's zero-shot tasks for reasoning models (milabench_gsm8k, milabench_arc_challenge, milabench_gpqa_diamond)
EVAL_TASKS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_tasks")


def main(argv):
    ensure_atom_aiter_compat()

    server_argv, bench_argv = split_args(argv, skip_program=True)

    backend = resolved_server_backend()
    command = build_server_command(
        server_argv, backend=backend, command=resolved_server_command()
    )

    observer, bench_monitor = make_observer()

    with bench_monitor():
        with InferenceServer(command, name=f"{backend} server") as server:
            # every pass in this process, each recorded as its own bench (<bench>.<pass>)
            run_benchmark_watched(
                server, bench_argv, client_passes("all"), record=True, eval_tasks=EVAL_TASKS,
            )


if __name__ == "__main__":
    main(sys.argv)
