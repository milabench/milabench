#!/usr/bin/env python
"""Milabench entrypoint for AllenAI open-instruct SFT."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# voir/accelerate run with cwd=src/; keep wrapper modules on sys.path.
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmate.monitor import get_rank, setupvoir

from compat import (
    apply_runtime_compat,
    bind_deepspeed_args,
    enable_skip_model_save,
    install_rate_hook,
    parse_milabench_argv,
)


def run() -> None:
    skip_model_save = parse_milabench_argv()
    apply_runtime_compat()

    from open_instruct.finetune import FlatArguments, TokenizerConfig, main as finetune_main
    from open_instruct.utils import ArgumentParserPlus, check_oe_eval_internal

    install_rate_hook()

    check_oe_eval_internal()
    parser = ArgumentParserPlus((FlatArguments, TokenizerConfig))
    args, tc = parser.parse_args_into_dataclasses()

    bind_deepspeed_args(args)
    if skip_model_save:
        enable_skip_model_save()

    monogpu = os.environ.get("WORLD_SIZE", "1") == "1"
    monitor = None
    if get_rank() in (-1, 0):
        _, monitor = setupvoir(monogpu=monogpu, interval=1)

    try:
        finetune_main(args, tc)
    finally:
        if monitor is not None:
            monitor.stop()


if __name__ == "__main__":
    run()
