#!/usr/bin/env python
"""Entry point: milabench argv -> TorchTitan config options -> torchtitan.train.

Usage (one process per GPU, under torchrun/benchrun, instrumented by voirfile.py):

    main.py --recipe qwen38 --flavor 27B --n-layers 16 --compile regions+graphs ...
    main.py --recipe mistral_7b_pretrain --seq-len 2048 --microbatch-sequences 1 ...

Every ``Options`` field in ``tt_native/options.py`` is a ``--kebab-case`` flag;
``--recipe`` names the config function in ``tt_configs``.
"""

import time

_PROCESS_START = time.time()

import argparse
import json
import os
import sys
import typing
from dataclasses import MISSING, fields


def _parse_args(argv):
    from tt_native.options import OPTIONS_ENV, Options

    hints = typing.get_type_hints(Options)
    parser = argparse.ArgumentParser(description=__doc__)
    for f in fields(Options):
        flag = "--" + f.name.replace("_", "-")
        hint = hints[f.name]
        args = typing.get_args(hint)
        if hint is bool:
            parser.add_argument(flag, type=lambda v: v.lower() in ("1", "true", "yes"))
        elif typing.get_origin(hint) is typing.Literal:
            parser.add_argument(flag, choices=args)
        else:
            # Optional[X] -> X
            base = next((a for a in args if a is not type(None)), hint)
            if typing.get_origin(base) is typing.Literal:
                parser.add_argument(flag, choices=typing.get_args(base))
            else:
                parser.add_argument(flag, type=base)
    ns = parser.parse_args(argv)
    options = {k: v for k, v in vars(ns).items() if v is not None}
    os.environ[OPTIONS_ENV] = json.dumps(options)
    return options


def _start_nodemon():
    """Start the node sampler beside the training processes (output: ``benchmate.toggles.nodemon_dir``).

    Every rank calls this: the first on each host takes the lock and samples, the
    others return at once. It runs before TorchTitan is imported, so model init and
    the first steps are covered too."""
    from benchmate.nodemon import run
    from benchmate.toggles import nodemon_dir

    out = nodemon_dir()
    if out is None:
        return None
    bench = json.loads(os.environ.get("MILABENCH_CONFIG") or "{}").get("name", "torchtitan")
    return run(out, bench=bench)


def main():
    import tt_native.instrument as instrument

    instrument.PROCESS_START = _PROCESS_START
    options = _parse_args(sys.argv[1:])
    sampler = _start_nodemon()

    try:
        from torch_compat import ensure_torchtitan_torch_compat

        ensure_torchtitan_torch_compat()

        from torchtitan.train import main as torchtitan_main

        recipe = options.get("recipe", "qwen38")
        sys.argv = [sys.argv[0], "--module", "tt_configs", "--config", recipe]
        torchtitan_main()
    finally:
        if sampler is not None:
            sampler.stop("bench done")


if __name__ == "__main__":
    main()
