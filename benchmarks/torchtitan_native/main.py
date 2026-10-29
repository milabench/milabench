#!/usr/bin/env python
"""Entry point: milabench argv -> TorchTitan config options -> torchtitan.train.

Usage (one process per GPU, under torchrun/benchrun):

    main.py --flavor 27B --n-layers 16 --compile regions+graphs --ac selective ...

Every ``Options`` field in ``tt_native/configs.py`` is a ``--kebab-case`` flag.
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
    from tt_native.configs import OPTIONS_ENV, Options

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


def main():
    import tt_native.instrument as instrument

    instrument.PROCESS_START = _PROCESS_START
    _parse_args(sys.argv[1:])

    from torchtitan.train import main as torchtitan_main

    sys.argv = [sys.argv[0], "--module", "tt_native.configs", "--config", "qwen38"]
    torchtitan_main()


if __name__ == "__main__":
    main()
