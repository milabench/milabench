"""Helpers shared by prepare.py and main.py."""

from __future__ import annotations

import json
import os
import shlex
from pathlib import Path

# flags that take several values (nargs>1): configured as one string in the yaml
_MULTI_VALUE = ("--spec", "--freeze-params-name-list", "--only-train-params-name-list")


def config() -> dict:
    raw = os.environ.get("MILABENCH_CONFIG")
    return json.loads(raw) if raw else {}


def code_dir() -> Path:
    return Path(os.environ.get("MILABENCH_DIR_CODE", Path(__file__).resolve().parent))


def data_dir() -> Path:
    return Path(os.environ["MILABENCH_DIR_DATA"]) / "slime"


def slime_src() -> Path:
    return code_dir() / "slime_src"


def megatron_src() -> Path:
    return code_dir() / "Megatron-LM"


def assemble(options: dict) -> list[str]:
    """Same flag rendering as ``milabench.utils.assemble_options`` (``true`` -> bare flag)."""
    args = []
    for k, v in (options or {}).items():
        if v is None:
            continue
        if v is True:
            args.append(k)
        else:
            args.extend([k, str(v)])
    return expand_multi_value(args)


def expand_multi_value(argv: list[str]) -> list[str]:
    out = []
    it = iter(argv)
    for a in it:
        out.append(a)
        if a in _MULTI_VALUE:
            out.extend(shlex.split(next(it)))
    return out


def model_argv() -> list[str]:
    """Architecture args (``model.argv`` in the config), shared by conversion and training."""
    return assemble(config().get("model", {}).get("argv"))


def resolve(path: str) -> Path:
    """Expand milabench's ``{milabench_data}`` placeholder (the raw config is not formatted)."""
    return Path(str(path).replace("{milabench_data}", os.environ["MILABENCH_DIR_DATA"]))
