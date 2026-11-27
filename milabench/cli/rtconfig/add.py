"""Add configs to the runtime config of a benchmark."""

import os
import sys
from dataclasses import dataclass
from typing import Optional

import yaml
from argklass.command import Command

from benchmate.config import RUNTIME_CONFIG_NAME, add_runtime_configs


def load_configs(path):
    """Read argv list from a yaml, either ``multiconfig: [...]`` or a plain list"""
    with open(path) as fp:
        data = yaml.safe_load(fp)

    if isinstance(data, dict):
        data = data.get("multiconfig")

    if not isinstance(data, list) or not data:
        raise ValueError(f"{path}: expected a non empty list of argv under `multiconfig`")

    return data


def resolve_target(bench, base=None, path=None, environ=None):
    """Where the benchmark ``bench`` looks for its runtime config"""
    environ = os.environ if environ is None else environ

    path = path or environ.get("MILABENCH_RUNTIME_CONFIG")
    if path:
        return path

    runs = environ.get("MILABENCH_DIR_RUNS")
    base = base or environ.get("MILABENCH_BASE")

    if not runs and base:
        runs = os.path.join(os.path.abspath(os.path.expanduser(base)), "runs")

    if not runs:
        raise ValueError("Set --base, $MILABENCH_BASE or --path")

    return os.path.join(runs, bench, RUNTIME_CONFIG_NAME)


class Add(Command):
    """Append the configs of a yaml file to a benchmark's runtime config."""

    name = "add"

    # fmt: off
    @dataclass
    class Arguments:
        """Append the configs of a yaml file to a benchmark's runtime config."""
        bench  : str            = ""    # Benchmark to add the configs to
        config : str            = ""    # Yaml file with a `multiconfig` list of argv
        base   : Optional[str]  = None  # Milabench base ($MILABENCH_BASE)
        path   : Optional[str]  = None  # Runtime config file to write to, overrides --bench/--base
    # fmt: on

    @staticmethod
    def execute(args):
        if not args.config or not (args.bench or args.path):
            print("Error: --config and --bench are required", file=sys.stderr)
            return 1

        try:
            configs = load_configs(args.config)
            target = resolve_target(args.bench, args.base, args.path)
        except (OSError, ValueError, yaml.YAMLError) as err:
            print(f"Error: {err}", file=sys.stderr)
            return 1

        add_runtime_configs(configs, target)
        print(f"Added {len(configs)} config(s) to {target}")
        return 0


COMMANDS = Add
