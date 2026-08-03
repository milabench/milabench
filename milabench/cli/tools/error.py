"""Extract readable errors from a milabench run directory."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from argklass.command import Command

from ...testing import interleave
from ...utils import multilogger, validation_layers


def _default_runs_root() -> Path | None:
    base = os.environ.get("MILABENCH_BASE")
    if base:
        runs = Path(base) / "runs"
        if runs.is_dir():
            return runs
    return None


def resolve_run_folder(folder: str | None) -> Path:
    """Resolve a run directory from an explicit path or the latest under MILABENCH_BASE/runs."""
    if folder:
        path = Path(folder).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Run folder not found: {path}")
        if path.is_file():
            path = path.parent
        return path

    runs_root = _default_runs_root()
    if runs_root is None:
        raise FileNotFoundError(
            "No folder given and MILABENCH_BASE/runs is not set or missing. "
            "Pass a run directory, e.g. milabench tools error /data/results/runs/<run>"
        )

    candidates = [p for p in runs_root.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No run directories under {runs_root}")

    return max(candidates, key=lambda p: p.stat().st_mtime)


class Error(Command):
    """Extract and pretty-print errors from a milabench run."""

    name = "error"

    # fmt: off
    @dataclass
    class Arguments:
        """Extract and pretty-print errors from a milabench run."""
        folder    : str           = None   # Run directory (default: latest under $MILABENCH_BASE/runs)
        fulltrace : bool          = False  # Show full stack traces (default: grouped summary)
        select    : Optional[str] = None   # Optional substring filter on *.data filenames
    # fmt: on

    @staticmethod
    def execute(args):
        try:
            folder = resolve_run_folder(args.folder)
        except FileNotFoundError as e:
            print(e)
            return 1

        print(f"Run: {folder}")

        files = [f for f in folder.iterdir() if f.name.endswith(".data")]
        if args.select:
            files = [f for f in files if args.select in f.name]
        if not files:
            msg = f"No *.data files in {folder}"
            if args.select:
                msg += f" matching select={args.select!r}"
            print(msg)
            return 1

        layers = validation_layers("error")
        with multilogger(*layers, short=not args.fulltrace) as log:
            for entry in interleave(*files):
                log(entry)

        return log.result()


COMMANDS = Error
