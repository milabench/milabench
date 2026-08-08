"""Merge or deduplicate milabench scaling profile YAML files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from argklass.command import Command
from argklass.arguments import argument


class MergeScaling(Command):
    """Merge scaling profiles or deduplicate a single profile."""

    name = "merge-scaling"

    # fmt: off
    @dataclass
    class Arguments:
        """Merge scaling profiles or deduplicate a single profile."""
        files  : list[str] = argument(nargs="+")  # Scaling YAML files to merge
        output : str       = "merged.yaml"        # Output path
        dedupe : bool      = False                # Deduplicate one file instead of merging
    # fmt: on

    @staticmethod
    def execute(args):
        from ...sizer import deduplicate_scaling_file, merge_scaling_files

        if not args.files:
            raise SystemExit("merge-scaling: at least one scaling file is required")

        output = Path(args.output)

        if args.dedupe:
            if len(args.files) != 1:
                raise SystemExit("merge-scaling: --dedupe requires exactly one input file")
            source = Path(args.files[0])
            if not source.is_file():
                raise SystemExit(f"merge-scaling: file not found: {source}")
            deduplicate_scaling_file(str(source), output=str(output))
            print(f"Wrote deduplicated scaling profile to {output}")
            return 0

        missing = [path for path in args.files if not Path(path).is_file()]
        if missing:
            raise SystemExit(
                "merge-scaling: file(s) not found: "
                + ", ".join(str(path) for path in missing)
            )

        merge_scaling_files(*args.files, output=str(output))
        print(f"Merged {len(args.files)} scaling file(s) into {output}")
        return 0


COMMANDS = MergeScaling
