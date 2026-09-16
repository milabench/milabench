"""Analyze milabench run folders for metric timeline anomalies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from pathlib import Path

from argklass.command import Command
from ..diagnostic import analyze_run, plot_run, print_report
from .tools.error import resolve_run_folder


class Diagnostic(Command):
    """Scan a run folder for metric hangs, epoch gaps, and collection issues."""

    name = "diagnostic"

    # fmt: off
    @dataclass
    class Arguments:
        """Analyze *.data metrics in a milabench run directory."""
        folder        : str           = None                    # Run directory (default: latest under $MILABENCH_BASE/runs)
        select        : Optional[str] = None                    # Filter pack names (substring)
        min_severity  : str           = "info"                  # Minimum severity: info, warn, error
        show_clean    : bool          = False                   # List packs with no issues
        json          : bool          = False                   # Emit JSON instead of text
        plot          : bool          = False                   # Write scatter plots (load, mem, power, rate)
        plot_dir      : Optional[str] = None                    # Plot output directory (default: <run>/diagnostic_plots)
    # fmt: on

    @staticmethod
    def execute(args):
        try:
            folder = resolve_run_folder(args.folder)
        except FileNotFoundError as e:
            print(e)
            return 1

        if args.min_severity not in ("info", "warn", "error"):
            print("min_severity must be one of: info, warn, error")
            return 1

        report = analyze_run(folder, select=args.select)
        if not report.packs:
            msg = f"No *.data files in {folder}"
            if args.select:
                msg += f" matching select={args.select!r}"
            print(msg)
            return 1

        print_report(
            report,
            min_severity=args.min_severity,
            show_clean=args.show_clean,
            json_out=args.json,
        )

        if args.plot:
            try:
                plot_dir = Path(args.plot_dir) if args.plot_dir else None
                written = plot_run(folder, select=args.select, plot_dir=plot_dir)
            except ImportError as e:
                print(e)
                return 1
            if not written:
                print("No plots written (no matching *.data files)")
                return 1
            for path in written:
                print(f"Wrote {path}")

        return 0


COMMANDS = Diagnostic
