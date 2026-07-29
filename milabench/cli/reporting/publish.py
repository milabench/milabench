"""Publish run results to the dashboard via push key."""

import os
import sys
from dataclasses import dataclass
from typing import Optional

from argklass.arguments import argument
from argklass.command import Command

from .._push_results import publish_results


class Publish(Command):
    """Publish run results to the dashboard.

    Example::

        milabench publish ./runs/my-run --key $MILABENCH_PUBLISH_KEY
    """

    name = "publish"

    # fmt: off
    @dataclass
    class Arguments:
        """Publish run results to the dashboard."""
        runs          : list[str]     = argument(default=[], nargs="+")               # Run directory to publish
        key           : Optional[str] = os.getenv("MILABENCH_PUBLISH_KEY", None)     # Push key
        dashboard_url : Optional[str] = os.getenv("MILABENCH_DASHBOARD_URL", "https://www.milabench.com")  # Dashboard URL
    # fmt: on

    @staticmethod
    def execute(args):
        if not args.key:
            print(
                "error: push key required (pass --key or set MILABENCH_PUBLISH_KEY)",
                file=sys.stderr,
            )
            sys.exit(2)

        success = publish_results(
            args.runs,
            push_key=args.key,
            dashboard_url=args.dashboard_url,
        )
        if not success:
            sys.exit(1)


COMMANDS = Publish
