"""Deprecated: SQL schema generation moved to the dashboard."""

import sys
from dataclasses import dataclass

from argklass.command import Command


class Sql(Command):
    """Generate the SQL setup scripts for the milabench database (removed)."""

    name = "sqlsetup"

    @dataclass
    class Arguments:
        """Generate the SQL setup scripts for the milabench database (removed)."""

    @staticmethod
    def execute(args):
        print(
            "milabench sqlsetup was removed.\n"
            "Database schema and Alembic migrations now live in the dashboard package:\n"
            "  python -m dashboard.server.database.models\n"
            "  cd dashboard/dashboard && alembic upgrade head",
            file=sys.stderr,
        )
        sys.exit(2)


COMMANDS = Sql
