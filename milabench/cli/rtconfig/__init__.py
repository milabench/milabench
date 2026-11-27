"""Runtime config commands."""

from argklass.command import ParentCommand


class RuntimeConfig(ParentCommand):
    """Add configs to a running benchmark (see benchmate.config)."""

    name: str = "rtconfig"

    @staticmethod
    def module():
        import milabench.cli.rtconfig
        return milabench.cli.rtconfig


COMMANDS = RuntimeConfig
