"""Print milabench-relevant environment information."""

from dataclasses import dataclass

from argklass.command import Command

from ...system import get_tracked_options, as_environment_variable, SystemConfig


class Env(Command):
    """Print milabench environment variables."""

    name = "env"

    @dataclass
    class Arguments:
        """Print milabench environment variables."""

    @staticmethod
    def execute(args):
        _ = SystemConfig()

        for k, option in get_tracked_options().items():
            env_name = as_environment_variable(k)
            value = option["value"]
            default = option["default"]

            if value is None or value == default:
                print("# ", end="")

            print(f"export {env_name}={value}")


COMMANDS = Env
