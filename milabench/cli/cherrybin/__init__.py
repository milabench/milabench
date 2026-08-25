"""Optional cherrybin data archive commands."""

from argklass.command import ParentCommand


class Cherrybin(ParentCommand):
    """Checkout or update a cherrybin archive of benchmark datasets."""

    name: str = "cherrybin"

    @staticmethod
    def module():
        import milabench.cli.cherrybin
        return milabench.cli.cherrybin


COMMANDS = Cherrybin
