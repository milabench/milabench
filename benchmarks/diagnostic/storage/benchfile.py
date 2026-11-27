"""Storage throughput diagnostic: O_DIRECT sequential read/write and small-file reads."""

from milabench.commands import PackCommand
from milabench.pack import Package


class Storage(Package):
    base_requirements = "requirements.in"
    prepare_script = "prepare.py"
    main_script = "main.py"

    def build_run_plan(self):
        # One process on the main node; main.py logs through setupvoir (stdout smuggling)
        return PackCommand(self, lazy=True).use_stdout()


__pack__ = Storage
