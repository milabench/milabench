"""Interconnect bandwidth diagnostic: CPU<->GPU, GPU<->GPU, collectives, node<->node.

One process per GPU on every node through torchrun. Main node runs locally and
owns milabench metrics; workers are launched via ``srun -x <main>`` (``nolog``).
"""

from milabench.commands import PackCommand
from milabench.commands.srun import TorchrunSrun
from milabench.pack import Package


class Netbench(Package):
    base_requirements = "requirements.in"
    prepare_script = "prepare.py"
    main_script = "main.py"

    def build_run_plan(self):
        # PackCommand (no VoirCommand): metrics via BenchObserver(stdout=True).
        return TorchrunSrun(PackCommand(self, lazy=True)).use_stdout()


__pack__ = Netbench
