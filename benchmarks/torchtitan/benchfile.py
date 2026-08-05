"""Torchtitan milabench pack — torchrun/srun launcher for ``torchtitan.train``."""

from __future__ import annotations

import os
from copy import deepcopy

from milabench.commands import (
    PackCommand,
    TorchrunAllGPU,
    TorchrunAllNodes,
    VoirCommand,
    WorkingDir,
)
from milabench.commands.srun import ForeachSrun
from milabench.pack import Package


class TorchtitanRun(TorchrunAllGPU):
    """Always wrap with benchrun/torchrun (even on a single GPU)."""

    def should_wrap(self):
        return True


class TorchtitanSrun(ForeachSrun):
    """Main locally; workers via ``srun -x main`` (same layout as TorchtuneSrun)."""

    def __init__(self, executor, *args, **kwargs) -> None:
        base_exec = TorchrunAllNodes.make_base_executor(
            TorchtitanRun,
            executor,
            *args,
            **kwargs,
        )
        super().__init__(WorkingDir(base_exec))

    def main_executor(self):
        main = deepcopy(self.executor)
        main.wrapper_argv = (*main.wrapper_argv, "--node-rank=0")
        return main


class Torchtitan(Package):
    base_requirements = "requirements.in"
    prepare_script = "prepare.py"
    main_script = "main.py"

    def make_env(self):
        env = super().make_env()
        code = str(self.dirs.code)
        prev = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = code if not prev else f"{code}{os.pathsep}{prev}"
        env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
        return env

    def build_run_plan(self):
        # voir main.py … so voirfile.py instruments the trainer; torchrun on top.
        plan = VoirCommand(PackCommand(self, lazy=True))
        return TorchtitanSrun(plan).use_stdout()


__pack__ = Torchtitan
