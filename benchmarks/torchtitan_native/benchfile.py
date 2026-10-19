"""Native TorchTitan pre-training throughput benchmark (Qwen3.8).

Separate from ``benchmarks/torchtitan`` (Hugging Face modeling backend, older
TorchTitan pin): this pack tracks a TorchTitan commit with native Qwen3.8
support and the PyTorch nightly it requires, in its own install group.
"""

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
from milabench.pack import Package, install_benchmate


class NativeTorchrun(TorchrunAllGPU):
    """Always go through benchrun/torchrun, even on a single GPU."""

    def should_wrap(self):
        return True


class NativeTorchrunSrun(ForeachSrun):
    """Main node locally; workers through one ``srun -x main``.

    Workers get ``--node-rank`` from ``SLURM_NODEID`` (see benchrun).
    """

    def __init__(self, executor, *args, **kwargs) -> None:
        base_exec = TorchrunAllNodes.make_base_executor(
            NativeTorchrun,
            executor,
            *args,
            **kwargs,
        )
        super().__init__(WorkingDir(base_exec))

    def main_executor(self):
        main = deepcopy(self.executor)
        main.wrapper_argv = (*main.wrapper_argv, "--node-rank=0")
        return main


class TorchtitanNative(Package):
    base_requirements = "requirements.in"
    prepare_script = None
    main_script = "main.py"

    async def install(self):
        # The torch nightly index is declared inside requirements.in, which the
        # TOML/legacy install paths do not handle; install it directly.
        # milabench installs without build isolation, so seed build backends
        # first (torchtitan's git build needs setuptools).
        await self.pip_install("setuptools", "wheel", build_isolation=True)
        await self.pip_install("-r", self.dirs.code / "requirements.in")
        await install_benchmate(self)

    async def prepare(self):
        # Random tokens, random init: nothing to download or stage.
        pass

    def make_env(self):
        env = super().make_env()
        code = str(self.dirs.code)
        prev = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = code if not prev else f"{code}{os.pathsep}{prev}"
        env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
        env.setdefault("VOIR_PLAIN_SMUGGLE", "1")
        # The trainer owns the stop condition; never let voir early-stop it.
        env["BENCHMATE_OBSERVATION_COUNT"] = str(10**9)
        # Compute nodes have no internet access.
        env.setdefault("HF_HUB_OFFLINE", "1")
        env.setdefault("HF_DATASETS_OFFLINE", "1")
        return env

    def build_run_plan(self):
        plan = VoirCommand(PackCommand(self, lazy=True))
        return NativeTorchrunSrun(plan).use_stdout()


__pack__ = TorchtitanNative
