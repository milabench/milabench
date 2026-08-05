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

    def _argv(self, **kwargs):
        # Prefer explicit nproc from configured devices (smoke uses devices: [0]).
        # Parent uses --nproc-per-node=gpu which ignores a 1-entry devices list.
        # milabench also rewrites devices to all GPUs, so key off smoke tags.
        tags = list(self.pack.config.get("tags") or [])
        name = str(self.pack.config.get("name", ""))
        if "smoke" in tags or "smoke" in name:
            nproc = 1
        else:
            devices = self.pack.config.get("devices") or [0]
            nproc = max(1, len(devices))
        argv = list(super()._argv(**kwargs))
        replaced = False
        out = []
        for a in argv:
            s = str(a)
            if s.startswith("--nproc-per-node="):
                out.append(f"--nproc-per-node={nproc}")
                replaced = True
            else:
                out.append(a)
        if not replaced:
            out.append(f"--nproc-per-node={nproc}")
        return out


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
        # Host has no A/AAAA for hostname; c10d getaddrinfo hangs without this.
        env.setdefault("MASTER_ADDR", "127.0.0.1")
        env.setdefault("GLOO_SOCKET_IFNAME", "lo")
        env.setdefault("NCCL_SOCKET_IFNAME", "lo")
        env.setdefault("VOIR_PLAIN_SMUGGLE", "1")
        # Pin smoke to 1 GPU (benchrun uses --nproc-per-node=gpu).
        tags = list(self.config.get("tags") or [])
        name = str(self.config.get("name", "") or getattr(self, "name", ""))
        if "smoke" in tags or "smoke" in name:
            env["CUDA_VISIBLE_DEVICES"] = "0"
        # Debug: always expose which devices we chose
        env.setdefault("TORCH_DISTRIBUTED_DEBUG", "OFF")
        return env

    def build_run_plan(self):
        # voir main.py … so voirfile.py instruments the trainer; torchrun on top.
        plan = VoirCommand(PackCommand(self, lazy=True))
        return TorchtitanSrun(plan).use_stdout()


__pack__ = Torchtitan
