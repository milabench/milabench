"""slime pack: Megatron-LM training (and GRPO) driven by THUDM/slime on Ray.

The slime command line is assembled from the bench config, like the vllm pack
does with ``server``/``client``:

    model:
      hf: <huggingface repo>
      argv: {...}      # Megatron architecture flags (also used to convert the checkpoint)
    argv: {...}        # everything else passed to slime's train.py

Multi-node runs use ``RayCluster`` (one symmetric srun brings Ray up on every
node, the driver then runs on the main node). Single-node runs let the driver
start a local Ray.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import milabench.commands as cmd
from milabench.commands.ray import RayCluster
from milabench.pack import Package
from milabench.utils import assemble_options


SLIME_REPO = "https://github.com/THUDM/slime.git"
# Keep in sync with requirements.in
SLIME_COMMIT = "a4202e0595467d9374d9a68d45a34f45c3f8d230"
MEGATRON_REPO = "https://github.com/NVIDIA/Megatron-LM.git"
MEGATRON_COMMIT = "4b4acac9a1d28ea6829c8d4f566d75698a21249d"
# slime ships patches against the Megatron-LM commit its docker image uses
MEGATRON_PATCHES = ("megatron.patch", "megatron-sglang-aligned.patch")
PATCH_VERSION = "v0.5.21"
TE_REQUIREMENT = "transformer-engine[pytorch]==2.18.0"
BUILD_REQUIRES = ("setuptools", "wheel", "pybind11", "ninja", "packaging", "numpy")


class Slime(Package):
    base_requirements = "requirements.in"
    prepare_script = "prepare.py"
    main_script = "main.py"

    def clone(self):
        # train.py and tools/ are not part of slime's wheel, and megatron.training
        # is not part of megatron-core: both are used straight from a checkout.
        slime = self.dirs.code / "slime_src"
        megatron = self.dirs.code / "Megatron-LM"

        if not slime.exists():
            slime.clone_subtree(SLIME_REPO, SLIME_COMMIT)

        if not megatron.exists():
            megatron.clone_subtree(MEGATRON_REPO, MEGATRON_COMMIT)
            for name in MEGATRON_PATCHES:
                patch = slime / "docker" / "patch" / PATCH_VERSION / name
                if patch.exists():
                    subprocess.run(["git", "apply", str(patch)], cwd=megatron, check=True)

    async def install(self):
        self.clone()
        # megatron-core / transformer-engine are built with --no-build-isolation
        # and their setup.py import these at metadata time
        await self.pip_install(*BUILD_REQUIRES, build_isolation=True)
        await super().install()
        # TE compiles against the installed torch/CUDA, so it needs torch to be
        # importable already: it cannot be part of the batch that installs torch
        await self.pip_install(TE_REQUIREMENT, build_isolation=False)

    @property
    def num_machines(self) -> int:
        return max(1, int(self.config.get("num_machines", 1)))

    def uses_ray_cluster(self) -> bool:
        return self.num_machines > 1 or bool(self.config.get("ray", {}).get("cluster"))

    @property
    def argv(self):
        model = (self.config.get("model") or {}).get("argv") or {}
        return assemble_options(model) + assemble_options(self.config.get("argv", {}))

    def make_env(self):
        env = super().make_env()
        paths = [str(self.dirs.code), str(self.dirs.code / "slime_src"), str(self.dirs.code / "Megatron-LM")]
        if prev := env.get("PYTHONPATH"):
            paths.append(prev)
        env["PYTHONPATH"] = os.pathsep.join(paths)
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        if self.uses_ray_cluster():
            env["RAY_ADDRESS"] = "auto"
        self.add_nvidia_wheel_paths(env)
        return env

    def add_nvidia_wheel_paths(self, env):
        # The cluster's CUDA module has no cudnn/nccl headers, but torch's nvidia-*
        # wheels ship them: point the TE build (cudnn.h, nccl.h) at those.
        site = sorted(Path(str(self.dirs.venv)).glob("lib/python*/site-packages/nvidia"))
        includes = [str(d) for s in site for d in sorted(s.glob("*/include")) if d.is_dir()]
        libs = [str(d) for s in site for d in sorted(s.glob("*/lib")) if d.is_dir()]
        for var, paths in (("CPATH", includes), ("LIBRARY_PATH", libs)):
            if paths:
                env[var] = os.pathsep.join([*paths, *filter(None, [env.get(var) or os.environ.get(var)])])
        for d in includes:
            if (Path(d) / "cudnn.h").exists():
                env.setdefault("CUDNN_PATH", str(Path(d).parent))
            if (Path(d) / "nccl.h").exists():
                env.setdefault("NCCL_HOME", str(Path(d).parent))

    def build_run_plan(self):
        # No VoirCommand: main.py reports through benchmate.monitor.setupvoir(), which
        # needs use_stdout() (see diagnostic/ray_smoke).
        workload = cmd.PackCommand(self, lazy=True).use_stdout()
        if self.uses_ray_cluster():
            return RayCluster(workload)
        return workload


__pack__ = Slime
