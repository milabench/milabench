"""megatron pack: GLM-5 training on plain Megatron-LM (torchrun/srun, no Ray, no rollout), with frozen parameters.

Same install as the ``slime`` pack (the GLM-5 layer spec, DSA indexer and sparse MLA live in slime's ``slime_plugins``;
slime, Megatron-LM, TE and sglang are cloned/installed the same way). What differs:

* the launcher is torchrun over srun (like torchtitan) and ``main.py`` drives Megatron's ``pretrain`` with random
  packed (THD) data: no checkpoint, no dataset, no tokenizer,
* ``frozen:`` selects which parameters get ``requires_grad = False`` (see main.py),
* ``placement:`` controls the node order / Megatron rank order, see placement.py.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

from milabench.commands import PackCommand, TorchrunAllGPU
from milabench.commands.srun import TorchrunSrun
from milabench.utils import assemble_options

sys.path.insert(0, str(Path(__file__).resolve().parent))
import placement as placement_lib  # noqa: E402


def _load_slime():
    path = Path(__file__).resolve().parent.parent / "slime" / "benchfile.py"
    spec = importlib.util.spec_from_file_location("milabench_slime_benchfile", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


Slime = _load_slime().Slime


class MegatronGlm5(Slime):
    base_requirements = "requirements.in"
    prepare_script = "prepare.py"
    main_script = "main.py"

    def uses_ray_cluster(self) -> bool:
        return False

    @property
    def argv(self):
        # Megatron flags only: everything slime-specific lives in the config (frozen:, placement:, model.hf)
        model = (self.config.get("model") or {}).get("argv") or {}
        argv = assemble_options(model) + assemble_options(self.config.get("argv", {}))
        placement = self.config.get("placement") or {}
        if placement.get("order") == "tp-cp-ep-pp-dp":
            argv.append("--use-tp-pp-dp-mapping")
        return argv

    def make_env(self):
        env = super().make_env()
        env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
        env.setdefault("VOIR_PLAIN_SMUGGLE", "1")
        placement = self.config.get("placement") or {}
        # NCCL / network knobs of the cluster (NCCL_TOPO_FILE, NCCL_CROSS_NIC, NCCL_IB_HCA, ...)
        for key, value in (placement.get("nccl") or {}).items():
            env[str(key)] = str(value)
        return env

    def place_nodes(self):
        """Reorder ``system.nodes`` by ``placement.nodes``: torchrun's node rank is the position in that list."""
        placement = self.config.get("placement") or {}
        system = self.config.get("system") or {}
        nodes = system.get("nodes")
        if not nodes or len(nodes) <= 1:
            return
        system["nodes"] = placement_lib.order_nodes(nodes, placement)

    def build_run_plan(self):
        self.place_nodes()
        # voir main.py ... is not used: main.py reports through benchmate.monitor.setupvoir() (stdout smuggling)
        workload = PackCommand(self, lazy=True)
        # torchrun on the main node locally, the same argv on the others through one srun (node rank from SLURM_NODEID)
        if self.num_machines > 1:
            return TorchrunSrun(workload).use_stdout()
        return TorchrunAllGPU(workload).use_stdout()


__pack__ = MegatronGlm5
