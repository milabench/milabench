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
from copy import deepcopy
from pathlib import Path

from milabench.commands import PackCommand, TorchrunAllGPU, TorchrunAllNodes, max_node_count
from milabench.commands.srun import ForeachSrun, SrunCommand, _node_hostlist_token
from milabench.utils import assemble_options, select_nodes

sys.path.insert(0, str(Path(__file__).resolve().parent))
import placement as placement_lib  # noqa: E402


def _load_slime():
    path = Path(__file__).resolve().parent.parent / "slime" / "benchfile.py"
    spec = importlib.util.spec_from_file_location("milabench_slime_benchfile", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


Slime = _load_slime().Slime


class SrunOneNode(SrunCommand):
    """``srun -w <host>`` on exactly one node.

    ``milabench slurm srun`` forwards its arguments to Slurm's srun, but its own parser rejects the node-count flags
    and only ``-x`` gets a node count: with ``-w`` alone the step spans the whole job (default node count) and also
    starts the worker on the main node (rank 4 and rank 0 on the same GPU: "Duplicate GPU detected").
    So call srun directly, with what ``milabench slurm srun`` would exec (``--overlap``)."""

    def _argv(self, **kwargs):
        host = self.resolve_nodelist()
        return ["srun", "--overlap", "--nodes=1", "--ntasks=1", "--ntasks-per-node=1", "-w", host, "--"]


class PlacedTorchrunSrun(ForeachSrun):
    """torchrun with the node order of ``system.nodes``: node i runs on host i with ``--node-rank=i``.

    ``TorchrunSrun`` starts all the workers with one ``srun -x main`` and takes their rank from ``SLURM_NODEID``,
    which is Slurm's own (sorted) order: it cannot follow a placement. Here every worker is its own
    ``srun -w <host>`` step with an explicit rank (benchrun keeps an explicit ``--node-rank``)."""

    def __init__(self, executor, *args, **kwargs) -> None:
        base_exec = TorchrunAllNodes.make_base_executor(TorchrunAllGPU, executor, *args, **kwargs)
        super().__init__(base_exec)

    def main_executor(self):
        main = deepcopy(self.executor)
        main.wrapper_argv = (*main.wrapper_argv, "--node-rank=0")
        return main

    @property
    def executors(self):
        config = self.executor.pack.config
        nodes = select_nodes(config["system"]["nodes"], max_node_count(config))
        if len(nodes) <= 1:
            return [self.single_node()]
        executors = [self.main_executor()]
        for rank, node in enumerate(nodes[1:], start=1):
            worker = self.worker_executor()
            worker.wrapper_argv = (*worker.wrapper_argv, f"--node-rank={rank}")
            executors.append(SrunOneNode(worker, nodelist=_node_hostlist_token(node)))
        return executors


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

    def placement(self) -> dict:
        """``placement:`` with the size of an expert group, in nodes (EP size / GPUs per node)."""
        placement = dict(self.config.get("placement") or {})
        gpus = int(placement.get("gpus_per_node", 4))
        ep = int((self.config.get("argv") or {}).get("--expert-model-parallel-size", 1))
        placement.setdefault("group_nodes", max(1, ep // gpus))
        return placement

    def place_nodes(self):
        """Order (and, for the aligned / crossed / mixed modes, select) ``system.nodes``: the rank of a node is its position."""
        system = self.config.get("system") or {}
        nodes = system.get("nodes")
        if not nodes or len(nodes) <= 1:
            return
        ordered = placement_lib.order_nodes(nodes, self.placement())
        want = self.num_machines
        if len(ordered) < want:
            raise RuntimeError(
                f"placement.nodes={self.placement().get('nodes')} gives {len(ordered)} nodes, the entry wants {want} "
                f"(num_machines): nodes were excluded, set num_machines to {len(ordered)} for all the runs you compare"
            )
        system["nodes"] = ordered

    def build_run_plan(self):
        self.place_nodes()
        # voir main.py ... is not used: main.py reports through benchmate.monitor.setupvoir() (stdout smuggling)
        workload = PackCommand(self, lazy=True)
        # torchrun on the main node locally; one explicit `srun -w host --node-rank=i` per other node, in the order
        # of system.nodes (the placement)
        if self.num_machines > 1:
            return PlacedTorchrunSrun(workload).use_stdout()
        return TorchrunAllGPU(workload).use_stdout()


__pack__ = MegatronGlm5
