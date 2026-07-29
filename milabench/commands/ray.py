"""Ray cluster bring-up via srun, then run a workload on the head (main) node.

Flow (sequential)::

    ray start --head                 # local main
    milabench slurm srun -x main -- ray start --address=...   # all workers
    <executor>                       # local main, against the live cluster

Per-node identity / placement is left to Slurm; milabench does not fan out
worker-specific argv.
"""

from __future__ import annotations

from copy import deepcopy

from ..system import option
from ..utils import select_nodes
from . import (
    CmdCommand,
    Command,
    SequenceCommand,
    clone_with,
    max_node_count,
    node_address,
)
from .srun import SrunExceptMain, _main_node


def _ray_bin(pack) -> str:
    return f"{pack.config['dirs']['venv']}/bin/ray"


class RayCluster(SequenceCommand):
    """Bring up a Ray cluster, then run *executor* on the main (head) node.

    Arguments:
        executor: workload to run on the head once the cluster is up
        port: Ray GCS port (default ``MILABENCH_RAY_PORT`` / ``ray.port``, else 6379)
        head_args: extra argv for ``ray start --head``
        worker_args: extra argv for worker ``ray start --address=...``
    """

    def __init__(
        self,
        executor: Command,
        *,
        port: int | None = None,
        head_args: tuple[str, ...] = (),
        worker_args: tuple[str, ...] = (),
        **kwargs,
    ) -> None:
        super().__init__(None, **kwargs)
        self.options.update(kwargs)
        self.executor = executor
        self.port = port
        self.head_args = tuple(head_args)
        self.worker_args = tuple(worker_args)

    def resolve_port(self) -> int:
        if self.port is not None:
            return self.port
        return option("ray.port", int, 6379)

    def _nolog_pack(self, *extra_tags):
        config = self.executor.pack.config
        tags = [*config["tag"], *extra_tags, "nolog"]
        return self.executor.pack.copy(clone_with(config, {"tag": tags}))

    def head_executor(self) -> Command:
        """``ray start --head`` on the main node (local)."""
        main = _main_node(self.executor.pack.config)
        pack = self._nolog_pack("ray-head")
        ip = node_address(main)
        port = self.resolve_port()
        return CmdCommand(
            pack,
            _ray_bin(pack),
            "start",
            "--head",
            f"--node-ip-address={ip}",
            f"--port={port}",
            *self.head_args,
        )

    def worker_executor(self) -> Command:
        """``ray start --address=...`` (same argv on every non-main node)."""
        main = _main_node(self.executor.pack.config)
        pack = self._nolog_pack("ray-worker")
        address = f"{node_address(main)}:{self.resolve_port()}"
        return CmdCommand(
            pack,
            _ray_bin(pack),
            "start",
            f"--address={address}",
            *self.worker_args,
        )

    def workload_executor(self) -> Command:
        """User command on the head; owns milabench/voir metrics."""
        return self.executor

    @property
    def executors(self):
        config = self.executor.pack.config
        nodes = select_nodes(config["system"]["nodes"], max_node_count(config))

        steps = [self.head_executor()]
        if len(nodes) > 1:
            steps.append(SrunExceptMain(self.worker_executor()))
        steps.append(self.workload_executor())
        return steps

    def set_run_options(self, **kwargs):
        self.executor.set_run_options(**kwargs)
        return self

    def copy(self, pack):
        copy = deepcopy(self)
        copy.executor._set_pack(pack)
        return copy
