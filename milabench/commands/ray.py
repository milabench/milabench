"""Ray cluster bring-up via srun, then run a workload on the head (main) node.

Flow (sequential)::

    ray start --head                 # local main
    milabench slurm srun -x main -- ray start --address=...   # all workers
    wait until cluster size == expected
    <executor>                       # local main, against the live cluster
    milabench slurm srun -x main -- ray stop   # workers
    ray stop                         # head

Per-node identity / placement is left to Slurm; milabench does not fan out
worker-specific argv.
"""

from __future__ import annotations

from copy import deepcopy
from textwrap import dedent

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


def _python_bin(pack) -> str:
    return f"{pack.config['dirs']['venv']}/bin/python"


class RayCluster(SequenceCommand):
    """Bring up a Ray cluster, then run *executor* on the main (head) node.

    Arguments:
        executor: workload to run on the head once the cluster is up
        port: Ray GCS port (default ``MILABENCH_RAY_PORT`` / ``ray.port``, else 6379)
        head_args: extra argv for ``ray start --head``
        worker_args: extra argv for worker ``ray start --address=...``
        init_timeout: seconds to wait for all nodes to join (default
            ``MILABENCH_RAY_INIT_TIMEOUT`` / ``ray.init_timeout``, else 600)
    """

    def __init__(
        self,
        executor: Command,
        *,
        port: int | None = None,
        head_args: tuple[str, ...] = (),
        worker_args: tuple[str, ...] = (),
        init_timeout: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(None, **kwargs)
        self.options.update(kwargs)
        self.executor = executor
        self.port = port
        self.head_args = tuple(head_args)
        self.worker_args = tuple(worker_args)
        self.init_timeout = init_timeout

    def resolve_port(self) -> int:
        if self.port is not None:
            return self.port
        return option("ray.port", int, 6379)

    def resolve_init_timeout(self) -> int:
        if self.init_timeout is not None:
            return self.init_timeout
        return option("ray.init_timeout", int, 600)

    def selected_nodes(self):
        config = self.executor.pack.config
        return select_nodes(config["system"]["nodes"], max_node_count(config))

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

    def wait_executor(self) -> Command:
        """Poll until the Ray cluster has the expected number of alive nodes."""
        pack = self._nolog_pack("ray-wait")
        expected = len(self.selected_nodes())
        timeout = self.resolve_init_timeout()
        script = dedent(
            f"""\
            import sys
            import time

            import ray

            expected = {expected}
            timeout = {timeout}
            ray.init(address="auto")
            deadline = time.time() + timeout
            alive = 0
            while time.time() < deadline:
                alive = sum(1 for n in ray.nodes() if n.get("Alive"))
                if alive >= expected:
                    print(f"Ray cluster ready: {{alive}}/{{expected}}")
                    sys.exit(0)
                print(f"Waiting for Ray workers: {{alive}}/{{expected}}")
                time.sleep(5)
            print(f"Timed out waiting for Ray cluster: {{alive}}/{{expected}}")
            sys.exit(1)
            """
        )
        return CmdCommand(pack, _python_bin(pack), "-c", script)

    def head_stop_executor(self) -> Command:
        """``ray stop`` on the main node (local)."""
        pack = self._nolog_pack("ray-stop-head")
        return CmdCommand(pack, _ray_bin(pack), "stop")

    def worker_stop_executor(self) -> Command:
        """``ray stop`` on every non-main node."""
        pack = self._nolog_pack("ray-stop-worker")
        return CmdCommand(pack, _ray_bin(pack), "stop")

    def workload_executor(self) -> Command:
        """User command on the head; owns milabench/voir metrics."""
        return self.executor

    @property
    def executors(self):
        nodes = self.selected_nodes()

        steps = [self.head_executor()]
        if len(nodes) > 1:
            steps.append(SrunExceptMain(self.worker_executor()))
        steps.append(self.wait_executor())
        steps.append(self.workload_executor())
        if len(nodes) > 1:
            steps.append(SrunExceptMain(self.worker_stop_executor()))
        steps.append(self.head_stop_executor())
        return steps

    def set_run_options(self, **kwargs):
        self.executor.set_run_options(**kwargs)
        return self

    def copy(self, pack):
        copy = deepcopy(self)
        copy.executor._set_pack(pack)
        return copy
