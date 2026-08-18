"""Ray cluster bring-up via srun, then run a workload on the head (main) node.

Flow (sequential)::

    scripts/ray_start_head.sh        # detached ``ray start --head --block`` (local main)
    milabench slurm srun -x main -- ray start --address=...   # all workers
    wait until cluster size == expected
    <executor>                       # local main, against the live cluster
    milabench slurm srun -x main -- ray stop   # workers
    ray stop                         # head

Per-node identity / placement is left to Slurm; milabench does not fan out
worker-specific argv.
"""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path

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


def ray_wait_script() -> str:
    path = Path(__file__).resolve().parent.parent / "scripts" / "ray_wait.py"
    if not path.is_file():
        raise FileNotFoundError(f"Ray wait script missing: {path}")
    return str(path)


def ray_start_head_script() -> str:
    path = Path(__file__).resolve().parent.parent / "scripts" / "ray_start_head.sh"
    if not path.is_file():
        raise FileNotFoundError(f"Ray head start script missing: {path}")
    return str(path)


def ray_head_pidfile(pack) -> str:
    runs = pack.config["dirs"]["runs"]
    job = os.environ.get("SLURM_JOB_ID", "local")
    return f"{runs}/ray-head.{job}.pid"


_RAY_TAG = "ray"
_NOLOG_TAG = "nolog"


def _ray_step_tags(config) -> list[str]:
    """Build ``[<bench>, …job suffixes…, ray, nolog]`` without stacking ray tags."""
    name = config.get("name") or (config.get("tag") or [""])[0]
    skip = {_RAY_TAG, _NOLOG_TAG}
    skip |= {t for t in config.get("tag", []) if t.startswith("ray")}
    suffixes = [t for t in config.get("tag", []) if t not in skip and t != name]
    return [name, *suffixes, _RAY_TAG, _NOLOG_TAG]


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

    def _ray_pack(self):
        config = self.executor.pack.config
        return self.executor.pack.copy(
            clone_with(config, {"tag": _ray_step_tags(config)})
        )

    @property
    def pack(self):
        """Workload pack — not the first Ray bring-up step (see ``ListCommand.pack``)."""
        return self.executor.pack

    def head_executor(self) -> Command:
        """Detached ``ray start --head --block`` on the main node (local).

        See Ray's Slurm guide: the head must stay alive while workers join.
        """
        main = _main_node(self.executor.pack.config)
        pack = self._ray_pack()
        ip = node_address(main)
        port = self.resolve_port()
        pidfile = ray_head_pidfile(pack)
        return CmdCommand(
            pack,
            "/bin/bash",
            ray_start_head_script(),
            _ray_bin(pack),
            "--head",
            f"--node-ip-address={ip}",
            f"--port={port}",
            *self.head_args,
            env={"MILABENCH_RAY_HEAD_PIDFILE": pidfile},
        )

    def worker_executor(self) -> Command:
        """``ray start --address=...`` (same argv on every non-main node)."""
        main = _main_node(self.executor.pack.config)
        pack = self._ray_pack()
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
        pack = self._ray_pack()
        main = _main_node(self.executor.pack.config)
        address = f"{node_address(main)}:{self.resolve_port()}"
        expected = len(self.selected_nodes())
        timeout = self.resolve_init_timeout()
        return CmdCommand(
            pack,
            _python_bin(pack),
            ray_wait_script(),
            "--address",
            address,
            "--expected",
            str(expected),
            "--timeout",
            str(timeout),
        )

    def head_stop_executor(self) -> Command:
        """``ray stop`` on the main node (local)."""
        pack = self._ray_pack()
        pidfile = ray_head_pidfile(pack)
        return CmdCommand(
            pack,
            "/bin/bash",
            "-c",
            f"{_ray_bin(pack)} stop; rm -f {pidfile}",
        )

    def worker_stop_executor(self) -> Command:
        """``ray stop`` on every non-main node."""
        pack = self._ray_pack()
        return CmdCommand(pack, _ray_bin(pack), "stop")

    def workload_executor(self) -> Command:
        """User command on the head; owns milabench/voir metrics."""
        return self.executor

    def _head_managed_externally(self) -> bool:
        return os.environ.get("MILABENCH_RAY_HEAD_EXTERNAL") == "1"

    @property
    def executors(self):
        nodes = self.selected_nodes()
        external_head = self._head_managed_externally()

        steps = []
        if not external_head:
            steps.append(self.head_executor())
        if len(nodes) > 1:
            steps.append(SrunExceptMain(self.worker_executor()))
        steps.append(self.wait_executor())
        steps.append(self.workload_executor())
        if len(nodes) > 1:
            steps.append(SrunExceptMain(self.worker_stop_executor()))
        if not external_head:
            steps.append(self.head_stop_executor())
        return steps

    def set_run_options(self, **kwargs):
        self.executor.set_run_options(**kwargs)
        return self

    def copy(self, pack):
        copy = deepcopy(self)
        copy.executor._set_pack(pack)
        return copy
