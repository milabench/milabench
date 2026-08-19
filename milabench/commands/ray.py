"""Ray cluster bring-up via a single symmetric srun, then run a workload on
the head (main) node.

Shape (using plain existing composition, nothing custom)::

    ListCommand(
        bringup_executor(),          # milabench slurm srun -- rayrun ...
        SequenceCommand(
            wait_executor(),          # raywait --expected N
            workload_executor(),      # local main, against the live cluster
            stop_executor(),          # milabench slurm srun -- ray stop
        ),
    )

``rayrun``/``raywait`` are benchmate console scripts (see
benchmate/ray_node.py, benchmate/ray_wait.py) -- installed alongside the
venv's other tools (like ``benchrun``), not milabench-relative script paths.

``bringup_executor()`` never returns on its own: every node's ``ray start
... --block`` runs as *that srun task's own foreground process* (see
benchmate/ray_node.py), so the whole symmetric srun step blocks for the
cluster's lifetime. ``ListCommand.execute()`` already runs its children
concurrently via ``asyncio.gather`` and waits for both -- once
``stop_executor()`` (the sequence's last step) runs ``ray stop`` on every
node, each node's local raylet/GCS dies, which is what the blocking
``ray start --block`` process on that same node is waiting on, so it returns
on its own and the bring-up step completes. No process-handle bookkeeping
or custom async orchestration needed on milabench's side; it's a plain
fork-join over primitives every other multi-step Command already uses.

Head vs. worker role is decided by each node comparing its own resolved IP
against the given head IP (see benchmate/ray_node.py) -- not by Slurm's own
node ordering/SLURM_NODEID, which is not guaranteed to put milabench's
designated main node first.

This replaced an earlier asymmetric design (detached local head + separate
`srun -x main` for workers) that was found to make freshly-joined worker
nodes get marked dead by Ray's GCS health check within seconds of joining,
reproducibly, across many runs -- see the ray_smoke diagnostic benchmark.
A single symmetric srun step (head and workers as tasks of the same step,
`ray start --block` on all of them) did not reproduce that failure in
standalone testing (ray_symmetric_test.sbatch).
"""

from __future__ import annotations

import os
from copy import deepcopy

from ..system import option
from ..utils import select_nodes
from . import (
    CmdCommand,
    Command,
    ListCommand,
    SequenceCommand,
    clone_with,
    max_node_count,
    node_address,
)
from .srun import SrunCommand, _main_node


def _ray_bin(pack) -> str:
    return f"{pack.config['dirs']['venv']}/bin/ray"


def _rayrun_bin(pack) -> str:
    return f"{pack.config['dirs']['venv']}/bin/rayrun"


def _raywait_bin(pack) -> str:
    return f"{pack.config['dirs']['venv']}/bin/raywait"


def ray_ready_marker(pack) -> str:
    runs = pack.config["dirs"]["runs"]
    job = os.environ.get("SLURM_JOB_ID", "local")
    return f"{runs}/ray-ready.{job}"


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
        init_timeout: seconds to wait for all nodes to join (default
            ``MILABENCH_RAY_INIT_TIMEOUT`` / ``ray.init_timeout``, else 600)
    """

    def __init__(
        self,
        executor: Command,
        *,
        port: int | None = None,
        init_timeout: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(None, **kwargs)
        self.options.update(kwargs)
        self.executor = executor
        self.port = port
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
        """Workload pack — not the bring-up step (see ``ListCommand.pack``)."""
        return self.executor.pack

    def bringup_executor(self) -> Command:
        """One symmetric ``rayrun`` dispatch across every node.

        Blocks for the cluster's whole lifetime (see module docstring and
        benchmate/ray_node.py) -- run concurrently with the wait/workload/
        stop sequence via ``ListCommand`` in ``executors``, not awaited on
        its own.
        """
        main = _main_node(self.executor.pack.config)
        pack = self._ray_pack()
        ip = node_address(main)
        hostname = main.get("hostname") or main.get("name") or ip
        port = self.resolve_port()
        cmd = CmdCommand(
            pack,
            _rayrun_bin(pack),
            "--ray-bin",
            _ray_bin(pack),
            "--head-ip",
            ip,
            "--head-hostname",
            hostname,
            "--port",
            str(port),
            "--ready-marker",
            ray_ready_marker(pack),
            "--timeout",
            str(self.resolve_init_timeout()),
        )
        return SrunCommand(cmd)

    def wait_executor(self) -> Command:
        """Poll until the Ray cluster has the expected number of alive nodes."""
        pack = self._ray_pack()
        main = _main_node(self.executor.pack.config)
        address = f"{node_address(main)}:{self.resolve_port()}"
        expected = len(self.selected_nodes())
        timeout = self.resolve_init_timeout()
        return CmdCommand(
            pack,
            _raywait_bin(pack),
            "--address",
            address,
            "--expected",
            str(expected),
            "--timeout",
            str(timeout),
        )

    def stop_executor(self) -> Command:
        """``ray stop`` on every node (one symmetric srun, same as bring-up).

        Kills the local raylet/GCS on each node, which is what that node's
        blocking ``ray start --block`` (from bringup_executor) is waiting
        on -- causes the bring-up step to return on its own once this runs.
        """
        pack = self._ray_pack()
        cmd = CmdCommand(pack, _ray_bin(pack), "stop")
        return SrunCommand(cmd)

    def workload_executor(self) -> Command:
        """User command on the head; owns milabench/voir metrics."""
        return self.executor

    def _head_managed_externally(self) -> bool:
        return os.environ.get("MILABENCH_RAY_HEAD_EXTERNAL") == "1"

    @property
    def executors(self):
        if self._head_managed_externally():
            return [SequenceCommand(self.wait_executor(), self.workload_executor())]

        return [
            ListCommand(
                self.bringup_executor(), 
                SequenceCommand(
                    self.wait_executor(),
                    self.workload_executor(),
                    self.stop_executor())
            )
        ]

    def set_run_options(self, **kwargs):
        self.executor.set_run_options(**kwargs)
        return self

    def copy(self, pack):
        copy = deepcopy(self)
        copy.executor._set_pack(pack)
        return copy
