"""Ray cluster bring-up via a single symmetric srun, then run a workload on
the head (main) node.

Shape (``RayCluster.execute()``; see its docstring for why this needs a
custom override rather than plain ``ListCommand``/``SequenceCommand``
composition)::

    bringup_executor()      # milabench slurm srun -- rayrun ...
    SequenceCommand(        # runs concurrently with bringup_executor() above
        wait_executor(),      # raywait --expected N
        workload_executor(),  # local main, against the live cluster
        stop_executor(),      # milabench slurm srun -- ray stop
    )
    # once the sequence above finishes, bringup_executor()'s own tracked
    # process(es) are explicitly stopped -- see execute()'s docstring

``rayrun``/``raywait`` are benchmate console scripts (see
benchmate/ray_node.py, benchmate/ray_wait.py) -- installed alongside the
venv's other tools (like ``benchrun``), not milabench-relative script paths.

``bringup_executor()`` never returns on its own: every node's ``ray start
... --block`` runs as *that srun task's own foreground process* (see
benchmate/ray_node.py), so the whole symmetric srun step blocks for the
cluster's lifetime.

``stop_executor()``'s ``ray stop`` does NOT make it return, despite the
tempting assumption that killing the local raylet/GCS is "what `ray start
--block` is waiting on": in practice ``ray start --block`` blocks waiting
for a signal *directed at itself*, not for its spawned raylet/GCS children
to die -- confirmed by a live run where `ray stop` completed cleanly
("Stopped all N Ray processes") while the bring-up step kept running,
completely unreaped, until milabench's own ~10-minute
``force_terminate_now`` timeout fallback finally killed it. So
``RayCluster.execute()`` explicitly stops bringup's own tracked process(es)
once the wait/workload/stop sequence finishes, instead of relying on it to
exit on its own.

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

import asyncio
import os
from copy import deepcopy

from ..alt_async import destroy
from ..system import option
from ..utils import assemble_options, select_nodes
from . import (
    CmdCommand,
    Command,
    ListCommand,
    SequenceCommand,
    clone_with,
    max_node_count,
    node_address,
)
from .executors import get_or_create_warden
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

    A benchmark's own config can also carry a ``ray:`` section -- the same
    ``argv``-dict idiom used by ``server:``/``client:`` elsewhere -- so
    different benchmarks sharing this same class can each tune their own
    Ray cluster instead of everyone being stuck with one process-wide env
    var::

        ray:
          port: 6380                    # same as the port= constructor arg
          init_timeout: 900             # same as the init_timeout= constructor arg
          start_args:                   # extra `ray start` argv, head + workers
            --num-cpus: 64
            --object-store-memory: 200000000000
          stop_args:                    # extra `ray stop` argv
            --grace-period: 30
          env:                          # extra env for the ray start / ray stop processes
            RAY_health_check_initial_delay_ms: 30000

    Precedence for ``port``/``init_timeout`` is: constructor argument, then
    this ``ray:`` config section, then the process-wide ``option()`` (env
    var / system config), then the hardcoded default.
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

    def _ray_config(self) -> dict:
        return self.executor.pack.config.get("ray") or {}

    def resolve_port(self) -> int:
        if self.port is not None:
            return self.port
        if (configured := self._ray_config().get("port")) is not None:
            return int(configured)
        return option("ray.port", int, 6379)

    def resolve_init_timeout(self) -> int:
        if self.init_timeout is not None:
            return self.init_timeout
        if (configured := self._ray_config().get("init_timeout")) is not None:
            return int(configured)
        return option("ray.init_timeout", int, 600)

    def _ray_start_args(self) -> list[str]:
        """Extra ``ray start`` argv (head + workers) from ``ray.start_args``."""
        return assemble_options(self._ray_config().get("start_args") or {})

    def _ray_stop_args(self) -> list[str]:
        """Extra ``ray stop`` argv from ``ray.stop_args``."""
        return assemble_options(self._ray_config().get("stop_args") or {})

    def _ray_env(self) -> dict:
        """Extra environment for the ``ray start`` / ``ray stop`` processes.

        Stringified: env values must be ``str`` for ``subprocess.Popen``,
        but plain YAML numbers/booleans parse as ``int``/``bool``.
        """
        return {k: str(v) for k, v in (self._ray_config().get("env") or {}).items()}

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
        start_args = self._ray_start_args()
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
            # rayrun's own extra_ray_args positional is argparse.REMAINDER,
            # which does not strip a leading `--` itself (see
            # benchmate/ray_node.py) -- always pass one, even with no
            # start_args, so rayrun's own parsing is uniform either way.
            "--",
            *start_args,
            env=self._ray_env(),
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

        Kills the local raylet/GCS/dashboard/etc processes it discovers on
        each node. This does NOT make bringup_executor()'s ``ray start
        --block`` return on its own (see the module docstring) -- that is
        handled separately, by ``RayCluster.execute()`` explicitly stopping
        the bring-up step's own tracked process(es) once this and the
        workload are done.

        Runs while bringup_executor()'s srun step is still alive on the same
        nodes, so it depends on the native srun exec always passing
        ``--overlap`` (see milabench/cli/slurm/srun.py::_exec_native_srun) --
        without it Slurm refuses the second step ("Requested nodes are
        busy") and this can retry indefinitely.
        """
        pack = self._ray_pack()
        cmd = CmdCommand(
            pack, _ray_bin(pack), "stop", *self._ray_stop_args(), env=self._ray_env()
        )
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

    async def _stop_bringup_now(self, pack) -> None:
        """Directly stop bring-up's own tracked process(es).

        ``ray stop`` only reaches ray's own service processes, not the
        ``ray start --block`` wrapper itself (see module docstring), so
        that wrapper has to be signaled directly once we know the workload
        and teardown are done.
        """
        for proc in pack.processes:
            if proc.poll() is None:
                await pack.message(
                    "Stopping bring-up step: workload and `ray stop` are done, "
                    "and `ray start --block` does not exit on its own."
                )
                destroy(proc)

    async def execute(
        self, phase="run", timeout=False, timeout_delay=600,
        warden=None, resource_cleaner=True, with_gpu_warden=True, **kwargs
    ):
        if self._head_managed_externally():
            return await super().execute(
                phase=phase, timeout=timeout, timeout_delay=timeout_delay,
                warden=warden, resource_cleaner=resource_cleaner,
                with_gpu_warden=with_gpu_warden, **kwargs,
            )

        bringup = self.bringup_executor()
        sequence = SequenceCommand(
            self.wait_executor(), self.workload_executor(), self.stop_executor()
        )
        run_kwargs = {
            **self._kwargs,
            **kwargs,
            "phase": phase,
            "timeout": timeout,
            "timeout_delay": timeout_delay,
        }

        with get_or_create_warden(
            warden, with_gpu_warden=with_gpu_warden, resource_cleaner=resource_cleaner
        ) as warden:
            bringup_task = asyncio.create_task(bringup.execute(**run_kwargs, warden=warden))
            try:
                error_count = await sequence.execute(**run_kwargs, warden=warden)
            finally:
                await self._stop_bringup_now(bringup.pack)
                try:
                    await asyncio.wait_for(bringup_task, timeout=30)
                except Exception:
                    bringup_task.cancel()

        return error_count

    def set_run_options(self, **kwargs):
        self.executor.set_run_options(**kwargs)
        return self

    def copy(self, pack):
        copy = deepcopy(self)
        copy.executor._set_pack(pack)
        return copy
