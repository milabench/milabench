"""Slurm/srun-backed multinode execution plans.

These wrap ``milabench slurm srun`` so the main node can keep milabench/voir
metrics locally while workers are dispatched through a single srun (native or
SSH). Rank / placement is left to Slurm (``SLURM_NODEID``, etc.), not baked
into per-node argv by milabench.
"""

from __future__ import annotations

from copy import deepcopy
from typing import List

from ..utils import select_nodes
from . import (
    Command,
    ListCommand,
    SingleCmdCommand,
    TorchrunAllGPU,
    TorchrunAllNodes,
    WrapperCommand,
    clone_with,
    max_node_count,
)


def _main_node(config) -> dict:
    """Return the main node from the selected system nodes."""
    nodes = select_nodes(config["system"]["nodes"], max_node_count(config))
    for node in nodes:
        if node.get("main"):
            return node
    return nodes[0]


def _node_hostlist_token(node: dict) -> str:
    """Token suitable for ``srun -x`` / ``-w`` (name, hostname, or ip)."""
    return node.get("name") or node.get("hostname") or node["ip"]


def _with_nolog(executor: Command) -> Command:
    """Return a copy of *executor* whose pack is tagged ``nolog``."""
    config = executor.pack.config
    tags = list(config.get("tag", []))
    if "nolog" in tags:
        return executor
    run = clone_with(config, {"tag": [*tags, "nolog"]})
    return executor.copy(executor.pack.copy(run))


class SrunCommand(WrapperCommand):
    """Wrap an executor with ``milabench slurm srun``.

    The wrapped pack is tagged ``nolog`` so worker-side processes are ignored
    by milabench metrics validation (main keeps the logs).

    Arguments:
        executor: command to dispatch remotely
        exclude: hostlist string for ``-x`` (optional)
        nodelist: hostlist string for ``-w`` (optional)
    """

    def __init__(
        self,
        executor: SingleCmdCommand,
        *,
        exclude: str | None = None,
        nodelist: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(_with_nolog(executor), **kwargs)
        self.exclude = exclude
        self.nodelist = nodelist

    def resolve_exclude(self) -> str | None:
        return self.exclude

    def resolve_nodelist(self) -> str | None:
        return self.nodelist

    def _argv(self, **kwargs) -> List:
        del kwargs
        argv = ["milabench", "slurm", "srun"]
        if exclude := self.resolve_exclude():
            argv.extend(["-x", exclude])
        if nodelist := self.resolve_nodelist():
            argv.extend(["-w", nodelist])
        argv.append("--")
        return argv


class SrunExceptMain(SrunCommand):
    """Dispatch *executor* to every system node except the main one.

    Expands to::

        milabench slurm srun -x <main> -- <executor argv...>

    Main is resolved lazily from the pack's system config so the plan stays
    valid after node selection / overrides. The same argv is sent to every
    worker; per-node identity comes from Slurm (``SLURM_NODEID``), not from
    milabench rewriting the command.
    """

    def resolve_exclude(self) -> str | None:
        if self.exclude is not None:
            return self.exclude
        return _node_hostlist_token(_main_node(self.pack.config))


class ForeachSrun(ListCommand):
    """Run *executor* on the main node locally; dispatch the rest via srun.

    The local (main) copy owns milabench/voir metrics. Workers are launched
    once with :class:`SrunExceptMain` — one srun, same command everywhere.

    Override :meth:`main_executor` / :meth:`worker_executor` when the two
    sides need different processing (e.g. strip voir on workers), or override
    :meth:`executors` for a different layout.
    """

    def __init__(self, executor: Command, **kwargs) -> None:
        super().__init__(None, **kwargs)
        self.options.update(kwargs)
        self.executor = executor

    def main_executor(self) -> Command:
        """Command that runs on the main node (owns milabench metrics)."""
        return self.executor

    def worker_executor(self) -> Command:
        """Command dispatched to non-main nodes (tagged ``nolog``)."""
        config = self.executor.pack.config
        tags = [*config["tag"], "nolog"]
        run = clone_with(config, {"tag": tags})
        return self.executor.copy(self.executor.pack.copy(run))

    def single_node(self) -> Command:
        return self.main_executor()

    @property
    def executors(self):
        config = self.executor.pack.config
        nodes = select_nodes(config["system"]["nodes"], max_node_count(config))
        if len(nodes) <= 1:
            return [self.single_node()]
        return [
            self.main_executor(),
            SrunExceptMain(self.worker_executor()),
        ]

    def set_run_options(self, **kwargs):
        self.executor.set_run_options(**kwargs)
        return self

    def copy(self, pack):
        copy = deepcopy(self)
        copy.executor._set_pack(pack)
        return copy


class TorchrunSrun(ForeachSrun):
    """torchrun on main locally; same torchrun argv on workers via one srun.

    No per-worker ``--node-rank`` / ``--local-addr`` — Slurm supplies node
    identity (``SLURM_NODEID``). Use a rendezvous backend that tolerates that,
    or have the launched command read the env itself.
    """

    def __init__(self, executor: Command, *args, **kwargs) -> None:
        base_exec = TorchrunAllNodes.make_base_executor(
            TorchrunAllGPU,
            executor,
            *args,
            **kwargs,
        )
        super().__init__(base_exec)
