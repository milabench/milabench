"""Run a command on every node from the milabench system config over SSH.

When a native ``srun`` binary is available, the Python process is replaced with
it so Slurm's real launcher is used unchanged.
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from typing import Optional

import argparse
from argklass.arguments import argument
from argklass.command import Command
from voir.proc import LogEntry, Multiplexer

from ...system import build_system_config
from .system import expand_node_list

SSH_OPTIONS = (
    "-oCheckHostIP=no",
    "-oPasswordAuthentication=no",
    "-oStrictHostKeyChecking=no",
)


class SrunLogEntry(LogEntry):
    def __init__(self, node=None, **rest):
        super().__init__(**rest)
        self.node = node


def _exec_native_srun() -> None:
    """Replace this process with the system ``srun`` and the original CLI args."""
    srun = shutil.which("srun")
    if srun is None:
        return

    try:
        idx = sys.argv.index("srun")
    except ValueError:
        return

    os.execvp(srun, [srun, *sys.argv[idx + 1 :]])


def _parse_hostlist(hostlist: Optional[str]) -> set[str]:
    if not hostlist:
        return set()
    return set(expand_node_list(hostlist.replace(" ", "")))


def _normalize_command(command: list[str]) -> list[str]:
    cmd = list(command or [])
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    return cmd


def _load_system() -> dict:
    """Load system config from ``$MILABENCH_SYSTEM``."""
    path = os.getenv("MILABENCH_SYSTEM")
    if not path:
        print("error: MILABENCH_SYSTEM is not set", file=sys.stderr)
        sys.exit(2)

    config = build_system_config(path, gpu=False)
    system = config.get("system") or {}
    if not system.get("nodes"):
        print(f"error: no nodes defined in system config: {path}", file=sys.stderr)
        sys.exit(2)
    return system


def _node_identifiers(node: dict) -> set[str]:
    return {
        value
        for key in ("name", "ip", "hostname")
        if (value := node.get(key))
    }


def _filter_nodes(
    nodes: list[dict],
    *,
    excluded: set[str] | None = None,
    nodelist: set[str] | None = None,
) -> list[dict]:
    selected = list(nodes)
    if nodelist:
        selected = [n for n in selected if _node_identifiers(n) & nodelist]
    if excluded:
        selected = [n for n in selected if not (_node_identifiers(n) & excluded)]
    return selected


def _node_rank(all_nodes: list[dict], node: dict) -> int:
    """Stable rank of *node* in the full system node list (Slurm-like)."""
    target = _node_identifiers(node)
    for i, candidate in enumerate(all_nodes):
        if _node_identifiers(candidate) & target:
            return i
    return 0


def _ssh_argv(node: dict, command: list[str], sshkey: Optional[str] = None) -> list[str]:
    """Build an ssh argv from a milabench system node entry."""
    host = node["ip"]
    user = node.get("user")
    port = node.get("sshport", 22)
    key = node.get("key") or sshkey

    target = f"{user}@{host}" if user else host
    argv = ["ssh", *SSH_OPTIONS, "-p", str(port)]
    if key:
        argv.append(f"-i{key}")
    argv.extend([target, "--", *command])
    return argv


def run_on_nodes(
    nodes: list[dict],
    command: list[str],
    sshkey: Optional[str] = None,
    *,
    all_nodes: list[dict] | None = None,
) -> int:
    """SSH to each system node and run *command* via voir's Multiplexer.

    Injects ``SLURM_NODEID`` / ``SLURM_NNODES`` into the remote command so
    callers (e.g. torchrun) can derive a stable per-node rank.
    """
    rank_source = all_nodes or nodes
    nnodes = len(rank_source)
    mp = Multiplexer(timeout=None, constructor=SrunLogEntry)

    for node in nodes:
        label = node.get("name") or node["ip"]
        rank = _node_rank(rank_source, node)
        remote_cmd = [
            "env",
            f"SLURM_NODEID={rank}",
            f"SLURM_PROCID={rank}",
            f"SLURM_NNODES={nnodes}",
            *command,
        ]
        argv = _ssh_argv(node, remote_cmd, sshkey=sshkey)
        print(f"[srun] {label}: {' '.join(argv)}", file=sys.stderr)
        mp.start(argv, info={"node": label})

    return_codes: dict[str, int] = {}

    for entry in mp:
        node = entry.node or "?"
        match entry.event:
            case "line":
                stream = sys.stderr if entry.pipe == "stderr" else sys.stdout
                text = (
                    entry.data
                    if isinstance(entry.data, str)
                    else entry.data.decode("utf8", "replace")
                )
                if not text.endswith("\n"):
                    text = text + "\n"
                print(f"[{node}] {text}", end="", file=stream, flush=True)
            case "end":
                rc = int(entry.data.get("return_code", 0))
                return_codes[node] = rc
                if rc != 0:
                    print(f"[srun] {node}: exited with {rc}", file=sys.stderr)
            case "start":
                pass
            case _:
                if entry.data is not None:
                    print(f"[{node}] {entry.event}: {entry.data}", file=sys.stderr)

    if not return_codes:
        return 1
    return max(return_codes.values())


class SlurmRun(Command):
    """Run a command on every node from the milabench system config."""

    name = "srun"

    # fmt: off
    @dataclass
    class Arguments:
        """Run a command on system nodes over SSH."""
        command : list[str]     = argument(nargs=argparse.REMAINDER)  # Command to run
        exclude : Optional[str] = argument("-x", default=None)        # Hosts to exclude (name/ip hostlist)
        nodelist: Optional[str] = argument("-w", default=None)        # Hosts to include (name/ip hostlist)
    # fmt: on

    @staticmethod
    def execute(args):
        # Prefer Slurm's real srun when present (replaces this process).
        _exec_native_srun()

        command = _normalize_command(args.command)
        if not command:
            print("error: missing command to run", file=sys.stderr)
            return 2

        system = _load_system()
        all_nodes = system["nodes"]
        excluded = _parse_hostlist(args.exclude)
        included = _parse_hostlist(args.nodelist)
        nodes = _filter_nodes(all_nodes, excluded=excluded, nodelist=included or None)

        if not nodes:
            print("error: no nodes left after applying --exclude/--nodelist", file=sys.stderr)
            return 2

        if excluded:
            print(f"[srun] excluding: {', '.join(sorted(excluded))}", file=sys.stderr)
        if included:
            print(f"[srun] nodelist: {', '.join(sorted(included))}", file=sys.stderr)

        return run_on_nodes(
            nodes, command, sshkey=system.get("sshkey"), all_nodes=all_nodes
        )


COMMANDS = SlurmRun
