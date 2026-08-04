"""Tests for ``milabench slurm srun`` (native srun vs SSH fallback)."""

from __future__ import annotations

import os
import stat
from types import SimpleNamespace

import pytest

from milabench.cli.slurm import srun as srun_mod
from milabench.cli.slurm.srun import (
    SlurmRun,
    _exclude_host_count,
    _exec_native_srun,
    _filter_nodes,
    _has_srun_nodes_override,
    _node_rank,
    _normalize_command,
    _parse_hostlist,
    _shrink_srun_allocation_for_exclude,
    _ssh_argv,
    run_on_nodes,
)


NODES = [
    {
        "name": "tri0001",
        "hostname": "tri0001",
        "ip": "172.30.1.1",
        "user": "tester",
        "main": True,
        "local": True,
        "sshport": 22,
    },
    {
        "name": "tri0002",
        "hostname": "tri0002",
        "ip": "172.30.1.2",
        "user": "tester",
        "main": False,
        "local": False,
        "sshport": 22,
    },
    {
        "name": "tri0003",
        "hostname": "tri0003",
        "ip": "172.30.1.3",
        "user": "tester",
        "main": False,
        "local": False,
        "sshport": 22,
    },
]


@pytest.fixture
def fake_srun(tmp_path, monkeypatch):
    """Install a dummy ``srun`` on PATH and capture ``os.execvp`` calls.

    ``_exec_native_srun`` replaces the process via ``execvp``, so we intercept
    that call instead of actually replacing pytest.
    """
    bindir = tmp_path / "bin"
    bindir.mkdir()
    srun_path = bindir / "srun"
    srun_path.write_text("#!/bin/sh\nexit 0\n")
    srun_path.chmod(srun_path.stat().st_mode | stat.S_IEXEC)

    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}")

    captured: dict = {}

    def fake_execvp(file, args):
        captured["file"] = file
        captured["args"] = list(args)
        raise SystemExit(0)

    monkeypatch.setattr(srun_mod.os, "execvp", fake_execvp)
    return srun_path, captured


@pytest.fixture
def no_srun(monkeypatch):
    """Force the SSH fallback by hiding any real ``srun``."""
    monkeypatch.setattr(srun_mod.shutil, "which", lambda name: None)


class TestShrinkAllocation:
    def test_no_exclude_unchanged(self, monkeypatch):
        monkeypatch.setenv("SLURM_JOB_NUM_NODES", "3")
        args = ["--", "echo", "hi"]
        assert _shrink_srun_allocation_for_exclude(args) == args

    def test_exclude_shrinks_to_remaining(self, monkeypatch):
        monkeypatch.setenv("SLURM_JOB_NUM_NODES", "3")
        out = _shrink_srun_allocation_for_exclude(["-x", "tri0001", "--", "cmd"])
        assert out[:3] == ["--nodes=2", "--ntasks=2", "--ntasks-per-node=1"]
        assert out[3:] == ["-x", "tri0001", "--", "cmd"]

    def test_exclude_hostlist_count(self, monkeypatch):
        monkeypatch.setenv("SLURM_NNODES", "4")
        out = _shrink_srun_allocation_for_exclude(
            ["-x", "tri[0001-0002]", "--", "cmd"]
        )
        assert out[0] == "--nodes=2"
        assert out[1] == "--ntasks=2"

    def test_respects_explicit_nodes_override(self, monkeypatch):
        monkeypatch.setenv("SLURM_JOB_NUM_NODES", "3")
        args = ["--nodes=1", "-x", "tri0001", "--", "cmd"]
        assert _shrink_srun_allocation_for_exclude(args) == args
        assert _has_srun_nodes_override(args)

    def test_no_job_size_leaves_args(self, monkeypatch):
        monkeypatch.delenv("SLURM_JOB_NUM_NODES", raising=False)
        monkeypatch.delenv("SLURM_NNODES", raising=False)
        args = ["-x", "tri0001", "--", "cmd"]
        assert _shrink_srun_allocation_for_exclude(args) == args

    def test_exclude_host_count(self):
        assert _exclude_host_count(["-x", "a,b"]) == 2
        assert _exclude_host_count(["--exclude", "tri0001"]) == 1
        assert _exclude_host_count(["--", "cmd"]) == 0


class TestFilterNodes:
    def test_exclude_main(self):
        excluded = _parse_hostlist("tri0001")
        out = _filter_nodes(NODES, excluded=excluded)
        assert [n["name"] for n in out] == ["tri0002", "tri0003"]

    def test_nodelist(self):
        included = _parse_hostlist("tri0002,tri0003")
        out = _filter_nodes(NODES, nodelist=included)
        assert [n["name"] for n in out] == ["tri0002", "tri0003"]

    def test_node_rank_in_full_list(self):
        assert _node_rank(NODES, NODES[0]) == 0
        assert _node_rank(NODES, NODES[2]) == 2


class TestNormalizeAndSsh:
    def test_normalize_strips_double_dash(self):
        assert _normalize_command(["--", "echo", "hi"]) == ["echo", "hi"]
        assert _normalize_command(["echo", "hi"]) == ["echo", "hi"]

    def test_ssh_argv(self):
        argv = _ssh_argv(NODES[1], ["env", "SLURM_NODEID=1", "cmd"])
        assert argv[0] == "ssh"
        assert "tester@172.30.1.2" in argv
        assert argv[-4:] == ["--", "env", "SLURM_NODEID=1", "cmd"]


class TestNativeSrun:
    def test_execs_dummy_srun_with_shrunk_allocation(self, fake_srun, monkeypatch):
        srun_path, captured = fake_srun
        monkeypatch.setenv("SLURM_JOB_NUM_NODES", "3")
        monkeypatch.setattr(
            srun_mod.sys,
            "argv",
            [
                "milabench",
                "slurm",
                "srun",
                "-x",
                "tri0001",
                "--",
                "benchrun",
                "--nnodes=3",
            ],
        )

        with pytest.raises(SystemExit) as exc:
            _exec_native_srun()

        assert exc.value.code == 0
        assert captured["file"] == str(srun_path)
        assert captured["args"][0] == str(srun_path)
        assert captured["args"][1:4] == [
            "--nodes=2",
            "--ntasks=2",
            "--ntasks-per-node=1",
        ]
        assert captured["args"][4:] == [
            "-x",
            "tri0001",
            "--",
            "benchrun",
            "--nnodes=3",
        ]

    def test_execute_prefers_native_srun(self, fake_srun, monkeypatch):
        _, captured = fake_srun
        monkeypatch.setenv("SLURM_JOB_NUM_NODES", "3")
        monkeypatch.setattr(
            srun_mod.sys,
            "argv",
            ["milabench", "slurm", "srun", "-x", "tri0001", "--", "true"],
        )

        args = SimpleNamespace(command=["--", "true"], exclude="tri0001", nodelist=None)
        with pytest.raises(SystemExit):
            SlurmRun.execute(args)

        assert captured["args"][1:4] == [
            "--nodes=2",
            "--ntasks=2",
            "--ntasks-per-node=1",
        ]

    def test_no_srun_on_path_does_not_exec(self, no_srun, monkeypatch):
        called = []

        def boom(*a, **k):
            called.append(True)
            raise AssertionError("execvp should not be called")

        monkeypatch.setattr(srun_mod.os, "execvp", boom)
        monkeypatch.setattr(srun_mod.sys, "argv", ["milabench", "slurm", "srun", "--", "true"])
        _exec_native_srun()
        assert called == []


class TestSshFallback:
    def test_execute_dispatches_filtered_nodes(self, no_srun, monkeypatch):
        monkeypatch.setattr(
            srun_mod,
            "_load_system",
            lambda: {"nodes": NODES, "sshkey": None},
        )

        seen = {}

        def fake_run_on_nodes(nodes, command, sshkey=None, *, all_nodes=None):
            seen["nodes"] = [n["name"] for n in nodes]
            seen["command"] = command
            seen["all_nodes"] = [n["name"] for n in all_nodes]
            return 0

        monkeypatch.setattr(srun_mod, "run_on_nodes", fake_run_on_nodes)

        args = SimpleNamespace(
            command=["--", "benchrun", "--nnodes=3"],
            exclude="tri0001",
            nodelist=None,
        )
        assert SlurmRun.execute(args) == 0
        assert seen["nodes"] == ["tri0002", "tri0003"]
        assert seen["command"] == ["benchrun", "--nnodes=3"]
        assert seen["all_nodes"] == ["tri0001", "tri0002", "tri0003"]

    def test_execute_missing_command(self, no_srun, monkeypatch):
        monkeypatch.setattr(
            srun_mod,
            "_load_system",
            lambda: {"nodes": NODES},
        )
        args = SimpleNamespace(command=["--"], exclude=None, nodelist=None)
        assert SlurmRun.execute(args) == 2

    def test_execute_exclude_all_nodes(self, no_srun, monkeypatch):
        monkeypatch.setattr(
            srun_mod,
            "_load_system",
            lambda: {"nodes": NODES},
        )
        args = SimpleNamespace(
            command=["--", "true"],
            exclude="tri0001,tri0002,tri0003",
            nodelist=None,
        )
        assert SlurmRun.execute(args) == 2

    def test_run_on_nodes_injects_slurm_env(self, monkeypatch):
        started = []

        class FakeMP:
            def __init__(self, timeout=None, constructor=None):
                self._started = started

            def start(self, argv, info=None):
                self._started.append((list(argv), info))

            def __iter__(self):
                for _, info in self._started:
                    yield SimpleNamespace(
                        event="end",
                        node=info["node"],
                        data={"return_code": 0},
                        pipe="stdout",
                    )

        monkeypatch.setattr(srun_mod, "Multiplexer", FakeMP)

        workers = NODES[1:]
        rc = run_on_nodes(workers, ["benchrun", "main.py"], all_nodes=NODES)
        assert rc == 0
        assert len(started) == 2

        argv0, info0 = started[0]
        assert info0["node"] == "tri0002"
        # Step-local ids among launched workers (same as native ``srun -x main``).
        assert "SLURM_NODEID=0" in argv0
        assert "SLURM_PROCID=0" in argv0
        assert "SLURM_NNODES=3" in argv0
        assert argv0[-2:] == ["benchrun", "main.py"]

        argv1, info1 = started[1]
        assert info1["node"] == "tri0003"
        assert "SLURM_NODEID=1" in argv1


def _parse_env_assignments(argv: list[str]) -> dict[str, str]:
    """Extract ``KEY=VAL`` tokens from an ``env ...`` ssh remote command."""
    env = {}
    for tok in argv:
        if "=" in tok and not tok.startswith("-"):
            key, _, val = tok.partition("=")
            if key.isidentifier() or key.startswith("SLURM_"):
                env[key] = val
    return env


def _benchrun_argv_from_remote(argv: list[str]) -> list[str]:
    """Return argv after the ``benchrun`` executable in a remote command."""
    for i, tok in enumerate(argv):
        if tok.endswith("benchrun") or tok == "benchrun":
            return list(argv[i + 1 :])
    raise AssertionError(f"benchrun not found in {argv}")


def _torch_node_rank(argv: list[str]) -> int:
    """Parse ``--node-rank`` from torchrun options (before ``--`` only)."""
    sep = argv.index("--") if "--" in argv else len(argv)
    torch_args = argv[:sep]
    for i, arg in enumerate(torch_args):
        if arg.startswith("--node-rank="):
            return int(arg.split("=", 1)[1])
        if arg == "--node-rank" and i + 1 < len(torch_args):
            return int(torch_args[i + 1])
    raise AssertionError(f"no --node-rank before -- in {argv}")


def _with_main_node_rank(argv: list[str], rank: int = 0) -> list[str]:
    """Plan shape for the main node: explicit ``--node-rank`` before ``--``."""
    argv = list(argv)
    sep = argv.index("--")
    return argv[:sep] + [f"--node-rank={rank}"] + argv[sep:]


BENCHRUN_STATIC = [
    "--nnodes=3",
    "--rdzv-backend=static",
    "--rdzv-endpoint=172.30.1.1:29400",
    "--master-addr=172.30.1.1",
    "--master-port=29400",
    "--nproc-per-node=1",
    "--",
    "main.py",
    "--repeats",
    "30",
]


class TestBenchrunThroughSrun:
    """End-to-end: srun worker env → benchrun ``--node-rank`` injection."""

    @pytest.fixture(autouse=True)
    def _import_benchrun(self):
        import sys
        from unittest.mock import MagicMock

        for name in (
            "torch",
            "torch.distributed",
            "torch.distributed.run",
            "torch.distributed.elastic",
            "torch.distributed.elastic.multiprocessing",
            "torch.distributed.elastic.multiprocessing.api",
            "torch.distributed.elastic.multiprocessing.subprocess_handler",
        ):
            sys.modules.setdefault(name, MagicMock())

        # Load source benchmate, not an older installed wheel.
        root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "benchmate")
        )
        sys.path.insert(0, root)
        for mod in list(sys.modules):
            if mod == "benchmate" or mod.startswith("benchmate."):
                del sys.modules[mod]

        from benchmate.benchrun import maybe_inject_node_rank

        self.maybe_inject_node_rank = maybe_inject_node_rank

    def test_native_step_ids_map_to_global_ranks(self, monkeypatch):
        """Native ``srun -x main`` uses step-local NODEID 0..N-2.

        Combined with main's plan ``--node-rank=0``, the job covers 0..N-1.
        """
        # Main runs under the outer allocation (NODEID=0) with an explicit rank.
        monkeypatch.setenv("SLURM_NODEID", "0")
        main = self.maybe_inject_node_rank(_with_main_node_rank(BENCHRUN_STATIC, 0))
        assert _torch_node_rank(main) == 0
        assert main.index("--node-rank=0") < main.index("--")
        assert main.count("--node-rank=0") == 1

        ranks = [_torch_node_rank(main)]
        for step_id in (0, 1):
            monkeypatch.setenv("SLURM_NODEID", str(step_id))
            out = self.maybe_inject_node_rank(BENCHRUN_STATIC)
            assert out.index(f"--node-rank={step_id + 1}") < out.index("--")
            ranks.append(_torch_node_rank(out))
        assert ranks == [0, 1, 2]

    def test_ssh_fallback_exclude_main_feeds_benchrun(self, no_srun, monkeypatch):
        """SSH path injects step-local NODEIDs; benchrun turns them into 1..N-1."""
        monkeypatch.setattr(
            srun_mod,
            "_load_system",
            lambda: {"nodes": NODES, "sshkey": None},
        )

        started = []

        class FakeMP:
            def __init__(self, timeout=None, constructor=None):
                pass

            def start(self, argv, info=None):
                started.append((list(argv), info))

            def __iter__(self):
                for _, info in started:
                    yield SimpleNamespace(
                        event="end",
                        node=info["node"],
                        data={"return_code": 0},
                        pipe="stdout",
                    )

        monkeypatch.setattr(srun_mod, "Multiplexer", FakeMP)

        args = SimpleNamespace(
            command=["--", "benchrun", *BENCHRUN_STATIC],
            exclude="tri0001",
            nodelist=None,
        )
        assert SlurmRun.execute(args) == 0
        assert len(started) == 2

        # Main keeps rank 0 under outer SLURM_NODEID=0 via plan argv.
        monkeypatch.setenv("SLURM_NODEID", "0")
        main_rank = _torch_node_rank(
            self.maybe_inject_node_rank(_with_main_node_rank(BENCHRUN_STATIC, 0))
        )

        worker_ranks = []
        for argv, info in started:
            env = _parse_env_assignments(argv)
            assert "SLURM_NODEID" in env
            bench_argv = _benchrun_argv_from_remote(argv)
            assert "--node-rank" not in " ".join(
                bench_argv[: bench_argv.index("--")]
            )

            monkeypatch.setenv("SLURM_NODEID", env["SLURM_NODEID"])
            injected = self.maybe_inject_node_rank(bench_argv)
            assert injected.index(f"--node-rank={_torch_node_rank(injected)}") < injected.index(
                "--"
            )
            worker_ranks.append((info["node"], _torch_node_rank(injected)))

        assert main_rank == 0
        assert worker_ranks == [("tri0002", 1), ("tri0003", 2)]
        assert sorted([main_rank, *(r for _, r in worker_ranks)]) == [0, 1, 2]

    def test_dummy_native_srun_then_benchrun(self, fake_srun, monkeypatch):
        """Captured native srun argv + step ids → same ranks as production."""
        srun_path, captured = fake_srun
        monkeypatch.setenv("SLURM_JOB_NUM_NODES", "3")
        monkeypatch.setattr(
            srun_mod.sys,
            "argv",
            [
                "milabench",
                "slurm",
                "srun",
                "-x",
                "tri0001",
                "--",
                "benchrun",
                *BENCHRUN_STATIC,
            ],
        )

        args = SimpleNamespace(
            command=["--", "benchrun", *BENCHRUN_STATIC],
            exclude="tri0001",
            nodelist=None,
        )
        with pytest.raises(SystemExit):
            SlurmRun.execute(args)

        assert captured["file"] == str(srun_path)
        srun_args = captured["args"][1:]
        assert srun_args[:3] == ["--nodes=2", "--ntasks=2", "--ntasks-per-node=1"]
        assert "-x" in srun_args and "tri0001" in srun_args

        monkeypatch.setenv("SLURM_NODEID", "0")
        ranks = [
            _torch_node_rank(
                self.maybe_inject_node_rank(_with_main_node_rank(BENCHRUN_STATIC, 0))
            )
        ]
        for step_id in range(2):
            monkeypatch.setenv("SLURM_NODEID", str(step_id))
            out = self.maybe_inject_node_rank(BENCHRUN_STATIC)
            assert out.index(f"--node-rank={step_id + 1}") < out.index("--")
            ranks.append(_torch_node_rank(out))
        assert ranks == [0, 1, 2]


class TestTorchrunSrunMainRank:
    """Plan-level: main gets ``--node-rank=0``; workers do not (benchrun fills)."""

    def _stub_pack(self, devices=(0, 1)):
        from pathlib import Path

        from milabench.merge import merge
        from milabench.pack import BasePackage

        class StubPack(BasePackage):
            def __init__(self, config):
                self.config = config
                self.core = SimpleNamespace()
                self.dirs = SimpleNamespace(
                    cache=Path("/tmp/c"),
                    venv=Path("/tmp/venv"),
                    code=Path("/tmp"),
                    data=Path("/tmp"),
                    runs=Path("/tmp"),
                    extra=Path("/tmp"),
                )
                self.working_directory = Path("/tmp")
                self.phase = None
                self.processes = []
                self.main_script = "main.py"

            def copy(self, config):
                return StubPack(merge(self.config, config))

        return StubPack(
            {
                "name": "torchsrun",
                "tag": [],
                "dirs": {"venv": "/tmp/venv"},
                "plan": {"method": "njobs", "n": 1},
                "devices": list(devices),
                "num_machines": 3,
                "argv": ["--repeats", "30"],
                "system": {"nodes": NODES[:3]},
            }
        )

    def test_torchrun_srun_main_pins_rank_zero(self):
        from milabench.commands import PackCommand
        from milabench.commands.srun import TorchrunSrun

        pack = self._stub_pack(devices=(0, 1))
        pack.main_script = "main.py"
        plan = TorchrunSrun(PackCommand(pack, "main.py", "--repeats", "30"))
        main = plan.main_executor()
        worker = plan.worker_executor()
        assert "--node-rank=0" in main.wrapper_argv
        assert "--node-rank=0" not in worker.wrapper_argv
        main_argv = main.argv()
        assert main_argv.index("--node-rank=0") < main_argv.index("--")

    def test_torchsrun_always_main_pins_rank_zero(self):
        import importlib.util
        from pathlib import Path

        from milabench.commands import PackCommand

        path = (
            Path(__file__).resolve().parents[1]
            / "benchmarks"
            / "torchsrun"
            / "benchfile.py"
        )
        spec = importlib.util.spec_from_file_location("torchsrun_benchfile", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        pack = self._stub_pack(devices=(0,))
        pack.main_script = "main.py"
        # torchsrun uses a single proc per node; TorchrunAlways still wraps.
        plan = mod.TorchrunSrunAlways(PackCommand(pack, "main.py", "--repeats", "30"))
        main = plan.main_executor()
        worker = plan.worker_executor()
        assert "--node-rank=0" in main.wrapper_argv
        assert "--node-rank=0" not in worker.wrapper_argv
        main_argv = main.argv()
        assert main_argv.index("--node-rank=0") < main_argv.index("--")
        # Worker plan argv has no node-rank; benchrun injects from SLURM_NODEID.
        worker_argv = worker.argv()
        assert "--" in worker_argv
        assert not any(
            a == "--node-rank" or a.startswith("--node-rank=")
            for a in worker_argv[: worker_argv.index("--")]
        )
