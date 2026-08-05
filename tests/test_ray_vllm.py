"""Tests for RayCluster and multi-node vLLM run plans."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from milabench.commands import PackCommand, VoirCommand
from milabench.commands.ray import RayCluster
from milabench.merge import merge
from milabench.pack import BasePackage


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
]


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


def _stub_pack(*, num_machines=2, devices=(0, 1), nodes=None):
    return StubPack(
        {
            "name": "vllm-moe-glm52-744b-bf16-nodes",
            "definition": str(
                Path(__file__).resolve().parents[1] / "benchmarks" / "vllm"
            ),
            "tag": [],
            "dirs": {
                "venv": "/tmp/venv",
                "code": "/tmp",
                "cache": "/tmp/c",
                "data": "/tmp",
                "runs": "/tmp",
                "extra": "/tmp",
                "base": "/tmp",
            },
            "plan": {"method": "njobs", "n": 1},
            "devices": list(devices),
            "num_machines": num_machines,
            "argv": [],
            "system": {"nodes": nodes if nodes is not None else NODES},
            "server": {
                "argv": {
                    "zai-org/GLM-5.2": True,
                    "--tensor-parallel-size": "{total_gpu_count}",
                    "--distributed-executor-backend": "ray",
                }
            },
            "client": {
                "argv": {
                    "--model": "zai-org/GLM-5.2",
                    "--num-prompts": 8,
                }
            },
        }
    )


def _flat_argv(plan):
    """Flatten argv lists from ``plan.commands()`` for assertions."""
    return [list(argv) for _pack, argv, _kwargs in plan.commands()]


class TestRayClusterPlan:
    def test_multi_node_sequence(self):
        pack = _stub_pack(num_machines=2)
        workload = PackCommand(pack, "main.py")
        plan = RayCluster(workload)
        steps = plan.executors

        # head start, worker start (srun), wait, workload, worker stop (srun), head stop
        assert len(steps) == 6
        head_start, worker_start, wait, work, worker_stop, head_stop = steps

        assert "start" in head_start.argv()
        assert "--head" in head_start.argv()
        assert "172.30.1.1" in " ".join(head_start.argv())

        worker_argv = worker_start.argv()
        assert worker_argv[:4] == ["milabench", "slurm", "srun", "-x"]
        assert "tri0001" in worker_argv
        assert any("172.30.1.1:6379" in a for a in worker_argv)

        wait_argv = wait.argv()
        assert wait_argv[0].endswith("/bin/python")
        assert "-c" in wait_argv
        assert "Ray cluster ready" in wait_argv[wait_argv.index("-c") + 1]

        assert work is workload

        worker_stop_argv = worker_stop.argv()
        assert worker_stop_argv[:4] == ["milabench", "slurm", "srun", "-x"]
        assert worker_stop_argv[-2:] == ["/tmp/venv/bin/ray", "stop"]

        assert head_stop.argv() == ["/tmp/venv/bin/ray", "stop"]

    def test_single_node_skips_workers(self):
        pack = _stub_pack(num_machines=1, nodes=NODES[:1])
        workload = PackCommand(pack, "main.py")
        plan = RayCluster(workload)
        steps = plan.executors
        # head start, wait, workload, head stop — no srun worker steps
        assert len(steps) == 4
        argv_lists = _flat_argv(plan)
        assert not any(a[:3] == ["milabench", "slurm", "srun"] for a in argv_lists)


class TestVLLMRunPlan:
    def _load_vllm(self):
        import importlib.util

        path = (
            Path(__file__).resolve().parents[1]
            / "benchmarks"
            / "vllm"
            / "benchfile.py"
        )
        spec = importlib.util.spec_from_file_location("vllm_benchfile", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def _make_vllm_pack(self, mod, *, num_machines=2, nodes=None, server=None):
        stub = _stub_pack(num_machines=num_machines, nodes=nodes)
        cfg = dict(stub.config)
        if server is not None:
            cfg["server"] = server

        class _Core:
            pass

        core = _Core()
        core.pack_path = Path(cfg["definition"])
        core.dirs = stub.dirs
        core.constraints = None
        core._nox_runner = None
        core._nox_session = None
        core.install_mark_file = Path("/tmp/mark")

        return mod.VLLM(cfg, core=core)

    def test_multi_node_wraps_ray_cluster(self):
        mod = self._load_vllm()
        pack = self._make_vllm_pack(mod, num_machines=2)
        plan = pack.build_run_plan()
        assert isinstance(plan, RayCluster)
        assert plan.executor is not None
        assert isinstance(plan.executor, VoirCommand)
        assert "--distributed-executor-backend" in pack.server_argv()
        assert "ray" in pack.server_argv()

    def test_single_node_no_ray(self):
        mod = self._load_vllm()
        pack = self._make_vllm_pack(
            mod,
            num_machines=1,
            nodes=NODES[:1],
            server={
                "argv": {
                    "mistralai/Mistral-Small-3.1-24B-Instruct-2503": True,
                    "--tensor-parallel-size": "1",
                }
            },
        )
        plan = pack.build_run_plan()
        assert not isinstance(plan, RayCluster)
        assert isinstance(plan, VoirCommand)
        assert "--distributed-executor-backend" not in pack.server_argv()
