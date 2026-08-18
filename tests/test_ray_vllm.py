"""Tests for RayCluster and multi-node vLLM run plans."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from milabench.commands import NJobs, PackCommand, VoirCommand
from milabench.commands.ray import RayCluster, _ray_step_tags
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

        head_argv = head_start.argv()
        assert head_argv[0] == "/bin/bash"
        assert any(a.endswith("ray_start_head.sh") for a in head_argv)
        assert "--head" in head_argv
        assert "172.30.1.1" in " ".join(head_argv)
        assert head_start.options.get("env", {}).get("MILABENCH_RAY_HEAD_PIDFILE")

        assert head_start.pack.config["tag"] == _ray_step_tags(pack.config)

        worker_argv = worker_start.argv()
        assert worker_argv[:4] == ["milabench", "slurm", "srun", "-x"]
        assert "tri0001" in worker_argv
        assert any("172.30.1.1:6379" in a for a in worker_argv)

        wait_argv = wait.argv()
        assert wait_argv[0].endswith("/bin/python")
        assert "ray_wait.py" in wait_argv[-1] or any(
            a.endswith("ray_wait.py") for a in wait_argv
        )
        assert "--address" in wait_argv
        assert "172.30.1.1:6379" in wait_argv
        assert "--expected" in wait_argv
        assert "2" in wait_argv

        assert work is workload

        worker_stop_argv = worker_stop.argv()
        assert worker_stop_argv[:4] == ["milabench", "slurm", "srun", "-x"]
        assert worker_stop_argv[-2:] == ["/tmp/venv/bin/ray", "stop"]

        assert head_stop.argv()[0] == "/bin/bash"
        assert "ray stop" in " ".join(head_stop.argv())
        assert "ray-head" in " ".join(head_stop.argv())  # pidfile name

    def test_njobs_does_not_duplicate_ray_tags(self):
        pack = _stub_pack(num_machines=2)
        workload = PackCommand(pack, "main.py")
        plan = NJobs(RayCluster(workload), 1)

        assert plan.pack.config["tag"] == [
            "vllm-moe-glm52-744b-bf16-nodes",
            "0",
        ]

        ray_plan = plan.executors[0]
        head_start = ray_plan.executors[0]
        assert head_start.pack.config["tag"] == [
            "vllm-moe-glm52-744b-bf16-nodes",
            "0",
            "ray",
            "nolog",
        ]
        assert head_start.pack.tag == "vllm-moe-glm52-744b-bf16-nodes.0.ray.nolog"

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
