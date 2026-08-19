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


class TestRayClusterPlan:
    def test_multi_node_sequence(self):
        pack = _stub_pack(num_machines=2)
        workload = PackCommand(pack, "main.py")
        plan = RayCluster(workload)
        steps = plan.executors

        # ListCommand(bringup, SequenceCommand(wait, workload, stop)): a
        # fork-join, not a flat sequence. Bring-up (head+workers as tasks of
        # the same srun step, role decided inside ray_node.sh) runs
        # concurrently with wait->workload->stop; `ray stop` (the sequence's
        # last step) kills the local raylet/GCS each node's blocking
        # `ray start --block` is waiting on, so bring-up returns on its own
        # once that runs (see module docstring) -- no separate teardown step
        # to list here.
        assert len(steps) == 1
        fork = steps[0]
        bringup, sequence = fork.executors
        wait, work, stop = sequence.executors

        bringup_argv = bringup.argv()
        assert bringup_argv[:3] == ["milabench", "slurm", "srun"]
        assert "-x" not in bringup_argv  # symmetric: no node excluded
        assert any(a.endswith("/bin/rayrun") for a in bringup_argv)
        assert "172.30.1.1" in bringup_argv  # head ip, for ray's own bind/connect
        assert "tri0001" in bringup_argv  # head hostname, for role detection

        assert bringup.pack.config["tag"] == _ray_step_tags(pack.config)

        wait_argv = wait.argv()
        assert wait_argv[0].endswith("/bin/raywait")
        assert "--address" in wait_argv
        assert "172.30.1.1:6379" in wait_argv
        assert "--expected" in wait_argv
        assert "2" in wait_argv

        assert work is workload

        stop_argv = stop.argv()
        assert stop_argv[:3] == ["milabench", "slurm", "srun"]
        assert "-x" not in stop_argv
        assert stop_argv[-2:] == ["/tmp/venv/bin/ray", "stop"]

    def test_njobs_does_not_duplicate_ray_tags(self):
        pack = _stub_pack(num_machines=2)
        workload = PackCommand(pack, "main.py")
        plan = NJobs(RayCluster(workload), 1)

        assert plan.pack.config["tag"] == [
            "vllm-moe-glm52-744b-bf16-nodes",
            "0",
        ]

        ray_plan = plan.executors[0]
        fork = ray_plan.executors[0]
        bringup = fork.executors[0]
        assert bringup.pack.config["tag"] == [
            "vllm-moe-glm52-744b-bf16-nodes",
            "0",
            "ray",
            "nolog",
        ]
        assert bringup.pack.tag == "vllm-moe-glm52-744b-bf16-nodes.0.ray.nolog"

    def test_single_node_uses_same_symmetric_bringup(self):
        # No more "skip srun for a single node" special case: `milabench
        # slurm srun` on a 1-node allocation just runs 1 task, so the same
        # symmetric bring-up (head decided by IP match inside ray_node.sh)
        # is used regardless of node count.
        pack = _stub_pack(num_machines=1, nodes=NODES[:1])
        workload = PackCommand(pack, "main.py")
        plan = RayCluster(workload)
        steps = plan.executors
        assert len(steps) == 1
        fork = steps[0]
        bringup, sequence = fork.executors
        assert bringup.argv()[:3] == ["milabench", "slurm", "srun"]
        _, work, _ = sequence.executors
        assert work is workload


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
