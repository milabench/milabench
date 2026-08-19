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
        # fork-join, not a flat sequence -- this ``.executors`` property is
        # purely descriptive/introspective (argv shape, tags). The actual
        # concurrency + teardown at run time goes through
        # RayCluster.execute()'s own custom override, not this ListCommand:
        # `ray stop` does not make bringup's `ray start --block` return on
        # its own (see module docstring), so execute() explicitly stops it
        # once wait->workload->stop is done.
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

    def test_ray_config_section_overrides_and_extra_args(self):
        """A benchmark's own `ray:` config section (port/init_timeout/
        start_args/stop_args/env) lets different benchmarks tune their Ray
        cluster independently, instead of one process-wide env var applying
        to every benchmark in the run."""
        pack = _stub_pack(num_machines=2)
        pack.config["ray"] = {
            "port": 7000,
            "init_timeout": 42,
            "start_args": {"--num-cpus": 64},
            "stop_args": {"--grace-period": 30},
            "env": {"RAY_health_check_period_ms": 5000},
        }
        workload = PackCommand(pack, "main.py")
        plan = RayCluster(workload)
        bringup, sequence = plan.executors[0].executors
        _, _, stop = sequence.executors

        assert plan.resolve_port() == 7000
        assert plan.resolve_init_timeout() == 42

        bringup_argv = bringup.argv()
        assert "7000" in bringup_argv
        assert "42" in bringup_argv
        # rayrun's own extra_ray_args positional (argparse.REMAINDER) needs a
        # literal `--` first -- see benchmate/ray_node.py.
        assert bringup_argv[-3:] == ["--", "--num-cpus", "64"]
        assert bringup.options.get("env") == {"RAY_health_check_period_ms": "5000"}

        stop_argv = stop.argv()
        assert stop_argv[-2:] == ["--grace-period", "30"]
        assert stop.options.get("env") == {"RAY_health_check_period_ms": "5000"}

    def test_ray_config_section_absent_keeps_defaults(self):
        """No `ray:` section -> unchanged argv (no stray trailing `--`
        breaking rayrun for benchmarks that don't need this feature)."""
        pack = _stub_pack(num_machines=2)
        workload = PackCommand(pack, "main.py")
        plan = RayCluster(workload)
        bringup, sequence = plan.executors[0].executors
        _, _, stop = sequence.executors

        assert bringup.argv()[-1] == "--"
        assert stop.argv()[-2:] == ["/tmp/venv/bin/ray", "stop"]

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
