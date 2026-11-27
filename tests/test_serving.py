"""milabench.commands.Service, milabench.commands.serving and benchmarks/serving."""

from __future__ import annotations

import importlib.util
import os
import sys
import time
import types
from pathlib import Path

import pytest

from milabench.alt_async import proceed
from milabench.commands import CmdCommand, SequenceCommand, Service, clone_with
from milabench.commands.serving import (
    AtomServer,
    InferenceServer,
    RawServer,
    SGLangServer,
    TRTLLMServer,
    VLLMServer,
)

from test_executors import benchio
from test_vllm_variants import StubPack

ROOT = Path(__file__).resolve().parents[1]


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _stub(arch="rocm", **config):
    return StubPack({"name": "bench", "tag": ["bench"], "system": {"arch": arch}, **config})


# --- serving ---------------------------------------------------------------


def test_inference_server_selects_backend():
    pack = _stub()
    assert type(InferenceServer(pack, backend="vllm")) is VLLMServer
    assert type(InferenceServer(pack, backend="atom", entry="e.py")) is AtomServer
    assert type(InferenceServer(pack, backend="sglang")) is SGLangServer
    assert type(InferenceServer(pack, backend="trtllm")) is TRTLLMServer
    assert type(InferenceServer(pack, backend="atom", command=["my-server"])) is RawServer
    with pytest.raises(ValueError):
        InferenceServer(pack, backend="nope")


def test_server_argv():
    pack = _stub()
    assert VLLMServer(pack, "model", "--dtype", "bfloat16").argv() == [
        "/tmp/venv/bin/vllm", "serve", "model", "--dtype", "bfloat16"
    ]
    assert SGLangServer(pack, "--model-path", "m").argv() == [
        "/tmp/venv/bin/python", "-m", "sglang.launch_server", "--model-path", "m"
    ]
    assert RawServer(pack, ["srv", "--x"], "--y").argv() == ["srv", "--x", "--y"]
    assert AtomServer(pack, "--model", "m", entry="/code/entry.py").argv() == [
        "/tmp/venv/bin/python", "/code/entry.py", "--model", "m"
    ]


def test_atom_drops_vllm_only_flags():
    argv = ["--model", "m", "--dtype", "bfloat16", "--kv-cache-dtype=fp8", "--enable-auto-tool-choice",
            "--trust-remote-code", "--tensor-parallel-size", "8", "-tp", "8"]
    assert AtomServer(_stub(), *argv).argv()[3:] == ["--model", "m", "--trust-remote-code", "-tp", "8"]


def test_dp_flags_match_vllm_bench():
    sys.path.insert(0, str(ROOT / "benchmarks" / "vllm"))
    try:
        backends = _load(ROOT / "benchmarks" / "vllm" / "server_backends.py", "vllm_server_backends")
    finally:
        sys.path.pop(0)

    for cls in (VLLMServer, AtomServer):
        for nnodes, dp_local, index in ((2, 1, 0), (2, 1, 1), (4, 8, 3)):
            expected = backends.multinode_dp_flags(
                cls.name, nnodes=nnodes, dp_local=dp_local, node_index=index, master_ip="10.0.0.1"
            )
            got = cls(_stub()).dp_flags(nnodes, dp_local, "10.0.0.1", rank=index * dp_local)
            assert got == expected, (cls.name, nnodes, dp_local, index)

    with pytest.raises(NotImplementedError):
        SGLangServer(_stub()).dp_flags(2, 1, "10.0.0.1")


def test_server_env():
    pack = _stub(env={"ROCM_PATH": "/custom"})
    atom = AtomServer(pack)
    assert atom.options["setsid"] is True
    # the benchmark's env config wins over the backend's defaults
    assert atom.options["env"]["ROCM_PATH"] == "/custom"
    assert "ATOM_USE_CUSTOM_ALL_GATHER" not in atom.options["env"]

    multinode = atom.data_parallel(2, 1, "10.0.0.1")
    assert multinode.options["env"]["ATOM_USE_CUSTOM_ALL_GATHER"] == "0"
    assert "--data-parallel-master-ip" in multinode.argv()
    # the original is left untouched
    assert "--data-parallel-master-ip" not in atom.argv()

    vllm = VLLMServer(_stub(arch="cuda"), "m", "--load-format", "milabench_sharded")
    env = vllm.options["env"]
    assert env["ATOM_DISABLE_VLLM_PLUGIN"] == "1"
    assert env["VLLM_MILABENCH_SHARD_DIR"] == "/tmp/vllm-shards/bench"
    assert "FLASHINFER_CACHE_DIR" in env


# --- Service ---------------------------------------------------------------


def _pack(tag):
    pack = benchio()
    return pack.copy(clone_with(pack.config, {"tag": [*pack.config["tag"], tag]}))


def _run(plan):
    result = []

    async def main():
        result.append(await plan.execute(resource_cleaner=False))

    for _ in proceed(main()):
        pass
    return result[0]


async def _always():
    return True


async def _never():
    return False


def _alive(command):
    return [p for pack in command.packs() for p in pack.processes if p.poll() is None]


def test_service_stopped_after_workload():
    service = CmdCommand(_pack("server"), "sleep", "1000")
    plan = Service(service, CmdCommand(_pack("client"), "true"), ready=_always)

    start = time.time()
    assert _run(plan) == 0
    assert time.time() - start < 60
    assert service.pack.processes and not _alive(service)


def test_service_stopped_after_failed_workload():
    service = CmdCommand(_pack("server"), "sleep", "1000")
    workload = SequenceCommand(CmdCommand(_pack("eval"), "false"), CmdCommand(_pack("perf"), "true"))
    plan = Service(service, workload, ready=_always)

    assert _run(plan) >= 1
    assert not _alive(service)


def test_workload_stopped_when_service_exits(tmp_path):
    service = CmdCommand(_pack("server"), "sh", "-c", "sleep 2; exit 3")
    workload = CmdCommand(_pack("client"), "sleep", "1000")
    plan = Service(service, workload, ready=_always)

    start = time.time()
    assert _run(plan) >= 1
    assert time.time() - start < 60
    assert not _alive(workload)


def test_workload_stopped_when_one_node_server_exits(tmp_path):
    """multi-node: the coordinator stays up when a worker node's server dies, the clients must not wait for it"""
    from milabench.commands.serving import FirstExit

    coordinator = CmdCommand(_pack("server"), "sleep", "1000")
    worker = CmdCommand(_pack("dpworker"), "sh", "-c", "sleep 2; exit 3")
    workload = CmdCommand(_pack("client"), "sleep", "1000")
    plan = Service(FirstExit(coordinator, worker), workload, ready=_always)

    start = time.time()
    assert _run(plan) >= 1
    assert time.time() - start < 60
    assert not _alive(workload)
    assert not _alive(coordinator)


def test_service_never_ready(tmp_path):
    marker = tmp_path / "started"
    service = CmdCommand(_pack("server"), "sleep", "1000")
    workload = CmdCommand(_pack("client"), "touch", str(marker))
    plan = Service(service, workload, ready=_never, ready_timeout=1, poll_interval=0.1)

    assert _run(plan) == 1
    assert not marker.exists()
    assert not _alive(service)


def test_service_stop_command(tmp_path):
    marker = tmp_path / "stopped"
    service = CmdCommand(_pack("server"), "sleep", "1000")
    stop = CmdCommand(_pack("stop"), "touch", str(marker))
    plan = Service(service, CmdCommand(_pack("client"), "true"), ready=_always, stop=stop)

    assert _run(plan) == 0
    assert marker.exists()


def test_service_copy_moves_workload_only():
    service = CmdCommand(_pack("server"), "sleep", "1000")
    workload = SequenceCommand(CmdCommand(_pack("eval"), "true"), CmdCommand(_pack("perf"), "true"))
    job = _pack("job")

    copy = Service(service, workload, ready=_always).copy(job)
    assert copy.service.pack.config["tag"][-1] == "server"
    assert all(e.pack is job for e in copy.workload.executors)
    assert list(copy.executors) == [copy.service, copy.workload]


# --- benchmarks/serving -----------------------------------------------


def _serving_bench(config):
    mod = _load(ROOT / "benchmarks" / "serving" / "benchfile.py", "serving_benchfile")
    from test_vllm_variants import _make_vllm_pack
    from milabench.merge import merge

    class Bench(mod.Serving):
        # Package.copy builds a new core, which creates a virtualenv
        def copy(self, config):
            return type(self)(merge(self.config, config), core=self.core)

    class Mod:
        VLLM = Bench

    return mod, _make_vllm_pack(Mod, config)


BENCH = {
    "name": "serving-smoke",
    "tag": ["serving-smoke"],
    "definition": str(ROOT / "benchmarks" / "serving"),
    "dirs": {"venv": "/tmp/venv"},
    "num_machines": 1,
    "system": {"arch": "rocm", "nodes": [{"name": "n0", "ip": "10.0.0.1", "main": True}]},
    "server": {"backend": "atom", "argv": {"--model": "m", "--dtype": "bfloat16"}},
    "client": {"argv": {"--model": "m", "--port": 8123}},
}


def test_serving_plan_single_client():
    mod, pack = _serving_bench(BENCH)
    plan = pack.build_run_plan()

    assert isinstance(plan, mod.InferencePlan)
    assert plan.pack is pack
    assert isinstance(plan.service, AtomServer)
    assert plan.service.pack.config["tag"] == ["serving-smoke", "server", "nolog"]
    assert plan.service.ready_url() == "http://127.0.0.1:8123/v1/models"
    assert "--dtype" not in plan.service.argv()
    # no passes: one perf client on the benchmark's own pack
    assert plan.workload.pack is pack
    assert plan.workload.argv()[-6:-4] == ["--mode", "perf"]


def test_serving_plan_one_client_per_pass():
    passes = [{"name": "conc4", "argv": {}}, {"name": "ctx4k", "argv": {}}, {"name": "eval", "eval": {"tasks": ["t"]}}]
    mod, pack = _serving_bench({**BENCH, "tag": ["serving-smoke", "0"], "client": {**BENCH["client"], "passes": passes}})
    plan = pack.build_run_plan()

    # every pass is its own benchmark (own .data, own row in the report), in config order
    assert isinstance(plan.workload, mod.PassSequence)
    conc4, ctx4k, evaluation = plan.workload.executors
    assert conc4.pack.config["name"] == "serving-smoke.conc4"
    assert conc4.pack.config["tag"] == ["serving-smoke.conc4", "0"]
    assert ctx4k.pack.config["tag"] == ["serving-smoke.ctx4k", "0"]
    # lm-eval reports scores, not a rate
    assert evaluation.pack.config["tag"] == ["serving-smoke.eval", "0", "nolog"]
    assert conc4.argv()[-6:-4] == ["--pass", "conc4"]
    assert evaluation.argv()[-6:-4] == ["--pass", "eval"]


def test_pass_sequence_runs_every_pass(tmp_path):
    """a failed pass does not skip the following ones"""
    mod, _ = _serving_bench(BENCH)
    marker = tmp_path / "ran"
    sequence = mod.PassSequence(
        CmdCommand(_pack("a"), "false"),
        CmdCommand(_pack("b"), "touch", str(marker)),
    )
    assert _run(sequence) == 1
    assert marker.exists()


def test_serving_dp_workers():
    config = {
        **BENCH,
        "num_machines": 2,
        "system": {
            "arch": "rocm",
            "nodes": [{"name": "n0", "ip": "10.0.0.1", "main": True}, {"name": "n1", "ip": "10.0.0.2"}],
        },
        "server": {**BENCH["server"], "multinode": "dp", "dp_local": 1},
    }
    _, pack = _serving_bench(config)
    plan = pack.build_run_plan()

    coordinator, workers = plan.service.executors
    assert "--data-parallel-rank" in coordinator.argv()
    assert coordinator.argv()[coordinator.argv().index("--data-parallel-rank") + 1] == "0"
    worker_argv = workers.argv()
    assert "benchmate.server_worker" in worker_argv
    assert "@DP_RANK@" in worker_argv
    # the coordinator's env only spans nodes on the multi-node copy
    assert coordinator.options["env"]["ATOM_USE_CUSTOM_ALL_GATHER"] == "0"
    assert "--stop" in plan.stop.argv()
    assert plan.ready is not None


def test_serving_vllm_ray():
    from milabench.commands.ray import RayCluster

    config = {
        **BENCH,
        "num_machines": 2,
        "system": {
            "arch": "rocm",
            "nodes": [{"name": "n0", "ip": "10.0.0.1", "main": True}, {"name": "n1", "ip": "10.0.0.2"}],
        },
        "server": {"backend": "vllm", "argv": {"m": True}},
        "run_name": "test",
    }
    _, pack = _serving_bench(config)
    plan = pack.build_run_plan()

    assert isinstance(plan.service, RayCluster)
    server = plan.service.executor
    assert server.argv()[-2:] == ["--distributed-executor-backend", "ray"]
    env = server.options["env"]
    assert env["VLLM_HOST_IP"] == "10.0.0.1"
    assert "ATOM_DISABLE_VLLM_PLUGIN" in env["VLLM_RAY_EXTRA_ENV_VARS_TO_COPY"].split(",")
    assert "stop" in plan.stop.argv()


def test_single_node_deploy_is_the_server():
    server = VLLMServer(_stub(num_machines=1, system={"arch": "cuda", "nodes": [{"ip": "10.0.0.1"}]}), "m")
    assert server.deploy() == (server, None)
    with pytest.raises(NotImplementedError):
        two = {"arch": "cuda", "nodes": [{"ip": "10.0.0.1"}, {"ip": "10.0.0.2"}]}
        SGLangServer(_stub(num_machines=2, system=two)).deploy()


def test_server_worker_fills_rank(tmp_path, monkeypatch):
    # runs in the benchmark's venv; milabench's own env may have an older benchmate
    monkeypatch.setitem(sys.modules, "benchmate.ray_node", types.SimpleNamespace(ip_towards=None))
    worker = _load(ROOT / "benchmate" / "benchmate" / "server_worker.py", "server_worker")

    assert worker._marker(["/venv/bin/python", "/code/entry.py", "--model", "m"]) == "/code/entry.py"
    assert worker._marker(["/venv/bin/python", "-m", "sglang.launch_server"]) == "sglang.launch_server"
    assert worker._marker(["/venv/bin/vllm", "serve"]) == "/venv/bin/vllm"

    out = tmp_path / "argv"
    pidfile = tmp_path / "pid"
    monkeypatch.setattr(worker, "ip_towards", lambda _: "10.0.0.3")
    monkeypatch.setattr(sys, "argv", [
        "server_worker.py", "--nodes", "10.0.0.1,10.0.0.2,10.0.0.3", "--dp-local", "8", "--pidfile", str(pidfile),
        "--", "sh", "-c", f'echo "$@" > {out}', "sh", "--data-parallel-rank", "@DP_RANK@", "--node", "@NODE_INDEX@",
    ])
    assert worker.main() == 0
    assert out.read_text().split() == ["--data-parallel-rank", "16", "--node", "2"]
