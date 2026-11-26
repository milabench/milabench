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
    OmniServer,
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
    assert type(InferenceServer(pack, backend="vllm-omni")) is OmniServer
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


def test_omni_server_argv():
    pack = _stub(arch="cuda")
    assert OmniServer(pack, "black-forest-labs/FLUX.1-dev", "--tensor-parallel-size", "2").argv() == [
        "/tmp/venv/bin/vllm", "serve", "black-forest-labs/FLUX.1-dev", "--tensor-parallel-size", "2", "--omni"
    ]
    # --omni given in the config is not repeated
    assert OmniServer(pack, "m", "--omni").argv().count("--omni") == 1


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


def test_omni_plan():
    mod = _load(ROOT / "benchmarks" / "omni" / "benchfile.py", "omni_benchfile")
    from test_vllm_variants import _make_vllm_pack
    from milabench.merge import merge

    class Bench(mod.Omni):
        def copy(self, config):
            return type(self)(merge(self.config, config), core=self.core)

    class Mod:
        VLLM = Bench

    config = {
        **BENCH, "name": "omni-flux", "tag": ["omni-flux"], "definition": str(ROOT / "benchmarks" / "omni"),
        "system": {**BENCH["system"], "arch": "cuda"},
        "server": {"backend": "vllm-omni", "argv": {"black-forest-labs/FLUX.1-dev": True, "--tensor-parallel-size": 2}},
        "client": {"argv": {"--task": "t2i", "--steps": 20, "--port": 8124}},
    }
    pack = _make_vllm_pack(Mod, config)
    plan = pack.build_run_plan()

    assert isinstance(plan, mod.OmniPlan) and plan.pack is pack
    assert isinstance(plan.service, OmniServer)
    assert plan.service.pack.config["tag"] == ["omni-flux", "server", "nolog"]
    assert plan.service.argv()[-4:] == ["black-forest-labs/FLUX.1-dev", "--tensor-parallel-size", "2", "--omni"]
    assert plan.service.ready_url() == "http://127.0.0.1:8124/v1/models"
    assert plan.workload.pack is pack
    assert "--task" in plan.workload.argv() and "--omni" not in plan.workload.argv()


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


def test_memory_fraction_cap(monkeypatch):
    from milabench.commands.serving import cap_memory_fraction

    argv = ["m", "--gpu-memory-utilization", "0.9", "--mem-fraction-static=0.85", "--max-model-len", "8192"]
    assert cap_memory_fraction(argv) == argv
    monkeypatch.setenv("MILABENCH_GPU_MEMORY_FRACTION_MAX", "0.75")
    assert cap_memory_fraction(argv) == ["m", "--gpu-memory-utilization", "0.75", "--mem-fraction-static=0.75",
                                         "--max-model-len", "8192"]
    # a lower value in the config stays
    assert cap_memory_fraction(["--gpu-memory-utilization", "0.5"]) == ["--gpu-memory-utilization", "0.5"]
    assert VLLMServer(_stub(arch="cuda"), "m", "--gpu-memory-utilization", "0.95").argv()[-1] == "0.75"


def test_kv_cache_sized_for_the_passes(monkeypatch, tmp_path):
    import json as _json
    from milabench.commands import serving

    # Qwen3-8B: 36 layers, 8 KV heads of 128: 72 KiB per token with an fp8 KV cache
    snap = tmp_path / "hub" / "models--Qwen--Qwen3-8B" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text(_json.dumps({"num_hidden_layers": 36, "num_attention_heads": 32,
                                                   "num_key_value_heads": 8, "head_dim": 128}))
    (snap / "model.safetensors").write_bytes(b"0" * 1000)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setenv("MILABENCH_GPU_MEMORY_FRACTION_MAX", "0.65")
    monkeypatch.setattr(serving, "memory_total", lambda: 120 * 2**30)
    assert serving.kv_bytes_per_token(_json.loads((snap / "config.json").read_text()), "fp8") == 72 * 1024

    config = {"name": "b", "tag": ["b"], "system": {"arch": "cuda"},
              "client": {"argv": {"--random-range-ratio": 0.0},
                         "passes": [{"name": "c32", "argv": {"--max-concurrency": 32, "--random-input-len": 1024,
                                                             "--random-output-len": 1024}},
                                    {"name": "quality", "quality": {"thinking": ["off"]}}]}}
    argv = ["Qwen/Qwen3-8B", "--max-model-len", "8192", "--max-num-seqs", "32", "--kv-cache-dtype", "fp8"]
    # 32 x 2048 tokens (perf) vs 16 x 6144 (QualityBench, thinking off): 98304 tokens, x1.5
    assert serving.tokens_in_flight(config, 8192, 32) == 98304
    assert serving.kv_cache_budget(config, argv) == int(98304 * 72 * 1024 * 1.5)

    # never more than the memory cap leaves next to the weights (but always one full sequence)
    long = dict(config, client={"passes": [{"name": "s", "swebench": {"workers": 32}}]})
    budget = serving.kv_cache_budget(long, ["Qwen/Qwen3-8B", "--max-model-len", "65536", "--max-num-seqs", "32",
                                            "--kv-cache-dtype", "fp8"])
    assert budget == int(0.65 * 120 * 2**30 - 1000 - serving.runtime_overhead(1000))

    # only with MILABENCH_KV_CACHE_FIT, only when the config does not size it itself
    pack = _stub(arch="cuda", **config)
    assert "--kv-cache-memory-bytes" not in VLLMServer(pack, *argv).argv()
    monkeypatch.setenv("MILABENCH_KV_CACHE_FIT", "1")
    assert VLLMServer(pack, *argv).argv()[-2:] == ["--kv-cache-memory-bytes", str(int(98304 * 72 * 1024 * 1.5))]
    assert VLLMServer(pack, *argv, "--kv-cache-memory-bytes", "5").argv().count("--kv-cache-memory-bytes") == 1

    # page cache eviction: eager safetensors loading (mmapped shards can't be evicted), unless configured
    assert "--safetensors-load-strategy" not in VLLMServer(pack, *argv).argv()
    monkeypatch.setenv("MILABENCH_EVICT_PAGE_CACHE", "/hub")
    full = VLLMServer(pack, *argv).argv()
    assert full[full.index("--safetensors-load-strategy") + 1] == "eager"
    assert full[-2:] == ["--kv-cache-memory-bytes", str(int(98304 * 72 * 1024 * 1.5))]
    assert VLLMServer(pack, *argv, "--safetensors-load-strategy", "lazy").argv().count("--safetensors-load-strategy") == 1
    # unknown model: vLLM's own sizing
    assert serving.kv_cache_budget(config, ["Org/unknown", "--max-model-len", "8192"]) is None


def test_hybrid_models_count_attention_layers_only():
    from milabench.commands.serving import kv_bytes_per_token

    nemotron = {"num_hidden_layers": 52, "num_attention_heads": 32, "num_key_value_heads": 2, "head_dim": 128,
                "hybrid_override_pattern": "M-M-M-MM*-M-M-M*-M-M-M-M*-M-M-M-M*-M-MM*-M-M-M-M*-M-M-M-M*-M-M-M-M*-"}
    assert kv_bytes_per_token(nemotron, "fp8") == 2 * 8 * 2 * 128  # 8 attention layers (*)
    qwen_next = {"num_hidden_layers": 48, "num_attention_heads": 16, "num_key_value_heads": 2, "head_dim": 256,
                 "layer_types": ["linear_attention"] * 3 * 12 + ["full_attention"] * 12}
    assert kv_bytes_per_token(qwen_next, None) == 2 * 12 * 2 * 256 * 2
    gemma = {"text_config": {"num_hidden_layers": 30, "num_attention_heads": 16, "num_key_value_heads": 8,
                             "head_dim": 256}, "architectures": ["Gemma4ForConditionalGeneration"]}
    assert kv_bytes_per_token(gemma, "fp8") == 2 * 30 * 8 * 256


def test_sliding_window_and_shared_kv_layers():
    """Gemma 4: most layers keep only the last `sliding_window` tokens; the full-attention ones have their own
    head size; the last num_kv_shared_layers reuse another layer's KV"""
    from milabench.commands import serving

    gemma31 = {"text_config": {"num_hidden_layers": 60, "num_attention_heads": 32, "num_key_value_heads": 16,
                               "head_dim": 256, "global_head_dim": 512, "num_global_key_value_heads": 4,
                               "sliding_window": 1024, "num_kv_shared_layers": 0,
                               "layer_types": (["sliding_attention"] * 5 + ["full_attention"]) * 10}}
    layout = serving.kv_layout(gemma31, "fp8")
    assert layout == {"full": 2 * 10 * 4 * 512, "sliding": 2 * 50 * 16 * 256, "window": 1024}
    # 32 sequences of 3072 tokens: the sliding layers keep 1024 of each
    assert serving.kv_bytes(layout, 32 * 3072, 32) == 32 * 3072 * layout["full"] + 32 * 1024 * layout["sliding"]
    # a short sequence: every layer keeps all of it
    assert serving.kv_bytes(layout, 500, 1) == 500 * (layout["full"] + layout["sliding"])

    e4b = {"text_config": {"num_hidden_layers": 42, "num_attention_heads": 8, "num_key_value_heads": 2,
                           "head_dim": 256, "global_head_dim": 512, "sliding_window": 512,
                           "num_kv_shared_layers": 18,
                           "layer_types": (["sliding_attention"] * 5 + ["full_attention"]) * 7}}
    layout = serving.kv_layout(e4b, None)
    # 24 layers own a KV cache: 20 sliding, 4 full
    assert layout == {"full": 2 * 4 * 2 * 512 * 2, "sliding": 2 * 20 * 2 * 256 * 2, "window": 512}


def test_server_model_dirs(monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", "/hub")
    server = VLLMServer(_stub(arch="cuda"), "nvidia/Gemma-4-26B-A4B-NVFP4", "--speculative-config",
                        '{"model": "google/gemma-4-26B-A4B-it-assistant", "num_speculative_tokens": 3}')
    assert server.model_dirs() == ["/hub/models--nvidia--Gemma-4-26B-A4B-NVFP4",
                                   "/hub/models--google--gemma-4-26B-A4B-it-assistant"]
