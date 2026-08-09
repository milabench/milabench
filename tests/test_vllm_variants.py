"""Tests for arch-specific serving variants in benchmarks/vllm."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from milabench.capability import is_system_capable_with_reasons
from milabench.merge import merge
from milabench.pack import BasePackage


def _load_vllm_module():
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


def _make_vllm_pack(mod, config):
    class _Core:
        pass

    core = _Core()
    core.pack_path = Path(config["definition"])
    core.dirs = StubPack(config).dirs
    core.constraints = None
    core._nox_runner = None
    core._nox_session = None
    core.install_mark_file = Path("/tmp/mark")
    return mod.VLLM(config, core=core)


SERVING_CONFIG = {
    "name": "serving-moe-kimi-k3-mxfp4-gpus",
    "definition": str(Path(__file__).resolve().parents[1] / "benchmarks" / "vllm"),
    "num_machines": 1,
    "client": {
        "argv": {
            "--dataset-name": "random",
            "--num-prompts": 64,
        }
    },
    "server": {
        "argv": {
            "--served-model-name": "model",
        }
    },
    "variants": {
        "cuda": {
            "server": {
                "backend": "vllm",
                "argv": {
                    "moonshotai/Kimi-K3": True,
                    "--tensor-parallel-size": "8",
                },
            },
            "client": {
                "argv": {
                    "--model": "moonshotai/Kimi-K3",
                }
            },
        },
        "rocm": {
            "server": {
                "backend": "atom",
                "argv": {
                    "--model": "moonshotai/Kimi-K3",
                    "-tp": "8",
                },
            },
            "client": {
                "argv": {
                    "--model": "moonshotai/Kimi-K3",
                }
            },
        },
    },
}


class TestServingVariants:
    def test_cuda_resolves_vllm_server(self):
        mod = _load_vllm_module()
        cfg = merge(
            SERVING_CONFIG,
            {"system": {"arch": "cuda", "gpu": {"count": 8}}},
        )
        pack = _make_vllm_pack(mod, cfg)

        assert pack.server_backend() == "vllm"
        assert pack.server_argv()[0] == "moonshotai/Kimi-K3"
        assert "--tensor-parallel-size" in pack.server_argv()
        assert pack.client_argv() == [
            "--dataset-name",
            "random",
            "--num-prompts",
            "64",
            "--model",
            "moonshotai/Kimi-K3",
        ]

    def test_rocm_resolves_atom_server(self):
        mod = _load_vllm_module()
        cfg = merge(
            SERVING_CONFIG,
            {"system": {"arch": "rocm", "gpu": {"count": 8}}},
        )
        pack = _make_vllm_pack(mod, cfg)

        assert pack.server_backend() == "atom"
        assert "--model" in pack.server_argv()
        assert "moonshotai/Kimi-K3" in pack.server_argv()
        assert "-tp" in pack.server_argv()
        assert "--served-model-name" in pack.server_argv()
        assert "--dtype" not in pack.server_argv()
        assert pack.client_argv()[-2:] == ["--model", "moonshotai/Kimi-K3"]
        assert not pack.uses_ray()

    def test_missing_variant_fails_capability(self):
        pack = StubPack(
            merge(
                SERVING_CONFIG,
                {"system": {"arch": "hpu", "gpu": {"count": 8}}},
            )
        )
        ok, whys = is_system_capable_with_reasons(pack)
        assert ok is False
        assert any("variants['hpu']" in why for why in whys)

    def test_no_variants_keeps_legacy_behavior(self):
        mod = _load_vllm_module()
        cfg = {
            "name": "vllm-single",
            "definition": SERVING_CONFIG["definition"],
            "system": {"arch": "cuda"},
            "server": {
                "argv": {
                    "meta-llama/Meta-Llama-3-8B-Instruct": True,
                    "--tensor-parallel-size": "1",
                }
            },
            "client": {"argv": {"--model": "meta-llama/Meta-Llama-3-8B-Instruct"}},
        }
        pack = _make_vllm_pack(mod, cfg)
        assert pack.server_backend() == "vllm"
        assert pack.server_argv()[0] == "meta-llama/Meta-Llama-3-8B-Instruct"


class TestServerCommandBuilder:
    def _load_server_backends(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "benchmarks"
            / "vllm"
            / "server_backends.py"
        )
        spec = importlib.util.spec_from_file_location("vllm_server_backends", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_build_vllm_command(self):
        mod = self._load_server_backends()

        assert mod.build_server_command(["model", "--dtype", "bf16"], backend="vllm") == [
            "vllm",
            "serve",
            "model",
            "--dtype",
            "bf16",
        ]

    def test_build_atom_command(self):
        mod = self._load_server_backends()

        cmd = mod.build_server_command(["--model", "Kimi-K3"], backend="atom")
        assert cmd[0] == mod.sys.executable
        assert cmd[1].endswith("atom_server_entry.py")
        assert cmd[-2:] == ["--model", "Kimi-K3"]

    def test_reads_backend_from_milabench_config(self, monkeypatch):
        mod = self._load_server_backends()

        cfg = merge(
            SERVING_CONFIG,
            {"system": {"arch": "rocm"}},
        )
        monkeypatch.setenv("MILABENCH_CONFIG", json.dumps(cfg))
        assert mod.resolved_server_backend() == "atom"

    def test_unknown_backend_raises(self):
        mod = self._load_server_backends()

        with pytest.raises(mod.InferenceServerError):
            mod.build_server_command([], backend="unknown")
