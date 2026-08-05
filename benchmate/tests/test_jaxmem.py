from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from benchmate.jaxmem import JaxGpuRequiredError, jaxmem_fetcher, memory_peak_fetcher
from benchmate.monitor import _jaxmem_kwargs
from benchmate.toggles import jaxmem_enabled


def _jax_device(stats, platform="gpu", kind=""):
    device = MagicMock()
    device.memory_stats.return_value = stats
    device.platform = platform
    device.device_kind = kind
    return device


def _jax_module(devices, backend="gpu"):
    jax = MagicMock()
    jax.devices.return_value = devices
    jax.default_backend.return_value = backend
    return jax


@contextmanager
def _voir_arch(arch):
    smi = MagicMock()
    smi.arch = arch
    with patch("voir.instruments.gpu.select_backend", return_value=smi):
        yield


class TestJaxmemEnabled:
    def test_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("BENCHMATE_JAXMEM", raising=False)
        assert jaxmem_enabled() is False

    def test_enabled_with_one(self, monkeypatch):
        monkeypatch.setenv("BENCHMATE_JAXMEM", "1")
        assert jaxmem_enabled() is True

    def test_kwargs_omitted_when_disabled(self, monkeypatch):
        monkeypatch.setenv("BENCHMATE_JAXMEM", "0")
        assert _jaxmem_kwargs() == {}

    def test_kwargs_present_when_enabled(self, monkeypatch):
        monkeypatch.setenv("BENCHMATE_JAXMEM", "1")
        kwargs = _jaxmem_kwargs()
        assert set(kwargs) == {"jaxmem"}
        assert callable(kwargs["jaxmem"])


class TestJaxmemFetcher:
    def test_payload_shape_all_devices(self):
        devices = [
            _jax_device(
                {
                    "bytes_in_use": 1 * 1024**2,
                    "bytes_reserved": 2 * 1024**2,
                    "peak_bytes_in_use": 3 * 1024**2,
                    "peak_bytes_reserved": 4 * 1024**2,
                }
            ),
            _jax_device(
                {
                    "bytes_in_use": 2 * 1024**2,
                    "bytes_reserved": 3 * 1024**2,
                    "peak_bytes_in_use": 4 * 1024**2,
                    "peak_bytes_reserved": 5 * 1024**2,
                }
            ),
        ]
        jax = _jax_module(devices)

        with _voir_arch("cpu"), patch.dict("sys.modules", {"jax": jax}):
            result = jaxmem_fetcher()()

        assert result == {
            0: {
                "allocated": 1.0,
                "reserved": 2.0,
                "max_allocated": 3.0,
                "max_reserved": 4.0,
            },
            1: {
                "allocated": 2.0,
                "reserved": 3.0,
                "max_allocated": 4.0,
                "max_reserved": 5.0,
            },
        }

    def test_single_device(self):
        devices = [
            _jax_device({"bytes_in_use": 1, "bytes_reserved": 2, "peak_bytes_in_use": 3, "peak_bytes_reserved": 4}),
            _jax_device(
                {
                    "bytes_in_use": 5 * 1024**2,
                    "bytes_reserved": 8 * 1024**2,
                    "peak_bytes_in_use": 9 * 1024**2,
                    "peak_bytes_reserved": 10 * 1024**2,
                }
            ),
        ]
        jax = _jax_module(devices)

        with _voir_arch("cpu"), patch.dict("sys.modules", {"jax": jax}):
            result = jaxmem_fetcher(device=1)()

        assert result == {
            1: {
                "allocated": 5.0,
                "reserved": 8.0,
                "max_allocated": 9.0,
                "max_reserved": 10.0,
            }
        }
        devices[1].memory_stats.assert_called_once()
        devices[0].memory_stats.assert_not_called()

    def test_returns_empty_when_no_devices(self):
        jax = _jax_module([])

        with _voir_arch("cpu"), patch.dict("sys.modules", {"jax": jax}):
            assert jaxmem_fetcher()() == {}

    def test_returns_empty_when_import_fails(self):
        import builtins
        import sys

        real_import = builtins.__import__

        def boom(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "jax" or name.startswith("jax."):
                raise ImportError("no jax")
            return real_import(name, globals, locals, fromlist, level)

        saved = {
            k: sys.modules.pop(k)
            for k in list(sys.modules)
            if k == "jax" or k.startswith("jax.")
        }
        try:
            with _voir_arch("cpu"), patch("builtins.__import__", side_effect=boom):
                assert jaxmem_fetcher()() == {}
        finally:
            sys.modules.update(saved)

    def test_returns_empty_on_fetch_error(self):
        device = MagicMock()
        device.platform = "gpu"
        device.device_kind = ""
        device.memory_stats.side_effect = RuntimeError("not ready")
        jax = _jax_module([device])

        with _voir_arch("cpu"), patch.dict("sys.modules", {"jax": jax}):
            assert jaxmem_fetcher()() == {}

    def test_skips_devices_without_stats(self):
        devices = [
            _jax_device(None),
            _jax_device(
                {
                    "bytes_in_use": 2 * 1024**2,
                    "bytes_reserved": 0,
                    "peak_bytes_in_use": 4 * 1024**2,
                    "peak_bytes_reserved": 0,
                }
            ),
        ]
        jax = _jax_module(devices)

        with _voir_arch("cpu"), patch.dict("sys.modules", {"jax": jax}):
            result = jaxmem_fetcher()()

        assert result == {
            1: {
                "allocated": 2.0,
                "reserved": 0.0,
                "max_allocated": 4.0,
                "max_reserved": 0.0,
            }
        }


class TestJaxmemGpuArchRequired:
    def test_cpu_arch_allows_cpu_devices(self):
        jax = _jax_module([_jax_device(None, platform="cpu")], backend="cpu")

        with _voir_arch("cpu"), patch.dict("sys.modules", {"jax": jax}):
            assert jaxmem_fetcher()() == {}

    def test_select_backend_failure_allows_cpu_devices(self):
        jax = _jax_module([_jax_device(None, platform="cpu")], backend="cpu")

        with patch(
            "voir.instruments.gpu.select_backend",
            side_effect=RuntimeError("no smi"),
        ), patch.dict("sys.modules", {"jax": jax}):
            assert jaxmem_fetcher()() == {}

    @pytest.mark.parametrize("arch", ["rocm", "cuda", "xpu", "hpu"])
    def test_gpu_arch_raises_on_cpu_backend(self, arch):
        device = _jax_device(None, platform="cpu")
        jax = _jax_module([device], backend="cpu")

        with _voir_arch(arch), patch.dict("sys.modules", {"jax": jax}):
            with pytest.raises(JaxGpuRequiredError, match="requires a matching JAX device"):
                jaxmem_fetcher()()

    def test_rocm_arch_rejects_cuda_device(self):
        device = _jax_device(
            {"bytes_in_use": 1, "peak_bytes_in_use": 1},
            platform="gpu",
            kind="NVIDIA H100",
        )
        jax = _jax_module([device], backend="gpu")

        with _voir_arch("rocm"), patch.dict("sys.modules", {"jax": jax}):
            with pytest.raises(JaxGpuRequiredError, match="voir GPU arch 'rocm'"):
                jaxmem_fetcher()()

    def test_rocm_arch_ok_with_rocm_device(self):
        devices = [
            _jax_device(
                {
                    "bytes_in_use": 2 * 1024**2,
                    "bytes_reserved": 0,
                    "peak_bytes_in_use": 4 * 1024**2,
                    "peak_bytes_reserved": 0,
                },
                platform="gpu",
                kind="AMD Instinct MI355X",
            )
        ]
        jax = _jax_module(devices, backend="gpu")

        with _voir_arch("rocm"), patch.dict("sys.modules", {"jax": jax}):
            result = jaxmem_fetcher()()

        assert result[0]["max_allocated"] == 4.0

    def test_gpu_arch_raises_when_import_fails(self):
        import builtins
        import sys

        real_import = builtins.__import__

        def boom(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "jax" or name.startswith("jax."):
                raise ImportError("no jax")
            return real_import(name, globals, locals, fromlist, level)

        saved = {
            k: sys.modules.pop(k)
            for k in list(sys.modules)
            if k == "jax" or k.startswith("jax.")
        }
        try:
            with _voir_arch("rocm"), patch("builtins.__import__", side_effect=boom):
                fetch = jaxmem_fetcher()
                with pytest.raises(JaxGpuRequiredError, match="importing jax failed"):
                    fetch()
        finally:
            sys.modules.update(saved)

    def test_memory_peak_raises_when_gpu_required(self):
        jax = _jax_module([_jax_device(None, platform="cpu")], backend="cpu")

        with _voir_arch("rocm"), patch.dict("sys.modules", {"jax": jax}):
            with pytest.raises(JaxGpuRequiredError):
                memory_peak_fetcher()()


class TestMemoryPeakFetcher:
    def test_returns_max_allocated(self):
        devices = [
            _jax_device({"peak_bytes_in_use": 3 * 1024**2}, kind="AMD Instinct"),
            _jax_device({"peak_bytes_in_use": 7 * 1024**2}, kind="AMD Instinct"),
        ]
        jax = _jax_module(devices)

        with _voir_arch("cpu"), patch.dict("sys.modules", {"jax": jax}):
            assert memory_peak_fetcher()() == 7.0
