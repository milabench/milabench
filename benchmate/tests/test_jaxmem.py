from unittest.mock import MagicMock, patch

from benchmate.jaxmem import jaxmem_fetcher, memory_peak_fetcher
from benchmate.monitor import _jaxmem_kwargs
from benchmate.toggles import jaxmem_enabled


def _jax_device(stats):
    device = MagicMock()
    device.memory_stats.return_value = stats
    return device


def _jax_module(devices):
    jax = MagicMock()
    jax.devices.return_value = devices
    return jax


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

        with patch.dict("sys.modules", {"jax": jax}):
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

        with patch.dict("sys.modules", {"jax": jax}):
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

        with patch.dict("sys.modules", {"jax": jax}):
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
            with patch("builtins.__import__", side_effect=boom):
                assert jaxmem_fetcher()() == {}
        finally:
            sys.modules.update(saved)

    def test_returns_empty_on_fetch_error(self):
        device = MagicMock()
        device.memory_stats.side_effect = RuntimeError("not ready")
        jax = _jax_module([device])

        with patch.dict("sys.modules", {"jax": jax}):
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

        with patch.dict("sys.modules", {"jax": jax}):
            result = jaxmem_fetcher()()

        assert result == {
            1: {
                "allocated": 2.0,
                "reserved": 0.0,
                "max_allocated": 4.0,
                "max_reserved": 0.0,
            }
        }


class TestMemoryPeakFetcher:
    def test_returns_max_allocated(self):
        devices = [
            _jax_device({"peak_bytes_in_use": 3 * 1024**2}),
            _jax_device({"peak_bytes_in_use": 7 * 1024**2}),
        ]
        jax = _jax_module(devices)

        with patch.dict("sys.modules", {"jax": jax}):
            assert memory_peak_fetcher()() == 7.0
