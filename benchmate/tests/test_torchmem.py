from unittest.mock import MagicMock, patch

from benchmate.monitor import _torchmem_kwargs
from benchmate.toggles import torchmem_enabled
from benchmate.torchmem import torchmem_fetcher


def _cuda_torch(*, available=True, cuda_version="12.1", hip=None, device_count=1):
    torch = MagicMock()
    torch.cuda.is_available.return_value = available
    torch.cuda.device_count.return_value = device_count
    torch.version.cuda = cuda_version
    torch.version.hip = hip
    return torch


class TestTorchmemEnabled:
    def test_enabled_by_default(self, monkeypatch):
        monkeypatch.delenv("BENCHMATE_TORCHMEM", raising=False)
        assert torchmem_enabled() is True

    def test_disabled_with_zero(self, monkeypatch):
        monkeypatch.setenv("BENCHMATE_TORCHMEM", "0")
        assert torchmem_enabled() is False

    def test_kwargs_omitted_when_disabled(self, monkeypatch):
        monkeypatch.setenv("BENCHMATE_TORCHMEM", "0")
        assert _torchmem_kwargs() == {}

    def test_kwargs_present_when_enabled(self, monkeypatch):
        monkeypatch.setenv("BENCHMATE_TORCHMEM", "1")
        kwargs = _torchmem_kwargs()
        assert set(kwargs) == {"torchmem"}
        assert callable(kwargs["torchmem"])


class TestTorchmemFetcher:
    def test_payload_shape_all_devices(self):
        torch = _cuda_torch(device_count=2)
        torch.cuda.memory_allocated.side_effect = lambda i: (i + 1) * 1024**2
        torch.cuda.memory_reserved.side_effect = lambda i: (i + 2) * 1024**2
        torch.cuda.max_memory_allocated.side_effect = lambda i: (i + 3) * 1024**2
        torch.cuda.max_memory_reserved.side_effect = lambda i: (i + 4) * 1024**2

        with patch.dict("sys.modules", {"torch": torch}):
            result = torchmem_fetcher()()

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
        torch.cuda.memory_allocated.assert_any_call(0)
        torch.cuda.memory_reserved.assert_any_call(0)
        torch.cuda.max_memory_allocated.assert_any_call(0)
        torch.cuda.max_memory_reserved.assert_any_call(0)

    def test_works_on_rocm(self):
        torch = _cuda_torch(cuda_version=None, hip="6.0", device_count=1)
        torch.cuda.memory_allocated.return_value = 3 * 1024**2
        torch.cuda.memory_reserved.return_value = 4 * 1024**2
        torch.cuda.max_memory_allocated.return_value = 5 * 1024**2
        torch.cuda.max_memory_reserved.return_value = 6 * 1024**2

        with patch.dict("sys.modules", {"torch": torch}):
            result = torchmem_fetcher()()

        assert result == {
            0: {
                "allocated": 3.0,
                "reserved": 4.0,
                "max_allocated": 5.0,
                "max_reserved": 6.0,
            }
        }

    def test_single_device(self):
        torch = _cuda_torch(device_count=4)
        torch.cuda.memory_allocated.return_value = 5 * 1024**2
        torch.cuda.memory_reserved.return_value = 8 * 1024**2
        torch.cuda.max_memory_allocated.return_value = 9 * 1024**2
        torch.cuda.max_memory_reserved.return_value = 10 * 1024**2

        with patch.dict("sys.modules", {"torch": torch}):
            result = torchmem_fetcher(device=1)()

        assert result == {
            1: {
                "allocated": 5.0,
                "reserved": 8.0,
                "max_allocated": 9.0,
                "max_reserved": 10.0,
            }
        }
        torch.cuda.memory_allocated.assert_called_once_with(1)
        torch.cuda.memory_reserved.assert_called_once_with(1)
        torch.cuda.max_memory_allocated.assert_called_once_with(1)
        torch.cuda.max_memory_reserved.assert_called_once_with(1)

    def test_returns_empty_when_unavailable(self):
        torch = _cuda_torch(available=False)
        torch.cuda.memory_allocated = MagicMock()
        torch.cuda.memory_reserved = MagicMock()

        with patch.dict("sys.modules", {"torch": torch}):
            assert torchmem_fetcher()() == {}

        torch.cuda.memory_allocated.assert_not_called()

    def test_returns_empty_without_cuda_or_hip(self):
        torch = _cuda_torch(cuda_version=None, hip=None)

        with patch.dict("sys.modules", {"torch": torch}):
            assert torchmem_fetcher()() == {}

    def test_returns_empty_when_import_fails(self):
        import builtins
        import sys

        real_import = builtins.__import__

        def boom(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "torch" or name.startswith("torch."):
                raise ImportError("no torch")
            return real_import(name, globals, locals, fromlist, level)

        saved = {
            k: sys.modules.pop(k)
            for k in list(sys.modules)
            if k == "torch" or k.startswith("torch.")
        }
        try:
            with patch("builtins.__import__", side_effect=boom):
                assert torchmem_fetcher()() == {}
        finally:
            sys.modules.update(saved)

    def test_returns_empty_on_fetch_error(self):
        torch = _cuda_torch(device_count=1)
        torch.cuda.memory_allocated.side_effect = RuntimeError("cuda not init")

        with patch.dict("sys.modules", {"torch": torch}):
            assert torchmem_fetcher()() == {}
