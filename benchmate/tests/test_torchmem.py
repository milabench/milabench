from unittest.mock import MagicMock, patch

import torchcompat.core as accelerator

from benchmate.monitor import _torchmem_kwargs
from benchmate.toggles import torchmem_enabled
from benchmate.torchmem import torchmem_fetcher


def _device_module(*, device_count=1, alloc=lambda i: 0, res=lambda i: 0,
                   max_alloc=lambda i: 0, max_res=lambda i: 0):
    module = MagicMock()
    module.device_count.return_value = device_count
    module.memory_allocated.side_effect = alloc
    module.memory_reserved.side_effect = res
    module.max_memory_allocated.side_effect = max_alloc
    module.max_memory_reserved.side_effect = max_res
    return module


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
        module = _device_module(
            device_count=2,
            alloc=lambda i: (i + 1) * 1024**2,
            res=lambda i: (i + 2) * 1024**2,
            max_alloc=lambda i: (i + 3) * 1024**2,
            max_res=lambda i: (i + 4) * 1024**2,
        )
        with patch.object(accelerator, "device_module", module):
            result = torchmem_fetcher()()

        assert result == {
            0: {"allocated": 1.0, "reserved": 2.0, "max_allocated": 3.0, "max_reserved": 4.0},
            1: {"allocated": 2.0, "reserved": 3.0, "max_allocated": 4.0, "max_reserved": 5.0},
        }
        module.memory_allocated.assert_any_call(0)
        module.memory_reserved.assert_any_call(0)
        module.max_memory_allocated.assert_any_call(0)
        module.max_memory_reserved.assert_any_call(0)

    def test_single_device(self):
        module = _device_module(
            device_count=4,
            alloc=lambda i: 5 * 1024**2,
            res=lambda i: 8 * 1024**2,
            max_alloc=lambda i: 9 * 1024**2,
            max_res=lambda i: 10 * 1024**2,
        )
        with patch.object(accelerator, "device_module", module):
            result = torchmem_fetcher(device=1)()

        assert result == {
            1: {"allocated": 5.0, "reserved": 8.0, "max_allocated": 9.0, "max_reserved": 10.0}
        }
        module.memory_allocated.assert_called_once_with(1)
        module.max_memory_reserved.assert_called_once_with(1)

    def test_returns_empty_when_no_namespace(self):
        with patch.object(accelerator, "device_module", None):
            assert torchmem_fetcher()() == {}

    def test_returns_empty_when_namespace_has_no_memory_api(self):
        # CPU fallback namespace exposes no allocator stats.
        with patch.object(accelerator, "device_module", MagicMock(spec=[])):
            assert torchmem_fetcher()() == {}

    def test_returns_empty_on_fetch_error(self):
        module = _device_module(device_count=1, alloc=lambda i: (_ for _ in ()).throw(RuntimeError("nope")))
        with patch.object(accelerator, "device_module", module):
            assert torchmem_fetcher()() == {}
