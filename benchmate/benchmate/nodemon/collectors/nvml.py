from __future__ import annotations

from collections.abc import Iterator
from types import ModuleType
from typing import Any

from ..collector import NodeCollector


def _text(value: str | bytes) -> str:
    return value.decode() if isinstance(value, bytes) else value


class Gpu:
    """One GPU's NVML handle."""

    def __init__(self, index: int, handle: Any, uuid: str) -> None:
        self.index = index
        self.handle = handle
        self.uuid = uuid
        #: Queries this GPU answered "not supported", which are not asked again.
        self.query_unsupported: set[str] = set()


class NvmlCollector(NodeCollector):
    """A collector over the GPUs NVML sees, with NVML initialized in probe() and shut down in close()."""

    def __init__(self) -> None:
        #: The pynvml module, imported in probe() so a host without the bindings still runs.
        self._nvml: ModuleType | None = None
        self._gpus: list[Gpu] = []
        #: index -> UUID, for the status file. Survives close(), which the status file is written after.
        self._uuid_of: dict[int, str] = {}

    def probe(self) -> bool:
        """Initialize NVML and open every GPU _open() accepts; False without bindings, a driver or GPUs."""
        try:
            import pynvml  # provided by nvidia-ml-py, which voir depends on
        except ImportError:
            return False
        try:
            pynvml.nvmlInit()
        except pynvml.NVMLError:
            return False
        self._nvml = pynvml
        if self._init():
            self._gpus = [
                gpu
                for gpu in self._devices()
                if self._open(gpu)
            ]
        if not self._gpus:
            self.close()
        return bool(self._gpus)

    def _devices(self) -> Iterator[Gpu]:
        """Every GPU NVML can open; one that fails is skipped."""
        pynvml = self._nvml
        try:
            count = pynvml.nvmlDeviceGetCount()
        except pynvml.NVMLError:
            return
        for index in range(count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                uuid = _text(pynvml.nvmlDeviceGetUUID(handle))
            except pynvml.NVMLError:
                continue
            self._uuid_of[index] = uuid
            yield Gpu(index, handle, uuid)

    def _init(self) -> bool:
        """Set up what every GPU shares, once NVML is initialized; False when unavailable."""
        # The base case: nothing shared to set up. NvlinkCollector overrides it.
        return True

    def _open(self, gpu: Gpu) -> bool:
        """Set up one GPU; False leaves it out."""
        # The base case: keep every GPU. NvlinkCollector overrides it.
        return True

    def describe(self) -> dict[str, Any]:
        """The GPUs this collector opened, so the rows can carry an index instead of a UUID."""
        return {
            "gpus": {
                str(index): uuid
                for index, uuid
                in sorted(self._uuid_of.items())
            }
        }

    def _query(self, gpu: Gpu, key: str, query: Any, *args: Any) -> Any:
        """One NVML query, None when it fails; never asked again once "not supported"."""
        if not query or key in gpu.query_unsupported:
            return None
        try:
            return query(gpu.handle, *args)
        except self._nvml.NVMLError as err:
            if err.value == self._nvml.NVML_ERROR_NOT_SUPPORTED:
                gpu.query_unsupported.add(key)
            return None

    def close(self) -> None:
        """Shut NVML down."""
        if self._nvml:
            try:
                self._nvml.nvmlShutdown()
            except self._nvml.NVMLError:
                pass
        self._nvml = None
        self._gpus = []
