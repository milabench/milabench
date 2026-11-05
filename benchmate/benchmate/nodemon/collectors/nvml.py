from __future__ import annotations

from collections.abc import Iterator
from types import ModuleType
from typing import Any

import pynvml  # provided by nvidia-ml-py, which voir depends on

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
    """A collector over the GPUs NVML sees.

    Every collector uses the one pynvml session. The registry calls ``nvmlInit`` once
    before probing and ``nvmlShutdown`` once after every collector has closed, so a
    collector that keeps no GPUs does not take the driver down with it.
    """

    def __init__(self, nvml: ModuleType = pynvml) -> None:
        """`nvml`: already initialized."""
        #: The pynvml module. The registry starts it.
        self._nvml = nvml
        self._gpus: list[Gpu] = []
        #: index -> UUID, for the status file. Survives close(), which the status file is written after.
        self._uuid_of: dict[int, str] = {}

    def probe(self) -> bool:
        """Open every GPU _open() accepts; False when NVML is not started or there are no GPUs."""
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
        try:
            count = self._nvml.nvmlDeviceGetCount()
        except self._nvml.NVMLError:
            return
        for index in range(count):
            try:
                handle = self._nvml.nvmlDeviceGetHandleByIndex(index)
                uuid = _text(self._nvml.nvmlDeviceGetUUID(handle))
            except self._nvml.NVMLError:
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
        """Forget the opened GPUs. The shared session stays up."""
        self._gpus = []
