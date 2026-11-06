from __future__ import annotations

import logging
from types import ModuleType
from typing import Any

import pynvml  # provided by nvidia-ml-py, which voir depends on

from ..collector import NodeCollector

log = logging.getLogger(__name__)


def _text(value: str | bytes) -> str:
    return value.decode() if isinstance(value, bytes) else value


def nvml_error_text(code: int) -> str:
    """An NVML return code's text from pynvml's table (e.g. "Not Supported"), without calling into NVML.

    Takes the code rather than the NVMLError, so a raw call's return code reads the same way.
    """
    return str(pynvml.NVMLError._errcode_to_string.get(code, f"NVML error {code}"))


class Gpu:
    """One GPU's NVML handle."""

    def __init__(self, index: int, handle: Any, uuid: str) -> None:
        self.index = index
        self.handle = handle
        self.uuid = uuid
        #: Queries this GPU answered "not supported", which are not asked again. Every NVML
        #: collector shares this GPU, so a key must name the query, not the collector asking.
        self.query_unsupported: set[str] = set()


class NvmlGpus:
    """The NVML session and the GPUs it sees, listed once and shared by every NVML collector.

    The handles are valid from start() until close(), so this object owns both.
    """

    def __init__(self, gpus: list[Gpu]) -> None:
        self.gpus = gpus
        #: The pynvml module start() initialized, for close(); None once closed, or for a list built directly.
        self._session: ModuleType | None = None

    @classmethod
    def start(cls, nvml: ModuleType = pynvml) -> NvmlGpus | None:
        """Start NVML and list its GPUs; None when NVML refuses to start."""
        # Each nvmlInit increments _nvmlLib_refcount and close() releases that one reference, so
        # other NVML users in the process (e.g. torch) keep their session. The wrapper returns None.
        try:
            nvml.nvmlInit()
        except nvml.NVMLError as err:
            log.debug(f"NVML did not start ({nvml_error_text(err.value)}).")
            return None
        gpus = cls.discover(nvml)
        gpus._session = nvml
        log.debug(f"NVML started; it sees {len(gpus.gpus)} GPUs: {gpus.uuid_of()}.")
        return gpus

    def close(self) -> None:
        """End the session: every handle in `gpus` is invalid after this. Safe to call twice."""
        nvml = self._session
        self._session = None  # cleared before shutting down, so a second call does nothing
        if nvml is None:
            return
        try:
            nvml.nvmlShutdown()
        except nvml.NVMLError as err:
            log.error(f"NVML did not shut down ({nvml_error_text(err.value)}).")
            return
        log.debug("NVML shut down.")

    @classmethod
    def discover(cls, nvml: ModuleType = pynvml) -> NvmlGpus:
        """Every GPU NVML can open; one that fails is skipped. `nvml`: already initialized."""
        try:
            count = nvml.nvmlDeviceGetCount()
        except nvml.NVMLError as err:
            log.warning(f"NVML started but would not count the GPUs ({nvml_error_text(err.value)}); none are listed.")
            return cls([])
        gpus = []
        for index in range(count):
            try:
                handle = nvml.nvmlDeviceGetHandleByIndex(index)
                uuid = _text(nvml.nvmlDeviceGetUUID(handle))
            except nvml.NVMLError as err:
                log.warning(f"GPU {index}: NVML would not open it ({nvml_error_text(err.value)}); it is left out.")
                continue
            gpus.append(Gpu(index, handle, uuid))
        return cls(gpus)

    def uuid_of(self) -> dict[str, str]:
        """index -> UUID, for the status file."""
        return {str(gpu.index): gpu.uuid for gpu in sorted(self.gpus, key=lambda gpu: gpu.index)}


class NvmlCollector(NodeCollector):
    """A collector over the GPUs NVML sees.

    Every collector shares one NvmlGpus, the session and its GPU list. The registry starts
    it before the first probe, hands it to each collector as ``nvml_gpus``, and closes it
    after every collector has closed, so a collector that keeps no GPUs does not take the
    driver down with it. Each collector keeps the GPUs its ``_open()`` accepts.
    """

    def __init__(self, nvml: ModuleType = pynvml) -> None:
        """`nvml`: already initialized."""
        #: The pynvml module. The registry starts it.
        self._nvml = nvml
        #: Every GPU NVML sees, set by the registry before probe(); None when NVML did not start.
        #: Kept after close(), since the status file is written after it.
        self.nvml_gpus: NvmlGpus | None = None
        #: The GPUs this collector samples.
        self._gpus: list[Gpu] = []
        #: (what, NVML code) pairs already logged by _log_nvml_error.
        self._nvml_errors_logged: set[tuple[str, int]] = set()

    def probe(self) -> bool:
        """Open every GPU _open() accepts; False when NVML is not started or there are no GPUs."""
        if self.nvml_gpus and self._init():
            self._gpus = [
                gpu
                for gpu in self.nvml_gpus.gpus
                if self._open(gpu)
            ]
        if self.nvml_gpus:
            log.debug(f"{self.name}: samples {len(self._gpus)} of the {len(self.nvml_gpus.gpus)} GPUs NVML sees.")
        if not self._gpus:
            self.close()
        return bool(self._gpus)

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
        return {"gpus": self.nvml_gpus.uuid_of() if self.nvml_gpus else {}}

    def _query(self, gpu: Gpu, key: str, query: Any, *args: Any) -> Any:
        """One NVML query, None when it fails; never asked again once "not supported"."""
        if not query or key in gpu.query_unsupported:
            return None
        try:
            return query(gpu.handle, *args)
        except self._nvml.NVMLError as err:
            self._log_nvml_error(f"GPU {gpu.index} {key}", err.value)
            if err.value == self._nvml.NVML_ERROR_NOT_SUPPORTED:
                gpu.query_unsupported.add(key)
            return None

    def _log_nvml_error(self, what: str, code: int) -> None:
        """Log an NVML failure once per (what, code), since most would repeat on every tick.

        At debug, as most are a query this GPU does not support; a warning when the GPU itself is in trouble.
        """
        if (what, code) in self._nvml_errors_logged:
            return
        self._nvml_errors_logged.add((what, code))
        log.log(
            logging.WARNING if code in (pynvml.NVML_ERROR_GPU_IS_LOST, pynvml.NVML_ERROR_RESET_REQUIRED) else logging.DEBUG,
            f"{self.name}: {what} failed ({nvml_error_text(code)}); logged once.",
        )

    def close(self) -> None:
        """Forget the opened GPUs. The shared session stays up."""
        self._gpus = []
