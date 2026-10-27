from __future__ import annotations

from typing import Any

from .nvml import NvmlCollector

# TODO: Upstream SM clock and throttle reasons to voir gpudata.


class GpuCollector(NvmlCollector):
    name = "gpu"
    columns = (
        "gpu",
        "gpu_uuid",
        "sm_clock_mhz",
        "clocks_event_reasons",
        "power_w",
        "temp_c",
        "util",
    )

    def __init__(self) -> None:
        super().__init__()
        self._reasons: Any = None

    def _init(self) -> bool:
        # The newer name, with the older one (kept as an alias) for older bindings.
        self._reasons = (
            getattr(self._nvml, "nvmlDeviceGetCurrentClocksEventReasons", None)
            or self._nvml.nvmlDeviceGetCurrentClocksThrottleReasons
        )
        return True

    def sample(self, now_s: float) -> list[dict[str, Any]]:
        """One row per GPU."""
        pynvml = self._nvml
        rows = []
        for gpu in self._gpus:
            reasons = self._query(gpu, "reasons", self._reasons)
            power = self._query(gpu, "power", pynvml.nvmlDeviceGetPowerUsage)
            util = self._query(gpu, "util", pynvml.nvmlDeviceGetUtilizationRates)
            rows.append(
                {
                    "gpu": gpu.index,
                    "gpu_uuid": gpu.uuid,
                    "sm_clock_mhz": self._query(gpu, "sm_clock", pynvml.nvmlDeviceGetClockInfo, pynvml.NVML_CLOCK_SM),
                    "clocks_event_reasons": None if reasons is None else f"0x{reasons:x}",
                    "power_w": None if power is None else power / 1000,
                    "temp_c": self._query(gpu, "temp", pynvml.nvmlDeviceGetTemperature, pynvml.NVML_TEMPERATURE_GPU),
                    "util": None if util is None else util.gpu / 100,
                }
            )
        return rows
