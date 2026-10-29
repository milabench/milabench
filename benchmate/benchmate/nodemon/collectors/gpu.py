from __future__ import annotations

from enum import IntFlag
from typing import Any

from ..collector import CollectorMetric, MetricKind
from .nvml import Gpu, NvmlCollector

# TODO: Upstream SM clock and throttle reasons to voir gpudata.


class ClockReason(IntFlag):
    """Why a GPU's SM clock is below its maximum: NVML's nvmlClocksEventReason* bits.

    The `clocks_event_reasons` column holds them as hex; readers decode it with parse().
    """

    GPU_IDLE = 0x1
    APPLICATIONS_CLOCKS_SETTING = 0x2
    SW_POWER_CAP = 0x4
    HW_SLOWDOWN = 0x8
    SYNC_BOOST = 0x10
    SW_THERMAL_SLOWDOWN = 0x20
    HW_THERMAL_SLOWDOWN = 0x40
    HW_POWER_BRAKE_SLOWDOWN = 0x80
    DISPLAY_CLOCK_SETTING = 0x100

    @classmethod
    def parse(cls, text: str | None) -> ClockReason:
        """The reasons from gpu.tsv's hex cell; none when the cell is empty or not a number."""
        try:
            return cls(int(text or "0", 16))
        except ValueError:
            return cls(0)

    def names(self) -> str:
        """The reasons set, by name: a combined flag has no name of its own before Python 3.11."""
        return "|".join(name for name, reason in ClockReason.__members__.items() if reason & self) or "-"


class GpuCollector(NvmlCollector):
    name = "gpu"
    metrics = (
        CollectorMetric("gpu", MetricKind.DEVICE, "", "NVML index of the GPU"),
        CollectorMetric("sm_clock_mhz", MetricKind.SAMPLE, "MHz", "SM clock at the tick; below its maximum when the GPU is slowed"),
        CollectorMetric(
            "clocks_event_reasons", MetricKind.SAMPLE, "",
            "Why the SM clock is below its maximum, as hex NVML clock event reason bits (see ClockReason)",
        ),
        CollectorMetric("power_w", MetricKind.SAMPLE, "W", "Board power draw"),
        CollectorMetric("temp_c", MetricKind.SAMPLE, "C", "GPU temperature"),
        CollectorMetric("util", MetricKind.SAMPLE, "fraction", "Utilization at the tick: one reading of a coarse counter"),
        CollectorMetric(
            "util_mean", MetricKind.SAMPLE, "fraction",
            "Mean utilization over the interval, from the samples the driver buffered: the duty cycle",
        ),
        CollectorMetric("util_min", MetricKind.SAMPLE, "fraction", "Lowest buffered utilization sample over the interval"),
        CollectorMetric("util_samples", MetricKind.SAMPLE, "count", "How many buffered samples util_mean and util_min cover"),
        CollectorMetric("xid", MetricKind.SAMPLE, "", "XID error codes the driver delivered since the last row, separated by ';'"),
    )

    def __init__(self) -> None:
        super().__init__()
        self._reasons: Any = None
        self._events: Any = None
        #: The last utilization sample already read, per GPU, so each call asks for newer ones only.
        self._seen: dict[int, int] = {}

    def _init(self) -> bool:
        # The newer name, with the older one (kept as an alias) for older bindings.
        self._reasons = (
            getattr(self._nvml, "nvmlDeviceGetCurrentClocksEventReasons", None)
            or self._nvml.nvmlDeviceGetCurrentClocksThrottleReasons
        )
        self._events = self._event_set()
        return True

    def _event_set(self) -> Any:
        """A set to deliver critical XID errors into, or None when NVML has no event support here."""
        try:
            return self._nvml.nvmlEventSetCreate()
        except self._nvml.NVMLError:
            return None

    def _open(self, gpu: Gpu) -> bool:
        """Subscribe this GPU to XID events; one that refuses is still sampled, it just reports none."""
        if self._events is None:
            return True
        try:
            self._nvml.nvmlDeviceRegisterEvents(gpu.handle, self._nvml.nvmlEventTypeXidCriticalError, self._events)
        except self._nvml.NVMLError:
            pass
        return True

    #: Events drained per tick at most, so a storm can't hold the sampling thread.
    max_events = 64

    def _xids(self) -> dict[int, list[str]]:
        """The XIDs delivered since the last call, per GPU index; the wait returns at once with timeout 0."""
        seen: dict[int, list[str]] = {}
        if self._events is None:
            return seen
        for _ in range(self.max_events):
            try:
                event = self._nvml.nvmlEventSetWait(self._events, 0)
            except self._nvml.NVMLError:
                # NVML_ERROR_TIMEOUT: the queue is empty, which is the normal case.
                return seen
            if not event.eventType & self._nvml.nvmlEventTypeXidCriticalError:
                continue
            try:
                index = self._nvml.nvmlDeviceGetIndex(event.device)
            except self._nvml.NVMLError:
                continue
            seen.setdefault(index, []).append(str(event.eventData))
        return seen

    def _duty_cycle(self, gpu: Gpu) -> dict[str, Any]:
        """Utilization over the whole interval, from the samples the driver buffered since the last call.

        ``util`` alone is one reading of a coarse counter; these say how much of
        the interval the GPU was busy, which is what separates a straggler.
        """
        empty = {"util_mean": None, "util_min": None, "util_samples": None}
        values = self._query(gpu, "samples", self._samples, gpu)
        if not values:
            return empty
        self._seen[gpu.index] = max(sample.timeStamp for sample in values)
        busy = [sample.sampleValue.uiVal / 100 for sample in values]
        return {
            "util_mean": sum(busy) / len(busy),
            "util_min": min(busy),
            "util_samples": len(busy),
        }

    def _samples(self, handle: Any, gpu: Gpu) -> list[Any]:
        """The utilization samples newer than the last one read; empty when there are none."""
        try:
            _, values = self._nvml.nvmlDeviceGetSamples(
                handle, self._nvml.NVML_GPU_UTILIZATION_SAMPLES, self._seen.get(gpu.index, 0)
            )
        except self._nvml.NVMLError as err:
            if err.value == self._nvml.NVML_ERROR_NOT_FOUND:  # nothing new since the last call
                return []
            raise
        return list(values)

    def sample(self, now_s: float) -> list[dict[str, Any]]:
        """One row per GPU, carrying any XID delivered since the last one."""
        xids = self._xids()
        rows = []
        for gpu in self._gpus:
            reasons = self._query(gpu, "reasons", self._reasons)
            power = self._query(gpu, "power", self._nvml.nvmlDeviceGetPowerUsage)
            util = self._query(gpu, "util", self._nvml.nvmlDeviceGetUtilizationRates)
            rows.append(
                {
                    "gpu": gpu.index,
                    "sm_clock_mhz": self._query(
                        gpu, "sm_clock", self._nvml.nvmlDeviceGetClockInfo, self._nvml.NVML_CLOCK_SM
                    ),
                    "clocks_event_reasons": None if reasons is None else f"0x{reasons:x}",
                    "power_w": None if power is None else power / 1000,
                    "temp_c": self._query(
                        gpu, "temp", self._nvml.nvmlDeviceGetTemperature, self._nvml.NVML_TEMPERATURE_GPU
                    ),
                    "util": None if util is None else util.gpu / 100,
                    **self._duty_cycle(gpu),
                    "xid": ";".join(xids[gpu.index]) if gpu.index in xids else None,
                }
            )
        return rows

    def close(self) -> None:
        """Free the event set before NVML goes away."""
        if self._events is not None:
            try:
                self._nvml.nvmlEventSetFree(self._events)
            except self._nvml.NVMLError:
                pass
            self._events = None
        super().close()
