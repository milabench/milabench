from __future__ import annotations

import logging
from enum import Enum, IntFlag
from types import ModuleType
from typing import Any

import pynvml

from ..collector import CollectorMetric, MetricKind
from .nvml import Gpu, NvmlCollector

log = logging.getLogger(__name__)

# TODO: Upstream SM clock and throttle reasons to voir gpudata.


class ClockReason(IntFlag):
    """Why a GPU's SM clock is below its maximum: NVML's nvmlClocksEventReason* bits.

    The `clocks_event_reasons` column holds them as hex; readers decode it with parse().
    """

    GPU_IDLE = pynvml.nvmlClocksEventReasonGpuIdle
    APPLICATIONS_CLOCKS_SETTING = pynvml.nvmlClocksEventReasonApplicationsClocksSetting
    SW_POWER_CAP = pynvml.nvmlClocksEventReasonSwPowerCap
    HW_SLOWDOWN = pynvml.nvmlClocksEventReasonHwSlowdown
    SYNC_BOOST = pynvml.nvmlClocksEventReasonSyncBoost
    SW_THERMAL_SLOWDOWN = pynvml.nvmlClocksEventReasonSwThermalSlowdown
    HW_THERMAL_SLOWDOWN = pynvml.nvmlClocksEventReasonHwThermalSlowdown
    HW_POWER_BRAKE_SLOWDOWN = pynvml.nvmlClocksEventReasonHwPowerBrakeSlowdown
    DISPLAY_CLOCK_SETTING = pynvml.nvmlClocksEventReasonDisplayClockSetting

    @classmethod
    def mask(cls, text: str) -> int:
        """The bits in a gpu.tsv cell (``f"0x{reasons:x}"``), the one parser of that column.

        Text without the ``0x`` prefix, an empty cell included, raises ValueError rather than
        reading a decimal mask as hex.
        """
        raw = text.strip()
        if not raw.lower().startswith("0x"):
            raise ValueError(f"clocks_event_reasons must be hex text starting with 0x, got {text!r}")
        return int(raw, 16)  # raises ValueError on a bad digit

    @classmethod
    def split(cls, mask: int) -> list[ClockReason]:
        """The declared reasons set in `mask`, in declaration order.

        ``0x84`` is ``[SW_POWER_CAP, HW_POWER_BRAKE_SLOWDOWN]``. Bits this enum does not declare
        are left out; describe() shows them.
        """
        return [reason for reason in cls if reason & mask]

    @classmethod
    def parse(cls, text: str) -> list[ClockReason]:
        """The declared reasons set in a gpu.tsv cell; see mask() and split()."""
        return cls.split(cls.mask(text))

    @classmethod
    def describe(cls, mask: int) -> str:
        """`mask` for display: ``SW_POWER_CAP|0x200``, the declared names then any other bits
        (a newer driver's) as hex, so none is dropped. Empty for 0.
        """
        names = [reason.name for reason in cls.split(mask)]
        # Plain ints: ~ on an IntFlag complements within the declared bits only, from Python 3.11.
        undeclared = mask & ~sum(int(reason) for reason in cls)
        if undeclared:
            names.append(f"0x{undeclared:x}")
        return "|".join(names)


class PowerQuery(str, Enum):
    """The NVML query behind each power reading, all in milliwatts.

    Each member is named after the metric it reads, so ``query.name`` is the row or status key.
    nvmlDeviceGetEnforcedPowerLimit is missing from older bindings: that reading is then None.
    """

    # A str mixin rather than StrEnum (Python 3.11+), which benchmate can't require yet.
    # Lowercase to match the metric names exactly.
    power_w = "nvmlDeviceGetPowerUsage"
    power_enforced_w = "nvmlDeviceGetEnforcedPowerLimit"
    power_limit_w = "nvmlDeviceGetPowerManagementLimit"
    power_default_w = "nvmlDeviceGetPowerManagementDefaultLimit"


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
        CollectorMetric(
            "power_enforced_w", MetricKind.SAMPLE, "W",
            "Power limit in effect: the configured one, or lower under an outside cap (e.g. 200 W)",
        ),
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

    def __init__(self, nvml: ModuleType = pynvml) -> None:
        """`nvml`: already initialized."""
        super().__init__(nvml)
        self._reasons: Any = None
        self._events: Any = None
        #: (GPU index, XID) pairs already logged as a warning; repeats go to debug.
        self._xids_logged: set[tuple[int, int]] = set()
        #: The last utilization sample already read, per GPU, so each call asks for newer ones only.
        self._seen: dict[int, int] = {}
        #: Per GPU, the power limits at the first sample, for the status file rather than every row:
        #: they change only when an admin sets them. The one in effect is the power_enforced_w series.
        #: Read on the sampling thread, not in probe(): the main thread has no timeout on a hung GPU.
        self._limits: dict[int, dict[str, Any]] = {}

    def _read_limits(self) -> dict[int, dict[str, Any]]:
        limits = (PowerQuery.power_default_w, PowerQuery.power_limit_w)
        return {
            gpu.index: {
                query.name: self._get_power_attr(gpu, query)
                for query in limits
            }
            for gpu in self._gpus
        }

    def _get_power_attr(self, gpu: Gpu, query: PowerQuery) -> float | None:
        """One power reading in W; None when the query fails or this NVML lacks it."""
        milliwatts = self._query(gpu, query.name, getattr(self._nvml, query.value, None))
        return None if milliwatts is None else milliwatts / 1000

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
        except self._nvml.NVMLError as err:
            self._log_nvml_error("creating the XID event set (no XID column)", err.value)
            return None

    def _open(self, gpu: Gpu) -> bool:
        """Subscribe this GPU to XID events; one that refuses is still sampled, it just reports none."""
        if self._events is None:
            return True
        try:
            self._nvml.nvmlDeviceRegisterEvents(gpu.handle, self._nvml.nvmlEventTypeXidCriticalError, self._events)
        except self._nvml.NVMLError as err:
            self._log_nvml_error(f"GPU {gpu.index} registering for XID events (none reported for it)", err.value)
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
            except self._nvml.NVMLError as err:
                # NVML_ERROR_TIMEOUT: the queue is empty, which is the normal case.
                if err.value != self._nvml.NVML_ERROR_TIMEOUT:
                    self._log_nvml_error("waiting for XID events", err.value)
                return seen
            if not event.eventType & self._nvml.nvmlEventTypeXidCriticalError:
                continue
            try:
                index = self._nvml.nvmlDeviceGetIndex(event.device)
            except self._nvml.NVMLError as err:
                self._log_nvml_error(f"indexing the GPU of XID {event.eventData} (dropped)", err.value)
                continue
            if (index, event.eventData) in self._xids_logged:
                log.debug(f"GPU {index}: XID {event.eventData} again.")
            else:
                self._xids_logged.add((index, event.eventData))
                log.warning(f"GPU {index}: XID {event.eventData}; repeats of it on this GPU are logged at debug.")
            seen.setdefault(index, []).append(str(event.eventData))
        log.debug(f"{self.max_events} events drained this tick; the rest wait for the next one.")
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
        if not self._limits:
            self._limits = self._read_limits()
        xids = self._xids()
        return [self._row(gpu, xids.get(gpu.index, [])) for gpu in self._gpus]

    def _row(self, gpu: Gpu, xids: list[str]) -> dict[str, Any]:
        """One GPU's row; `xids` are the ones delivered for it since the last row."""
        sm_clock_mhz = self._query(gpu, "sm_clock", self._nvml.nvmlDeviceGetClockInfo, self._nvml.NVML_CLOCK_SM)
        reasons = self._query(gpu, "reasons", self._reasons)
        temp_c = self._query(gpu, "temp", self._nvml.nvmlDeviceGetTemperature, self._nvml.NVML_TEMPERATURE_GPU)
        util = self._query(gpu, "util", self._nvml.nvmlDeviceGetUtilizationRates)
        return {
            "gpu": gpu.index,
            "sm_clock_mhz": sm_clock_mhz,
            "clocks_event_reasons": None if reasons is None else f"0x{reasons:x}",
            PowerQuery.power_w.name: self._get_power_attr(gpu, PowerQuery.power_w),
            PowerQuery.power_enforced_w.name: self._get_power_attr(gpu, PowerQuery.power_enforced_w),
            "temp_c": temp_c,
            "util": None if util is None else util.gpu / 100,
            **self._duty_cycle(gpu),
            "xid": ";".join(xids) or None,
        }

    def close(self) -> None:
        """Free the event set before NVML goes away."""
        if self._events is not None:
            try:
                self._nvml.nvmlEventSetFree(self._events)
            except self._nvml.NVMLError as err:
                self._log_nvml_error("freeing the XID event set", err.value)
            self._events = None
        super().close()

    def describe(self) -> dict[str, Any]:
        """The GPU map, plus each GPU's power limits at the first sample."""
        described = super().describe()
        if self._limits:
            described["power_limits"] = {
                str(index): values
                for index, values
                in sorted(self._limits.items())
            }
        return described
