from __future__ import annotations

import ctypes
from typing import Any, NamedTuple

from ..collector import CollectorMetric, MetricKind, Rate
from .nvml import Gpu, NvmlCollector


class NvlinkBytes(NamedTuple):
    """Cumulative NVLink bytes summed over every link, each None when unreadable."""

    tx: int | None
    rx: int | None


class _Links:
    """One GPU's NVML field-value buffer, and the per-direction peak of its active links."""

    def __init__(self, values: Any) -> None:
        #: Used by nothing else; see NvlinkCollector.values_capacity.
        self.values = values
        #: Bytes/s per direction over the active links, for nvlink_utilization.
        self.peak: float | None = None


class NvlinkCollector(NvmlCollector):
    """NVLink bytes, rates and utilization per GPU.

    Throughput counters take ~1.1 ms of CPU per counter, so 2.2 ms per GPU for
    both TX and RX on an A100 with 12 links.

    The counters are cumulative, so a longer period loses no bytes, only the
    resolution of the rates.
    """

    name = "nvlink"
    metrics = (
        CollectorMetric("gpu", MetricKind.DEVICE, "", "NVML index of the GPU"),
        CollectorMetric("nvlink_tx_bytes", MetricKind.COUNTER, "B", "Bytes sent over every NVLink since driver load"),
        CollectorMetric("nvlink_rx_bytes", MetricKind.COUNTER, "B", "Bytes received over every NVLink since driver load"),
        CollectorMetric("nvlink_tx_bytes_per_s", MetricKind.SAMPLE, "B/s", "Send rate over the interval, summed over the links"),
        CollectorMetric("nvlink_rx_bytes_per_s", MetricKind.SAMPLE, "B/s", "Receive rate over the interval, summed over the links"),
        CollectorMetric(
            "nvlink_utilization", MetricKind.SAMPLE, "fraction",
            "The busier direction's share of the active links' peak rate in one direction",
        ),
    )
    #: Entries in each field-value buffer; only a few are used. Under training, an NVML record
    #: was seen written one block past pynvml's own result array, onto a live object. The
    #: padding keeps a buffer (16 x 40 B) above pymalloc's 512-byte limit, so it never shares
    #: a pool with Python objects and such a write stays harmless.
    values_capacity = 16
    #: scopeId summing every NVLink; 0 would read link 0 alone.
    all_links = 0xFFFFFFFF
    #: NVML reports NVLink throughput in KiB.
    unit_bytes = 1024

    def __init__(self) -> None:
        super().__init__()
        self._get_field_values: Any = None
        self._links: dict[int, _Links] = {}
        self._rates = Rate()

    def _init(self) -> bool:
        try:
            # The raw entry point, so NVML writes into our buffers rather than pynvml's.
            self._get_field_values = self._nvml._nvmlGetFunctionPointer("nvmlDeviceGetFieldValues")
        except (self._nvml.NVMLError, AttributeError):
            return False
        return True

    def _open(self, gpu: Gpu) -> bool:
        """Keep a GPU whose counters can be read, with the peak of its active links."""
        pynvml = self._nvml
        links = _Links((pynvml.c_nvmlFieldValue_t * self.values_capacity)())
        self._links[gpu.index] = links
        if self._bytes(gpu).tx is None:
            del self._links[gpu.index]
            return False
        active = []
        for link in range(pynvml.NVML_NVLINK_MAX_LINKS):
            try:
                if pynvml.nvmlDeviceGetNvLinkState(gpu.handle, link) == pynvml.NVML_FEATURE_ENABLED:
                    active.append(link)
            except pynvml.NVMLError:
                continue
        links.peak = self._peak(gpu, active)
        return True

    def _peak(self, gpu: Gpu, active: list[int]) -> float | None:
        """Bytes/s per direction over the active links, None when NVML gives no speed.

        The common speed where NVML has it; otherwise each link's own, which is all
        H100 (driver 580) answers: it refuses the common speed as not supported.
        """
        if not active:
            return None
        pynvml = self._nvml
        common = self._read_fields(gpu, [(pynvml.NVML_FI_DEV_NVLINK_SPEED_MBPS_COMMON, 0)])
        if common and common[0]:
            return len(active) * common[0] * 1e6
        field = getattr(pynvml, "NVML_FI_DEV_NVLINK_GET_SPEED", None)  # absent from older bindings
        if field is None:
            return None
        total = 0.0
        # One link per read: 18 links on H100 would overflow the 16-entry buffer.
        for link in active:
            speed = self._read_fields(gpu, [(field, link)])
            if not speed or not speed[0]:
                return None
            total += speed[0]
        return total * 1e6

    def _read_fields(self, gpu: Gpu, fields: list[tuple[int, int]]) -> list[float] | None:
        """NVML field values for (field id, scope) pairs, or None if any is unreadable."""
        values = self._links[gpu.index].values
        # Cleared on every call: on H100 (driver 580) an entry that once answered NOT_SUPPORTED,
        # as the link speed does, kept answering it for every field read into it afterwards.
        ctypes.memset(values, 0, ctypes.sizeof(values))
        for i, (field_id, scope) in enumerate(fields):
            values[i].fieldId = field_id
            values[i].scopeId = scope
        try:
            ret = self._get_field_values(gpu.handle, ctypes.c_int32(len(fields)), ctypes.byref(values))
        except (ctypes.ArgumentError, OSError):
            return None
        if ret != 0:  # NVML_SUCCESS
            return None
        union = self._nvml.c_nvmlValue_t._fields_
        out = []
        for i in range(len(fields)):
            if values[i].nvmlReturn != 0:
                return None
            # The union's members are in NVML_VALUE_TYPE_* order.
            out.append(getattr(values[i].value, union[values[i].valueType][0]))
        return out

    def _bytes(self, gpu: Gpu) -> NvlinkBytes:
        """Cumulative TX and RX bytes over every link."""
        pynvml = self._nvml
        counters = self._read_fields(
            gpu,
            [
                (pynvml.NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX, self.all_links),
                (pynvml.NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX, self.all_links),
            ],
        )
        if not counters:
            return NvlinkBytes(None, None)
        return NvlinkBytes(int(counters[0]) * self.unit_bytes, int(counters[1]) * self.unit_bytes)

    def sample(self, now_s: float) -> list[dict[str, Any]]:
        """One row per GPU with NVLink."""
        rows = []
        for gpu in self._gpus:
            nvlink = self._bytes(gpu)
            tx_rate = self._rates.update((gpu.index, "tx"), now_s, nvlink.tx)
            rx_rate = self._rates.update((gpu.index, "rx"), now_s, nvlink.rx)
            rows.append(
                {
                    "gpu": gpu.index,
                    "nvlink_tx_bytes": nvlink.tx,
                    "nvlink_rx_bytes": nvlink.rx,
                    "nvlink_tx_bytes_per_s": tx_rate,
                    "nvlink_rx_bytes_per_s": rx_rate,
                    "nvlink_utilization": _utilization(tx_rate, rx_rate, self._links[gpu.index].peak),
                }
            )
        return rows

    def close(self) -> None:
        super().close()
        self._links = {}


def _utilization(tx_rate: float | None, rx_rate: float | None, peak: float | None) -> float | None:
    """The busier direction's share of the per-direction peak."""
    if tx_rate is None or rx_rate is None or not peak:
        return None
    return max(tx_rate, rx_rate) / peak
