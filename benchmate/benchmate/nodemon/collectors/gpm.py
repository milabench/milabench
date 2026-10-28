from __future__ import annotations

import logging
from types import ModuleType
from typing import Any, NamedTuple

from .nvml import Gpu, NvmlCollector

log = logging.getLogger(__name__)


class Samples(NamedTuple):
    """The two samples a metric is measured over."""

    newest: Any # Filled first
    previous: Any # Filled second


class SamplePair:
    """One GPU's two GPM sample buffers, written in turn so each sample survives
    as the next one's baseline.

    The buffers allocate NVML's memory on the device rather than on the host so
    they are created for the entire lifecycle.
    """

    def __init__(self, nvml: ModuleType) -> None:
        self._nvml = nvml
        #: Both buffers, NVML's to hand back; which one is spare follows from the samples.
        self.buffers: list[Any] = []
        #: What the buffers hold so far.
        self.samples = Samples(None, None)

    def allocate(self) -> bool:
        """Take the two buffers from NVML; False, with neither left allocated, when it will not give both."""
        try:
            self.buffers.append(self._nvml.nvmlGpmSampleAlloc())
            self.buffers.append(self._nvml.nvmlGpmSampleAlloc())
        except self._nvml.NVMLError:
            self.free()
            return False
        return True

    def take(self, sample_into: Any) -> Samples:
        """Write the next sample with `sample_into` into whichever buffer is not the newest, and return the pair.

        A sample that fails leaves the previous ones in place, so the next
        successful pair just spans a longer interval.
        """
        spare = next(buffer for buffer in self.buffers if buffer is not self.samples.newest)
        if sample_into(spare) is None:
            return Samples(None, None)
        self.samples = Samples(spare, self.samples.newest)
        return self.samples

    def free(self) -> None:
        """Give the buffers back to NVML."""
        for buffer in self.buffers:
            try:
                self._nvml.nvmlGpmSampleFree(buffer)
            except self._nvml.NVMLError:
                pass
        self.buffers = []
        self.samples = Samples(None, None)


class GpmCollector(NvmlCollector):
    """SM and tensor-pipe activity from NVML's GPM counters, the ones DCGM's ``DCGM_FI_PROF_*`` fields come from.

    Hopper and newer only; elsewhere the collector probes false and activity
    comes from ``gpu.util_mean``. A metric is the difference between two
    samples at least 100 ms apart, so the first tick writes empty cells.
    """

    name = "gpm"

    #: Each metric column -> the NVML constant naming the GPM metric behind it.
    metric_names = {
        "sm_util": "NVML_GPM_METRIC_SM_UTIL",
        "sm_occupancy": "NVML_GPM_METRIC_SM_OCCUPANCY",
        "tensor_util": "NVML_GPM_METRIC_ANY_TENSOR_UTIL",
        "dram_bw_util": "NVML_GPM_METRIC_DRAM_BW_UTIL",
    }
    #: The row: the GPU, then one cell per metric above, in that order.
    columns = ("gpu", *metric_names)

    def __init__(self) -> None:
        super().__init__()
        self._metric_id: dict[str, int] = {}
        #: One pair of sample buffers per GPU index.
        self._pairs: dict[int, SamplePair] = {}

    def _init(self) -> bool:
        """Resolve the metric identifiers; False when the bindings predate GPM (nvidia-ml-py 11.525)."""
        needed = (
            "nvmlGpmQueryDeviceSupport",
            "nvmlGpmSampleAlloc",
            "nvmlGpmSampleGet",
            "nvmlGpmSampleFree",
            "nvmlGpmMetricsGet",
            "c_nvmlGpmMetricsGet_t",
            "NVML_GPM_METRICS_GET_VERSION",
        )
        try:
            self._metric_id = {
                column: getattr(self._nvml, constant)
                for column, constant in self.metric_names.items()
            }
        except AttributeError:
            return False
        return all(hasattr(self._nvml, name) for name in needed)

    def _open(self, gpu: Gpu) -> bool:
        """Keep a GPU only when it supports GPM, and give it the buffers a metric needs."""
        try:
            support = self._nvml.nvmlGpmQueryDeviceSupport(gpu.handle)
        except self._nvml.NVMLError as error:
            log.debug(f"GPU {gpu.index}: NVML refused the GPM support query ({error}); no GPM rows for it.")
            return False
        if not support.isSupportedDevice:
            log.debug(f"GPU {gpu.index} has no GPM counters, so it is older than Hopper; no GPM rows for it.")
            return False
        pair = SamplePair(self._nvml)
        if not pair.allocate():
            log.debug(f"GPU {gpu.index}: NVML would not give two GPM sample buffers; no GPM rows for it.")
            return False
        self._pairs[gpu.index] = pair
        return True

    def sample(self, now_s: float) -> list[dict[str, Any]]:
        """One row per GPU, covering the interval since the last tick."""
        return [
            {"gpu": gpu.index, **self._metrics(gpu)}
            for gpu in self._gpus
        ]

    def _metrics(self, gpu: Gpu) -> dict[str, Any]:
        """Take this tick's sample and read the counters since the previous one."""
        pair = self._pairs.get(gpu.index)
        if pair is None:
            return self._empty()
        samples = pair.take(
            lambda buffer: self._query(gpu, "gpm_sample", self._nvml.nvmlGpmSampleGet, buffer)
        )
        if samples.previous is None:
            return self._empty()
        return self._between(samples.previous, samples.newest)

    def _between(self, first: Any, second: Any) -> dict[str, Any]:
        """Every metric over one pair of samples, in a single NVML call."""
        request = self._nvml.c_nvmlGpmMetricsGet_t()
        request.version = self._nvml.NVML_GPM_METRICS_GET_VERSION
        request.numMetrics = len(self._metric_id)
        request.sample1 = first
        request.sample2 = second
        for slot, metric_id in enumerate(self._metric_id.values()):
            request.metrics[slot].metricId = metric_id
        try:
            self._nvml.nvmlGpmMetricsGet(request)
        except self._nvml.NVMLError:
            return self._empty()
        return {
            column: self._fraction(request.metrics[slot])
            for slot, column in enumerate(self._metric_id)
        }

    def _fraction(self, metric: Any) -> float | None:
        """One metric as a fraction: NVML reports these as percentages, and a failed one is empty."""
        if metric.nvmlReturn != self._nvml.NVML_SUCCESS:
            return None
        return metric.value / 100

    def _empty(self) -> dict[str, Any]:
        """A row's metric cells when there is nothing to report: unavailable, not zero."""
        return {column: None for column in self._metric_id}

    def close(self) -> None:
        """Free every sample buffer before NVML goes away."""
        for pair in self._pairs.values():
            pair.free()
        self._pairs = {}
        super().close()
