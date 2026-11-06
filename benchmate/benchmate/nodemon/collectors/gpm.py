from __future__ import annotations

import logging
from types import ModuleType
from typing import Any, ClassVar, NamedTuple

import pynvml

from ..collector import CollectorMetric, MetricKind
from .nvml import Gpu, NvmlCollector, nvml_error_text

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
        """`nvml`: already initialized."""
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
        except self._nvml.NVMLError as err:
            log.debug(f"NVML would not allocate a GPM sample buffer ({nvml_error_text(err.value)}).")
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
            except self._nvml.NVMLError as err:
                log.debug(f"NVML would not free a GPM sample buffer ({nvml_error_text(err.value)}).")
        self.buffers = []
        self.samples = Samples(None, None)


class GpmCollector(NvmlCollector):
    """SM and tensor-pipe activity from NVML's GPM counters, the ones DCGM's ``DCGM_FI_PROF_*`` fields come from.

    Hopper and newer only; elsewhere the collector probes false and activity
    comes from ``gpu.util_mean``. A metric is the difference between two
    samples at least 100 ms apart, so the first tick writes empty cells.
    Pipe columns are fractions. The PCIe columns are MiB/s and stay empty
    when the bindings do not name the counter.
    """

    name = "gpm"

    #: Each metric column -> the NVML constant naming the GPM metric behind it.
    metric_names: ClassVar[dict[str, str]] = {
        "sm_util": "NVML_GPM_METRIC_SM_UTIL",
        "sm_occupancy": "NVML_GPM_METRIC_SM_OCCUPANCY",
        "tensor_util": "NVML_GPM_METRIC_ANY_TENSOR_UTIL",
        "dram_bw_util": "NVML_GPM_METRIC_DRAM_BW_UTIL",
        "fp64_util": "NVML_GPM_METRIC_FP64_UTIL",
        "fp32_util": "NVML_GPM_METRIC_FP32_UTIL",
        "fp16_util": "NVML_GPM_METRIC_FP16_UTIL",
        "hmma_tensor_util": "NVML_GPM_METRIC_HMMA_TENSOR_UTIL",
        "imma_tensor_util": "NVML_GPM_METRIC_IMMA_TENSOR_UTIL",
        "dmma_tensor_util": "NVML_GPM_METRIC_DMMA_TENSOR_UTIL",
        "pcie_tx_mib_s": "NVML_GPM_METRIC_PCIE_TX_PER_SEC",
        "pcie_rx_mib_s": "NVML_GPM_METRIC_PCIE_RX_PER_SEC",
    }
    #: NVML reports these in MiB/s. Every other column is a percentage.
    rates: ClassVar[frozenset[str]] = frozenset({
        "pcie_tx_mib_s", "pcie_rx_mib_s",
    })
    #: The row: the GPU, then one cell per metric above, in that order.
    metrics = (
        CollectorMetric("gpu", MetricKind.DEVICE, "", "NVML index of the GPU"),
        CollectorMetric("sm_util", MetricKind.SAMPLE, "fraction", "Share of SMs in use over the interval"),
        CollectorMetric("sm_occupancy", MetricKind.SAMPLE, "fraction", "Resident warps as a share of the most the SMs can hold"),
        CollectorMetric("tensor_util", MetricKind.SAMPLE, "fraction", "Share of time any tensor pipe was active"),
        CollectorMetric("dram_bw_util", MetricKind.SAMPLE, "fraction", "Device memory bandwidth in use as a share of its peak"),
        CollectorMetric("fp64_util", MetricKind.SAMPLE, "fraction", "Share of time the SMs did non-tensor FP64 math"),
        CollectorMetric("fp32_util", MetricKind.SAMPLE, "fraction", "Share of time the SMs did non-tensor FP32 math"),
        CollectorMetric("fp16_util", MetricKind.SAMPLE, "fraction", "Share of time the SMs did non-tensor FP16 math"),
        CollectorMetric("hmma_tensor_util", MetricKind.SAMPLE, "fraction", "Share of time the SMs did HMMA tensor operations (FP16/BF16)"),
        CollectorMetric("imma_tensor_util", MetricKind.SAMPLE, "fraction", "Share of time the SMs did IMMA tensor operations (INT8; FP8 sometimes counts here)"),
        CollectorMetric("dmma_tensor_util", MetricKind.SAMPLE, "fraction", "Share of time the SMs did DMMA tensor operations"),
        CollectorMetric("pcie_tx_mib_s", MetricKind.SAMPLE, "MiB/s", "PCIe traffic from this GPU"),
        CollectorMetric("pcie_rx_mib_s", MetricKind.SAMPLE, "MiB/s", "PCIe traffic to this GPU"),
    )

    def __init__(self, nvml: ModuleType = pynvml) -> None:
        """`nvml`: already initialized."""
        super().__init__(nvml)
        self._metric_id: dict[str, int] = {}
        #: One pair of sample buffers per GPU index.
        self._pairs: dict[int, SamplePair] = {}

    def _init(self) -> bool:
        """Resolve the metric identifiers; False when GPM cannot be queried or no counter is named.

        A counter the installed bindings do not declare is skipped, and its cell stays empty.
        The collector stays off only when none of them are declared, or the GPM calls themselves are missing.
        """
        needed = (
            "nvmlGpmQueryDeviceSupport",
            "nvmlGpmSampleAlloc",
            "nvmlGpmSampleGet",
            "nvmlGpmSampleFree",
            "nvmlGpmMetricsGet",
            "c_nvmlGpmMetricsGet_t",
            "NVML_GPM_METRICS_GET_VERSION",
        )
        missing = [name for name in needed if not hasattr(self._nvml, name)]
        if missing:
            log.debug(f"No GPM rows: the bindings do not define {', '.join(missing)}.")
            return False
        resolved: dict[str, int] = {}
        for column, constant in self.metric_names.items():
            metric_id = getattr(self._nvml, constant, None)
            if metric_id is None:
                continue
            resolved[column] = metric_id
        if not resolved:
            log.debug("No GPM rows: the bindings define none of the counters.")
            return False
        self._metric_id = resolved
        return True

    def _open(self, gpu: Gpu) -> bool:
        """Keep a GPU only when it supports GPM, and give it the buffers a metric needs."""
        try:
            support = self._nvml.nvmlGpmQueryDeviceSupport(gpu.handle)
        except self._nvml.NVMLError as err:
            log.debug(f"GPU {gpu.index}: NVML refused the GPM support query ({nvml_error_text(err.value)}); no GPM rows for it.")
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
        except self._nvml.NVMLError as err:
            self._log_nvml_error("reading the GPM metrics", err.value)
            return self._empty()
        readings = self._empty()
        for slot, column in enumerate(self._metric_id):
            readings[column] = self._reading(column, request.metrics[slot])
        return readings

    def _reading(self, column: str, metric: Any) -> float | None:
        """One metric. A failed one is empty. Percentages become fractions; MiB/s stays MiB/s."""
        if metric.nvmlReturn != self._nvml.NVML_SUCCESS:
            self._log_nvml_error(f"GPM metric {column}", metric.nvmlReturn)
            return None
        if column in self.rates:
            return metric.value
        return metric.value / 100

    def _empty(self) -> dict[str, Any]:
        """A row's metric cells when there is nothing to report: unavailable, not zero."""
        return {
            column: None 
            for column 
            in self.metric_names
        }

    def close(self) -> None:
        """Free every sample buffer before NVML goes away."""
        for pair in self._pairs.values():
            pair.free()
        self._pairs = {}
        super().close()
