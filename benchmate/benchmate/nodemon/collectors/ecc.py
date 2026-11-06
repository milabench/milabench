from __future__ import annotations

import logging
from types import ModuleType
from typing import Any, NamedTuple

import pynvml

from ..collector import CollectorMetric, MetricKind
from .nvml import Gpu, NvmlCollector

log = logging.getLogger(__name__)


class RemappedRows(NamedTuple):
    """What nvmlDeviceGetRemappedRows returns; every field None when the query fails."""

    corrected: int | None = None
    uncorrected: int | None = None
    pending: int | None = None
    failure: int | None = None


def difference(end: dict[str, Any], start: dict[str, Any]) -> dict[str, Any]:
    """Field-wise `end - start` of two error inventories; None where either side is unknown.

    An inventory is a GPU's lifetime error state, read for the status file rather than written as rows.
    """
    return {
        key: None if end.get(key) is None or start.get(key) is None else end[key] - start[key]
        for key in ("ecc_aggregate_uncorrected", "remapped_corrected", "remapped_uncorrected")
    }


class EccCollector(NvmlCollector):
    """Memory errors and row remapping, in their own collector.

    Separate from ``gpu`` on purpose: these queries are the ones that go slow
    on a failing GPU, which is when their values matter. Here a hang costs
    this collector alone, not the one-second series the figures are built on.
    """

    name = "ecc"
    metrics = (
        CollectorMetric("gpu", MetricKind.DEVICE, "", "NVML index of the GPU"),
        CollectorMetric(
            "ecc_uncorrected", MetricKind.COUNTER, "errors",
            "Uncorrected memory errors since driver load (the volatile count)",
        ),
        CollectorMetric("remap_pending", MetricKind.SAMPLE, "", "Nonzero when a row remap waits for a GPU reset to take effect"),
        CollectorMetric("remap_failure", MetricKind.SAMPLE, "", "Nonzero when a row remap failed: the GPU has no spare rows left"),
    )

    def __init__(self, nvml: ModuleType = pynvml) -> None:
        """`nvml`: already initialized."""
        super().__init__(nvml)
        #: Per GPU, the inventory when the sampler started and the one from the latest sample.
        #: Both have every GPU from probe() on, so the status file can read them while a sample runs.
        #: The end snapshot is the last sample's, not a read at close(): shutdown runs on the main
        #: thread with no timeout, and a GPU that hangs there would keep the status file from being written.
        self._start: dict[int, dict[str, Any]] = {}
        self._latest: dict[int, dict[str, Any]] = {}
        #: Per GPU, the last (pending, failure) remap flags, so a change is logged once.
        self._remap_flags: dict[int, tuple[int | None, int | None]] = {}

    def probe(self) -> bool:
        if not super().probe():
            return False
        remapped = {gpu.index: self._remapped(gpu) for gpu in self._gpus}
        self._start = {gpu.index: self._inventory(gpu, remapped[gpu.index]) for gpu in self._gpus}
        self._latest = dict(self._start)
        for index, rows in remapped.items():
            self._log_remap(index, rows)
        return True

    def _log_remap(self, index: int, remapped: RemappedRows) -> None:
        """Log a GPU's remap flags when they change: a failure is an error, a pending remap a warning.

        The first read is logged only when it is already a failure or pending.
        """
        flags = (remapped.pending, remapped.failure)
        previous = self._remap_flags.get(index)
        if flags == previous:
            return
        self._remap_flags[index] = flags
        if remapped.failure:
            log.error(f"GPU {index}: a row remap failed, so the GPU has no spare rows left.")
        elif remapped.pending:
            log.warning(f"GPU {index}: a row remap is pending; it takes effect at the next GPU reset.")
        elif previous is not None:
            log.debug(f"GPU {index}: row remap flags {previous} -> {flags} (pending, failure).")

    def _volatile_uncorrected(self, gpu: Gpu) -> Any:
        return self._query(
            gpu, "ecc", self._nvml.nvmlDeviceGetTotalEccErrors,
            self._nvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED, self._nvml.NVML_VOLATILE_ECC,
        )

    def _aggregate_uncorrected(self, gpu: Gpu) -> Any:
        return self._query(
            gpu, "ecc_aggregate", self._nvml.nvmlDeviceGetTotalEccErrors,
            self._nvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED, self._nvml.NVML_AGGREGATE_ECC,
        )

    def _remapped(self, gpu: Gpu) -> RemappedRows:
        result = self._query(gpu, "remapped_rows", self._nvml.nvmlDeviceGetRemappedRows)
        return RemappedRows(*result) if result else RemappedRows()

    def _inventory(self, gpu: Gpu, remapped: RemappedRows) -> dict[str, Any]:
        return {
            "ecc_aggregate_uncorrected": self._aggregate_uncorrected(gpu),
            "remapped_corrected": remapped.corrected,
            "remapped_uncorrected": remapped.uncorrected,
        }

    def sample(self, now_s: float) -> list[dict[str, Any]]:
        """One row per GPU: what changed during the run. The lifetime counts go to the status file."""
        rows = []
        for gpu in self._gpus:
            remapped = self._remapped(gpu)
            inventory = self._inventory(gpu, remapped)
            previous = self._latest.get(gpu.index, {})
            if inventory != previous:
                # A count going up is a new error; a value appearing or vanishing is a query that came or went.
                grew = any((delta or 0) > 0 for delta in difference(inventory, previous).values())
                log.log(
                    logging.WARNING if grew else logging.DEBUG,
                    f"GPU {gpu.index}: error inventory {previous} -> {inventory}.",
                )
            self._latest[gpu.index] = inventory
            self._log_remap(gpu.index, remapped)
            rows.append(
                {
                    "gpu": gpu.index,
                    "ecc_uncorrected": self._volatile_uncorrected(gpu),
                    "remap_pending": remapped.pending,
                    "remap_failure": remapped.failure,
                }
            )
        return rows

    def describe(self) -> dict[str, Any]:
        """The GPU map, plus each GPU's error inventory at start, at the latest read, and their difference.

        Reads no GPU: a sampler stuck in NVML must not keep the status file from being written.
        """
        described = super().describe()
        if self._start:
            described["snapshot_start"] = {str(index): values for index, values in sorted(self._start.items())}
            described["snapshot"] = {str(index): values for index, values in sorted(self._latest.items())}
            described["snapshot_delta"] = {
                str(index): difference(values, self._start[index])
                for index, values in sorted(self._latest.items())
                if index in self._start
            }
        return described
