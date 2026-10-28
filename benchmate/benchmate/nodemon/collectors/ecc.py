from __future__ import annotations

from typing import Any

from .nvml import Gpu, NvmlCollector


class EccCollector(NvmlCollector):
    """Memory errors and row remapping, in their own collector.

    Separate from ``gpu`` on purpose: these queries are the ones that go slow
    on a failing GPU, which is when their values matter. Here a hang costs
    this collector alone, not the one-second series the figures are built on.
    """

    name = "ecc"
    columns = (
        "gpu",
        "ecc_uncorrected",
        "remap_pending",
        "remap_failure",
    )

    def __init__(self) -> None:
        super().__init__()
        #: Lifetime values per GPU, read for the status file rather than written on every row.
        self._lifetime: dict[int, dict[str, Any]] = {}

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

    def sample(self, now_s: float) -> list[dict[str, Any]]:
        """One row per GPU: what changed during the run. The lifetime counts go to the status file."""
        rows = []
        for gpu in self._gpus:
            remapped = self._query(gpu, "remapped_rows", self._nvml.nvmlDeviceGetRemappedRows)
            corrected, uncorrected, pending, failure = remapped or (None, None, None, None)
            self._lifetime[gpu.index] = {
                "ecc_aggregate_uncorrected": self._aggregate_uncorrected(gpu),
                "remapped_corrected": corrected,
                "remapped_uncorrected": uncorrected,
            }
            rows.append(
                {
                    "gpu": gpu.index,
                    "ecc_uncorrected": self._volatile_uncorrected(gpu),
                    "remap_pending": pending,
                    "remap_failure": failure,
                }
            )
        return rows

    def describe(self) -> dict[str, Any]:
        """The GPU map, plus each GPU's lifetime error state: an inventory, not a series."""
        described = super().describe()
        if self._lifetime:
            described["lifetime"] = {str(index): values for index, values in sorted(self._lifetime.items())}
        return described
