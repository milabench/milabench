from __future__ import annotations

from pathlib import Path
from typing import Any

from ..collector import (
    CollectorMetric,
    MetricKind,
    NodeCollector,
    PolledFile,
    read_attr,
)


class _Controller:
    """One memory controller's open error counters."""

    def __init__(self, path: Path) -> None:
        self.name = path.name
        self.model = read_attr(path / "mc_name") or ""
        self.files = {name: PolledFile(path / name) for name in ("ce_count", "ue_count")}


class EdacCollector(NodeCollector):
    """Host memory errors per controller, from EDAC: correctable counts rise before a DIMM fails."""

    name = "edac"
    metrics = (
        CollectorMetric("mc", MetricKind.DEVICE, "", "Memory controller, as the kernel names it (mc0)"),
        CollectorMetric("mc_name", MetricKind.SAMPLE, "", "The controller's model"),
        CollectorMetric("ce_count", MetricKind.COUNTER, "errors", "Corrected memory errors since boot"),
        CollectorMetric("ue_count", MetricKind.COUNTER, "errors", "Uncorrected memory errors since boot"),
    )

    def __init__(self, root: Path = Path("/sys/devices/system/edac/mc")) -> None:
        self.root = Path(root)
        self._controllers: list[_Controller] = []

    def probe(self) -> bool:
        """Open the counters of every memory controller; False when the kernel exposes no EDAC."""
        self._controllers = [_Controller(path) for path in sorted(self.root.glob("mc*"))]
        return bool(self._controllers)

    def sample(self, now_s: float) -> list[dict[str, Any]]:
        """One row per controller; the counts are cumulative, so a reader takes the deltas."""
        return [
            {
                "mc": controller.name,
                "mc_name": controller.model,
                "ce_count": controller.files["ce_count"].read_int(),
                "ue_count": controller.files["ue_count"].read_int(),
            }
            for controller in self._controllers
        ]

    def close(self) -> None:
        for controller in self._controllers:
            for polled in controller.files.values():
                polled.close()
        self._controllers = []
