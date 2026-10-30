from __future__ import annotations

import logging
import re
from typing import Any, ClassVar
from urllib.error import URLError
from urllib.request import urlopen

from ..collector import CollectorMetric, MetricKind, NodeCollector

log = logging.getLogger(__name__)


class DcgmCollector(NodeCollector):
    """Tensor-pipe and SM activity, which NVML cannot report, from a dcgm-exporter already running here.

    Only from an exporter: scraping one is a plain read of what the host
    already publishes, and places no watch on the DCGM host engine. Without
    one the collector stays off rather than touching an engine this job
    shares with other people's, and GPU activity comes from NVML alone.
    """

    name = "dcgm"

    #: Each metric column -> the exporter series behind it.
    metric_fields: ClassVar[dict[str, str]] = {
        "sm_active": "DCGM_FI_PROF_SM_ACTIVE",
        "tensor_active": "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE",
    }
    #: The row: the GPU, then one cell per field above, in that order.
    metrics = (
        CollectorMetric("gpu", MetricKind.DEVICE, "", "The exporter's GPU number"),
        CollectorMetric("sm_active", MetricKind.SAMPLE, "fraction", "Share of time an SM had at least one warp active, averaged over the SMs (DCGM_FI_PROF_SM_ACTIVE)"),
        CollectorMetric("tensor_active", MetricKind.SAMPLE, "fraction", "Share of time the tensor pipe was active (DCGM_FI_PROF_PIPE_TENSOR_ACTIVE)"),
    )
    timeout_s = 2.0

    #: The labels a series carries that identify the GPU: the exporter's own number, and NVML's name for it.
    _labels = re.compile(r'\b(gpu|UUID)="([^"]*)"')

    def __init__(self, url: str = "http://localhost:9400/metrics") -> None:
        self.url = url
        #: The exporter's GPU number -> the UUID it labels that GPU with, for the status file.
        self._uuid_of: dict[int, str] = {}
        #: The pattern matching each column's series, compiled here because every tick scrapes.
        self._series = {
            column: self._pattern(field)
            for column, field in self.metric_fields.items()
        }

    @staticmethod
    def _pattern(field: str) -> re.Pattern:
        """One profiling field's exporter line: its labels, then the value."""
        return re.compile(rf'^{field}\{{([^}}]*)\}}\s+([0-9.eE+-]+)\s*$', re.MULTILINE)

    def probe(self) -> bool:
        """False when nothing answers at the exporter's address, or it publishes neither field."""
        if self.sample(0.0):
            return True
        log.info(f"No dcgm-exporter with the profiling fields at {self.url}; GPU activity comes from NVML alone.")
        return False

    def sample(self, now_s: float) -> list[dict[str, Any]]:
        """One row per GPU the exporter reports; empty when it stops answering."""
        text = self._scrape()
        if text is None:
            return []
        rows: dict[int, dict[str, Any]] = {}
        for column, pattern in self._series.items():
            for gpu, uuid, value in self._matches(text, pattern):
                rows.setdefault(gpu, {"gpu": gpu})[column] = value
                if uuid:
                    self._uuid_of[gpu] = uuid
        return [rows[gpu] for gpu in sorted(rows)]

    def _scrape(self) -> str | None:
        """The exporter's metrics, or None when it does not answer."""
        try:
            with urlopen(self.url, timeout=self.timeout_s) as response:
                return response.read().decode(errors="replace")
        except (URLError, OSError, ValueError):
            return None

    @classmethod
    def _matches(cls, text: str, pattern: re.Pattern) -> list[tuple[int, str, float]]:
        """The (gpu, uuid, value) triples of one series; a line without a GPU number is skipped."""
        found = []
        for labels, value in pattern.findall(text):
            named = dict(cls._labels.findall(labels))
            try:
                found.append((int(named["gpu"]), named.get("UUID", ""), float(value)))
            except (KeyError, ValueError):
                continue
        return found

    def describe(self) -> dict[str, Any]:
        """Where the numbers came from: they are another service's, not ours."""
        described: dict[str, Any] = {"url": self.url}
        if self._uuid_of:
            # Not "gpus", the key the NVML collectors publish: this is the exporter's own numbering,
            # and the point of recording it is that the analysis can check the two agree.
            described["exporter_gpus"] = {str(gpu): uuid for gpu, uuid in sorted(self._uuid_of.items())}
        return described
