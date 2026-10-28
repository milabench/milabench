from __future__ import annotations

import logging
import re
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

from ..collector import NodeCollector

log = logging.getLogger(__name__)


class DcgmCollector(NodeCollector):
    """Tensor-pipe and SM activity, which NVML cannot report, from a dcgm-exporter already running here.

    Only from an exporter: scraping one is a plain read of what the host
    already publishes, and places no watch on the DCGM host engine. Without
    one the collector stays off rather than touching an engine this job
    shares with other people's, and GPU activity comes from NVML alone.
    """

    name = "dcgm"
    columns = (
        "gpu",
        "sm_active",
        "tensor_active",
    )
    timeout_s = 2.0

    def __init__(self, url: str = "http://localhost:9400/metrics") -> None:
        self.url = url
        #: The exporter series behind each column, compiled here because every tick scrapes.
        self._series = {
            "sm_active": self._pattern("DCGM_FI_PROF_SM_ACTIVE"),
            "tensor_active": self._pattern("DCGM_FI_PROF_PIPE_TENSOR_ACTIVE"),
        }

    @staticmethod
    def _pattern(field: str) -> re.Pattern:
        """One profiling field's exporter line: its labels, of which only the GPU matters, then the value."""
        return re.compile(rf'^{field}\{{[^}}]*\bgpu="(\d+)"[^}}]*\}}\s+([0-9.eE+-]+)\s*$', re.MULTILINE)

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
            for gpu, value in self._matches(text, pattern):
                rows.setdefault(gpu, {"gpu": gpu})[column] = value
        return [rows[gpu] for gpu in sorted(rows)]

    def _scrape(self) -> str | None:
        """The exporter's metrics, or None when it does not answer."""
        try:
            with urlopen(self.url, timeout=self.timeout_s) as response:
                return response.read().decode(errors="replace")
        except (URLError, OSError, ValueError):
            return None

    def _matches(self, text: str, pattern: re.Pattern) -> list[tuple[int, float]]:
        """The (gpu, value) pairs of one series in the exporter's output."""
        found = []
        for gpu, value in pattern.findall(text):
            try:
                found.append((int(gpu), float(value)))
            except ValueError:
                continue
        return found

    def describe(self) -> dict[str, Any]:
        """Where the numbers came from: they are another service's, not ours."""
        return {"url": self.url}
