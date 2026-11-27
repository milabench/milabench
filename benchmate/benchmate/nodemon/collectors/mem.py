from __future__ import annotations

from pathlib import Path
from typing import Any

from ..collector import CollectorMetric, MetricKind, NodeCollector, PolledFile


def parse_keyed(text: str) -> dict[str, int]:
    """`key value [kB]` lines, as in /proc/meminfo and /proc/vmstat, with kB in bytes."""
    values = {}
    for line in text.splitlines():
        parts = line.replace(":", " ").split()
        if len(parts) < 2:
            continue
        try:
            value = int(parts[1])
        except ValueError:
            continue
        if len(parts) > 2 and parts[2] == "kB":
            value *= 1024
        values[parts[0]] = value
    return values


class MemCollector(NodeCollector):
    name = "mem"
    metrics = (
        CollectorMetric("mem_used_gib", MetricKind.SAMPLE, "GiB", "Memory in use: total less available"),
        CollectorMetric("mem_total_gib", MetricKind.SAMPLE, "GiB", "Total host memory"),
        CollectorMetric("mem_used_peak_gib", MetricKind.SAMPLE, "GiB", "Highest mem_used_gib since the sampler started"),
        CollectorMetric("anon_gib", MetricKind.SAMPLE, "GiB", "Anonymous memory (AnonPages)"),
        CollectorMetric("file_gib", MetricKind.SAMPLE, "GiB", "Page cache (Cached)"),
        CollectorMetric("oom_kill", MetricKind.COUNTER, "count", "Processes the kernel killed for memory since boot"),
    )
    #: Bytes per GiB, the unit of the memory columns.
    gib = 1 << 30

    def __init__(self, proc: Path = Path("/proc")) -> None:
        self.proc = Path(proc)
        self._meminfo: PolledFile | None = None
        self._vmstat: PolledFile | None = None
        self._peak = 0

    def probe(self) -> bool:
        self._meminfo = PolledFile(self.proc / "meminfo", max_bytes=None)
        self._vmstat = PolledFile(self.proc / "vmstat", max_bytes=None)
        return self._meminfo.available

    def sample(self, now_s: float) -> list[dict[str, Any]]:
        """One row: used (total - available) with its running peak, anon, page cache, OOM kills."""
        if not self._meminfo or not self._vmstat:  # closed
            return []
        meminfo = parse_keyed(self._meminfo.read() or "")
        vmstat = parse_keyed(self._vmstat.read() or "")
        total = meminfo.get("MemTotal")
        available = meminfo.get("MemAvailable")
        used = None if total is None or available is None else total - available
        if used is not None:
            self._peak = max(self._peak, used)
        return [
            {
                "mem_used_gib": self._gib(used),
                "mem_total_gib": self._gib(total),
                "mem_used_peak_gib": self._gib(self._peak if used is not None else None),
                "anon_gib": self._gib(meminfo.get("AnonPages")),
                "file_gib": self._gib(meminfo.get("Cached")),
                "oom_kill": vmstat.get("oom_kill"),
            }
        ]

    def _gib(self, value: int | None) -> float | None:
        return None if value is None else value / self.gib

    def close(self) -> None:
        """Close the /proc files."""
        if self._meminfo:
            self._meminfo.close()
        if self._vmstat:
            self._vmstat.close()
        self._meminfo = None
        self._vmstat = None
