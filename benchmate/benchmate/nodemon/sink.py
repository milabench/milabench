from __future__ import annotations

import logging
import math
import random
import threading
import time
from pathlib import Path
from typing import IO, Any

from .collector import NodeCollector

log = logging.getLogger(__name__)


def epoch_ms(seconds: float) -> int:
    """Unix time in whole milliseconds, the unit of every timestamp nodemon writes."""
    return round(seconds * 1000)


def format_cell(value: Any) -> str:
    """One TSV cell: empty for None or NaN, 6 significant digits for floats."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return "" if math.isnan(value) else f"{value:.6g}"
    if isinstance(value, int):
        return str(value)
    return str(value).replace("\t", " ").replace("\n", " ")


def tsv_line(cells: list[str]) -> str:
    return "\t".join(cells) + "\n"


class TsvSink:
    """Rows for out_dir/nodes/<host>.<name>.tsv, buffered and written every `flush_every_s`.

    One write per file per period, shifted by a random ±`flush_jitter_s` so the nodes
    don't all hit network storage at once. ``flush_every_s=0`` writes as rows arrive.
    Pending rows are lost on SIGKILL; ``close()`` writes them otherwise.
    """

    #: The format of everything one sampler writes: these columns, and the status file.
    schema = "benchmate.nodemon.v3"
    #: The rows go one level down, leaving the run's status and lock visible beside it.
    rows_dir = "nodes"
    #: Columns every row starts with, before the collector's own. The host is in the
    #: file name and the benchmark in the status file, so neither is repeated per row.
    prefix = ("time_ms",)

    def __init__(self,
        out_dir: Path,
        host_name: str,
        flush_every_s: float = 10.0,
        flush_jitter_s: float = 0.25
    ) -> None:
        self.directory = Path(out_dir)
        self.host_name = host_name
        self.flush_every_s = flush_every_s
        self.flush_jitter_s = flush_jitter_s
        #: Seeded from the OS, so every host draws its own delays.
        self._random = random.Random()
        self._files: dict[str, IO[str]] = {}
        self._pending: dict[str, list[str]] = {}
        # Reentrant: a SIGUSR1 flush can land on the main thread while it is inside close().
        self._lock = threading.RLock()
        self._closed = False
        self.next_flush_monotonic_s = math.inf
        self._schedule()

    def tsv_path(self, name: str) -> Path:
        """One collector's file: <out_dir>/nodes/<host>.<collector>.tsv."""
        return self.directory / self.rows_dir / f"{self.host_name}.{name}.tsv"

    def status_path(self) -> Path:
        """Where this host's sampler records how it started and stopped."""
        return self.directory / f"{self.host_name}.status.json"

    def lock_path(self) -> Path:
        """The lock that keeps one sampler per host."""
        return self.directory / f"{self.host_name}.lock"

    def write(self, collector: NodeCollector, rows: list[dict[str, Any]], sampled_at_unix_s: float) -> None:
        """Format rows and queue them, or write them at once when flush_every_s is 0."""
        stamp = str(epoch_ms(sampled_at_unix_s))
        lines = []
        for row in rows:
            cells = [format_cell(row.get(column)) for column in collector.columns]
            lines.append(tsv_line([stamp, *cells]))
        with self._lock:
            if self._closed:  # a sampler that came unstuck after close()
                log.debug(f"{len(rows)} {collector.name} row(s) dropped: the sink is closed.")
                return
            if collector.name not in self._files:
                self._open(collector)
            self._pending[collector.name].extend(lines)
            if self.flush_every_s <= 0:
                self._write(collector.name)

    def _open(self, collector: NodeCollector) -> None:
        """Open a collector's file for appending, writing its header at once if it is new or empty.

        At once rather than with the first flush: a reader that finds the file
        must find its columns too, not an empty file for the first period.
        """
        path = self.tsv_path(collector.name)
        path.parent.mkdir(parents=True, exist_ok=True)
        stream = path.open("a")
        self._files[collector.name] = stream
        self._pending[collector.name] = []
        if path.stat().st_size == 0:
            stream.write(tsv_line([*self.prefix, *collector.columns]))
            stream.flush()
            log.debug(f"Opened {path}, header written.")
        else:
            log.debug(f"Opened {path}, appending.")

    def _write(self, name: str) -> int:
        """One write and flush of a file's pending lines; how many it wrote."""
        lines = self._pending[name]
        if lines:
            self._pending[name] = []
            stream = self._files[name]
            stream.write("".join(lines))
            stream.flush()
        return len(lines)

    def _schedule(self) -> None:
        if self.flush_every_s > 0:
            jitter = self._random.uniform(-self.flush_jitter_s, self.flush_jitter_s) if self.flush_jitter_s > 0 else 0.0
            self.next_flush_monotonic_s = time.monotonic() + self.flush_every_s + jitter

    def flush(self) -> None:
        with self._lock:
            started = time.monotonic()
            written = sum(self._write(name) for name in self._files)
            if written:
                # Slow writes here point at the shared filesystem rather than the sampler.
                log.debug(f"Flushed {written} row(s) to {len(self._files)} file(s) in {1e3 * (time.monotonic() - started):.1f} ms.")
            self._schedule()

    def close(self) -> None:
        """Write the pending rows and close every file. Safe to call twice."""
        with self._lock:
            self.flush()
            self._closed = True
            for stream in self._files.values():
                stream.close()
            self._files.clear()
            self._pending.clear()
            self.next_flush_monotonic_s = math.inf
