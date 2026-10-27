from __future__ import annotations

import math
import threading
import time
from pathlib import Path
from typing import IO, Any

from .collector import NodeCollector


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
    """Rows for out_dir/nodes/<host>.<name>.tsv, held in memory and written every `flush_every` s.

    The output is usually on network storage, so rows are not written on
    every tick: each file with pending rows gets one write and flush per
    period, however wide its rows. ``flush_every=0`` writes as rows arrive.
    Rows still pending are lost on SIGKILL; ``close()`` writes them otherwise.
    """

    #: Columns every row starts with, before the collector's own. The host is in the
    #: file name and the benchmark in the status file, so neither is repeated per row.
    prefix = ("time_ms",)

    def __init__(self, out_dir: Path, host: str, flush_every: float = 10.0) -> None:
        self.directory = Path(out_dir) / "nodes"
        self.host = host
        self.flush_every = flush_every
        self._files: dict[str, IO[str]] = {}
        self._pending: dict[str, list[str]] = {}
        # Reentrant: a SIGUSR1 flush can land on the main thread while it is inside close().
        self._lock = threading.RLock()
        self._closed = False
        self.next_flush = math.inf
        self._schedule()

    def path(self, name: str) -> Path:
        return self.directory / f"{self.host}.{name}.tsv"

    def write(self, collector: NodeCollector, rows: list[dict[str, Any]], wall: float) -> None:
        """Format rows and queue them, or write them at once when flush_every is 0."""
        stamp = str(epoch_ms(wall))
        lines = []
        for row in rows:
            cells = [format_cell(row.get(column)) for column in collector.columns]
            lines.append(tsv_line([stamp, *cells]))
        with self._lock:
            if self._closed:  # a sampler that came unstuck after close()
                return
            if collector.name not in self._files:
                self._open(collector)
            self._pending[collector.name].extend(lines)
            if self.flush_every <= 0:
                self._write(collector.name)

    def _open(self, collector: NodeCollector) -> None:
        """Open a collector's file for appending; queue a header if it is new or empty."""
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.path(collector.name)
        stream = path.open("a")
        self._files[collector.name] = stream
        self._pending[collector.name] = []
        if path.stat().st_size == 0:
            self._pending[collector.name].append(tsv_line([*self.prefix, *collector.columns]))

    def _write(self, name: str) -> None:
        """One write and flush of a file's pending lines."""
        lines = self._pending[name]
        if lines:
            self._pending[name] = []
            stream = self._files[name]
            stream.write("".join(lines))
            stream.flush()

    def _schedule(self) -> None:
        if self.flush_every > 0:
            self.next_flush = time.monotonic() + self.flush_every

    def flush(self) -> None:
        with self._lock:
            for name in self._files:
                self._write(name)
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
            self.next_flush = math.inf
