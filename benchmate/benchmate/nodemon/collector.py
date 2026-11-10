from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, NamedTuple


class MetricKind(Enum):
    """How a reader treats a column."""

    #: Names the entity a row is for (a GPU, an HCA and its port).
    DEVICE = "device"
    #: A reading at the tick: a gauge, a rate, a flag or text.
    SAMPLE = "sample"
    #: Counts since boot or driver load rather than over the tick: a reader takes the deltas.
    COUNTER = "counter"


@dataclass(frozen=True)
class CollectorMetric:
    """One column a collector writes: what it is, in what unit, and how to read it."""

    name: str
    kind: MetricKind
    #: "MHz", "B/s", "GiB", a "fraction" of 1; empty for an id, a flag or text.
    unit: str
    description: str
    #: The collector writing it, the middle of nodes/<host>.<collector>.tsv; bound by NodeCollector.
    collector: str = ""


class NodeCollector(ABC):
    """One source of node-wide samples."""

    #: Unique key within a registry, and the file suffix: nodes/<host>.<name>.tsv
    name: str = ""
    #: Every column in wire order, after `time_ms`. Fixed whatever the hardware.
    metrics: tuple[CollectorMetric, ...] = ()
    #: The names of `metrics`, in order; derived from it.
    columns: tuple[str, ...] = ()
    #: Columns naming the entity a row is for (a GPU, an HCA and its port); empty when a row is the whole node.
    device_columns: tuple[str, ...] = ()
    #: Columns that count since boot or driver load rather than over the tick: a reader takes their deltas.
    counters: tuple[str, ...] = ()
    #: The single gate: a disabled collector is never sampled and writes no file.
    enabled: bool = True

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Bind each metric to the collector's name and derive the column lists from them."""
        super().__init_subclass__(**kwargs)
        cls.metrics = tuple(replace(metric, collector=cls.name) for metric in cls.metrics)
        cls.columns = tuple(metric.name for metric in cls.metrics)
        if len(set(cls.columns)) != len(cls.columns):
            raise ValueError(f"Collector {cls.name!r} declares a metric twice: {cls.columns}")
        cls.device_columns = cls._named(MetricKind.DEVICE)
        cls.counters = cls._named(MetricKind.COUNTER)

    @classmethod
    def _named(cls, kind: MetricKind) -> tuple[str, ...]:
        return tuple(metric.name for metric in cls.metrics if metric.kind is kind)

    @classmethod
    def metric(cls, name: str) -> CollectorMetric:
        """One of this collector's metrics; a misspelt name fails where it is written, not in a rule that never fires."""
        for metric in cls.metrics:
            if metric.name == name:
                return metric
        raise KeyError(f"Collector {cls.name!r} has no metric {name!r}; it has {cls.columns}")

    def probe(self) -> bool:
        """Open handles and resolve files once; False when unavailable on this host."""
        return True

    @abstractmethod
    def sample(self, now_s: float) -> list[dict[str, Any]]:
        """One row per entity; `now_s` is time.monotonic(), for rates."""

    def describe(self) -> dict[str, Any]:
        """What this collector adds to the status file; called before probe() and after close()."""
        return {}

    def close(self) -> None:
        """Release what probe() opened."""


class _Sample(NamedTuple):
    """A counter's value, and the monotonic time it was read."""

    time_s: float
    value: int


class Rate:
    """Per-second rates of cumulative counters, per key, timed on the monotonic clock.

    The monotonic clock keeps an NTP step of the wall clock from spiking or
    flipping a rate. A counter that went down was reset (a driver reload, a
    port reset).
    """

    def __init__(self) -> None:
        self._last: dict[Any, _Sample] = {}

    def update(self, key: Any, now_s: float, value: int | None) -> float | None:
        """Rate since this key's previous value; None on its first value, when unreadable, or after a reset."""
        if value is None:
            self._last.pop(key, None)
            return None
        previous = self._last.get(key)
        self._last[key] = _Sample(now_s, value)
        if previous is None or now_s <= previous.time_s or value < previous.value:
            return None
        return (value - previous.value) / (now_s - previous.time_s)


class PolledFile:
    """A file kept open and re-read in place, with no open/close per poll."""

    def __init__(self, path: Path, max_bytes: int | None = 4096) -> None:
        self.path = path
        self.max_bytes = max_bytes
        self.fd: int | None = None
        try:
            self.fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
        except OSError:
            pass

    @property
    def available(self) -> bool:
        return self.fd is not None

    def read(self) -> str | None:
        """The file's current content, or None when it can't be read."""
        if self.fd is None:
            return None
        try:
            return self._read(self.fd)
        except OSError:
            # A device that went away - still important to collect a value.
            return None

    def _read(self, fd: int) -> str:
        """Up to `max_bytes` from offset 0, or the whole file when it is None."""
        if self.max_bytes is not None:
            return os.pread(fd, self.max_bytes, 0).decode(errors="replace")
        # Re-reading an open sysfs file at offset 0 regenerates its value.
        # procfs returns about a page per read; reading on from offset 0 doesn't regenerate it.
        os.lseek(fd, 0, os.SEEK_SET)
        chunks = []
        while True:
            chunk = os.read(fd, 1 << 16)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks).decode(errors="replace")

    def read_int(self) -> int | None:
        """The integer in the file, or None when it can't be read or parsed."""
        text = self.read()
        try:
            return int(text) if text else None
        except ValueError:
            return None

    def close(self) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


def read_attr(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None
