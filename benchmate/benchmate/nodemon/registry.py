from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Iterator
from typing import Any, Protocol

from .collector import NodeCollector

log = logging.getLogger(__name__)


class Sink(Protocol):
    def write(self, collector: NodeCollector, rows: list[dict[str, Any]], wall: float) -> None: ...

    #: The monotonic time of the next timed flush; inf when none is pending.
    next_flush: float

    def flush(self) -> None: ...

    def close(self) -> None: ...


class _Entry:
    """A registered collector and its schedule."""

    def __init__(self, collector: NodeCollector, period_s: float) -> None:
        self.collector = collector
        self.period_s = period_s
        self.deadline = 0.0
        self.failures = 0
        self.cpu_time = 0.0


class CollectorRegistry:
    """Samples registered collectors on their own periods and hands the rows to a sink.

    Lifecycle: :meth:`register` -> :meth:`start` -> :meth:`tick` (repeatedly)
    -> :meth:`close`. ``close`` belongs in a ``finally``: it is safe to call
    twice and flushes whatever the sink still holds.
    """

    #: Consecutive failed samples before a collector is disabled.
    max_failures = 5

    def __init__(self, sink: Sink | None = None) -> None:
        self.sink = sink
        #: The collector being sampled right now, to name it if the sampler hangs.
        self.sampling: str | None = None
        self._entries: dict[str, _Entry] = {}
        self._logged: set[str] = set()
        self._started = time.monotonic()
        self._closed = False

    def register(self, collector: NodeCollector, period_s: float) -> NodeCollector:
        """Add a collector sampled every `period_s` seconds. Returns it."""
        if not collector.name:
            raise ValueError(f"{type(collector).__name__} must set a `name`")
        if collector.name in self._entries:
            raise ValueError(
                f"Collector {collector.name!r} is already registered; "
                f"registered: {sorted(self._entries)}"
            )
        if not period_s > 0:
            raise ValueError(f"Collector {collector.name!r}: period_s must be > 0, got {period_s}")
        self._entries[collector.name] = _Entry(collector, period_s)
        return collector

    def describe(self) -> dict[str, dict[str, Any]]:
        """Each collector's state, period, the CPU seconds its samples used, and what it adds itself."""
        return {
            name: {
                "enabled": entry.collector.enabled,
                "period_s": entry.period_s,
                "cpu_s": round(entry.cpu_time, 3),
                **self._described(entry.collector),
            }
            for name, entry
            in self._entries.items()
        }

    def _described(self, collector: NodeCollector) -> dict[str, Any]:
        """A collector's own status fields; a broken describe() must not cost us the status file."""
        try:
            return collector.describe()
        except Exception:
            if collector.name not in self._logged:
                self._logged.add(collector.name)
                log.exception(f"Collector {collector.name!r} failed to describe itself.")
            return {}

    def _enabled(self) -> Iterator[_Entry]:
        for entry in list(self._entries.values()):
            if entry.collector.enabled:
                yield entry

    def start(self, now: float | None = None, wall: float | None = None) -> None:
        """Probe every collector, and align each first sample on a multiple of its period."""
        if now is None:
            now = time.monotonic()
        if wall is None:
            wall = time.time()
        self._started = now
        for entry in self._enabled():
            collector = entry.collector
            try:
                available = collector.probe()
            except Exception:
                log.exception(f"Collector {collector.name!r} failed to start, disabling it.")
                collector.enabled = False
                continue
            if not available:
                log.info(f"Collector {collector.name!r} is not available, disabling it.")
                collector.enabled = False
                continue
            # Every node samples at the same wall-clock instants, so rows join on `time_ms`.
            entry.deadline = now + (-wall) % entry.period_s

    def tick(self, now: float | None = None, wall: float | None = None) -> None:
        """Sample the collectors that are due, then flush the sink if it is due. Never raises."""
        if now is None:
            now = time.monotonic()
        if wall is None:
            wall = time.time()
        for entry in self._enabled():
            if entry.deadline > now:
                continue
            # Deadlines rather than sleeping a fixed period each round: that drifts, since each
            # round takes the sampling time plus the sleep. The next deadline is one period
            # after the last, so samples stay on their aligned instants; a slow sample skips the
            # ticks it missed instead of bunching them up.
            missed = (now - entry.deadline) // entry.period_s
            entry.deadline += (missed + 1) * entry.period_s
            self._sample(entry, now, wall)
        if self.sink and self.sink.next_flush <= now:
            self._guard("sink", self.sink.flush)

    def _sample(self, entry: _Entry, now: float, wall: float) -> None:
        """Sample one collector, isolating its failures from the others."""
        collector = entry.collector
        self.sampling = collector.name
        started = time.thread_time()
        try:
            rows = collector.sample(now)
        except Exception:
            entry.failures += 1
            if collector.name not in self._logged:
                self._logged.add(collector.name)
                log.exception(f"Collector {collector.name!r} failed to sample.")
            if entry.failures >= self.max_failures:
                log.exception(f"Collector {collector.name!r} failed {entry.failures} times in a row, disabling it.")
                collector.enabled = False
            return
        finally:
            entry.cpu_time += time.thread_time() - started
            self.sampling = None
        entry.failures = 0
        if rows and self.sink:
            self._guard("sink", self.sink.write, collector, rows, wall)

    def _guard(self, name: str, fn: Callable[..., None], *args: Any) -> None:
        """Call fn, logging its first failure instead of raising: the sampler outlives a broken sink."""
        try:
            fn(*args)
        except Exception:
            if name not in self._logged:
                self._logged.add(name)
                log.exception(f"{name} failed.")

    def due(self) -> float:
        """The monotonic time of the next sample or sink flush, whichever comes first."""
        deadlines = [entry.deadline for entry in self._enabled()]
        if self.sink:
            deadlines.append(self.sink.next_flush)
        return min(deadlines, default=math.inf)

    def flush(self) -> None:
        if self.sink:
            self._guard("sink", self.sink.flush)

    def close(self, release_collectors: bool = True) -> None:
        """Release the collectors, then flush and close the sink. Safe to call twice.

        `release_collectors=False` leaves the collectors alone, for when the
        sampling thread is stuck inside one of them.
        """
        if self._closed:
            return
        self._closed = True
        if release_collectors:
            for entry in self._entries.values():
                try:
                    entry.collector.close()
                except Exception:
                    log.exception(f"Collector {entry.collector.name!r} failed to close.")
        if self.sink:
            self._guard("sink", self.sink.close)
        self._log_cpu()

    # TODO: Remove once the module is more mature: status.json already has each collector's cpu_s.
    def _log_cpu(self) -> None:
        """Log the CPU time the collectors used, in all and each."""
        elapsed = time.monotonic() - self._started
        entries = sorted(self._entries.values(), key=lambda entry: -entry.cpu_time)
        total = sum(entry.cpu_time for entry in entries)
        share = 100 * total / max(elapsed, 1e-9)
        each = ", ".join(f"{entry.collector.name} {entry.cpu_time:.3f}s" for entry in entries)
        log.info(f"CPU used over {elapsed:.1f}s: {total:.3f}s ({share:.2f}% of one core); {each}")
