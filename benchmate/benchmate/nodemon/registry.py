from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Iterator
from typing import Any, Protocol

from .collector import NodeCollector
from .collectors.nvml import NvmlCollector, NvmlGpus

log = logging.getLogger(__name__)


class Sink(Protocol):
    def write(
        self,
        collector: NodeCollector,
        rows: list[dict[str, Any]],
        sampled_at_unix_s: float
    ) -> None:
        ...

    #: The format the rows and the status file follow, recorded in the status file.
    schema: str
    #: Seconds between timed flushes; 0 writes rows as they arrive.
    flush_every_s: float
    #: The monotonic time of the next timed flush; inf when none is pending.
    next_flush_monotonic_s: float

    def flush(self) -> None: ...

    def close(self) -> None: ...


class _Entry:
    """A registered collector and its schedule."""

    def __init__(self, collector: NodeCollector, period_s: float) -> None:
        self.collector = collector
        self.period_s = period_s
        #: When this collector samples next, on the monotonic clock. Set by start(), then
        #: moved past the current time by each tick, so it always points at a future instant.
        self.next_tick_monotonic_s = 0.0
        self.failures = 0
        self.cpu_time = 0.0
        #: Instants a late tick passed without sampling. They have no row, so a count of rows is
        #: not a span of time; the status file reports this as `skipped`.
        self.skipped = 0


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
        #: The monotonic time start() ran; None before it.
        self._started: float | None = None
        self._closed = False
        #: The NVML session and its GPUs, shared by every NVML collector; None until started, or when NVML refused.
        self._nvml_gpus: NvmlGpus | None = None

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
                "skipped": entry.skipped,
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
        """Probe every collector, and align each first sample on a multiple of its period. Once only."""
        # A second start would open a second NVML session and lose the first, never closed.
        if self._started is not None:
            raise RuntimeError("CollectorRegistry.start() was already called")
        if now is None:
            now = time.monotonic()
        if wall is None:
            wall = time.time()
        # The real clock, not `now`: _log_cpu measures against time.monotonic(), and `now` may be a test's.
        self._started = time.monotonic()
        # Once for every NVML collector, so a refusal is not asked again by the next one.
        if any(isinstance(entry.collector, NvmlCollector) for entry in self._enabled()):
            self._nvml_gpus = NvmlGpus.start()
        for entry in self._enabled():
            if isinstance(entry.collector, NvmlCollector):
                entry.collector.nvml_gpus = self._nvml_gpus  # None when NVML did not start: the probe says no
            try:
                available = entry.collector.probe()
            except Exception:
                log.exception(f"Collector {entry.collector.name!r} failed to start, disabling it.")
                entry.collector.enabled = False
                continue
            if not available:
                log.info(f"Collector {entry.collector.name!r} is not available, disabling it.")
                entry.collector.enabled = False
                continue
            # Every node is due at the same wall-clock instants, so the first tick of each collector
            # lines up across hosts. Rows are stamped with the time they were read, not this instant.
            entry.next_tick_monotonic_s = now + (-wall) % entry.period_s
            log.debug(
                f"Collector {entry.collector.name!r}: every {entry.period_s}s, "
                f"first sample in {entry.next_tick_monotonic_s - now:.2f}s."
            )

    def tick(self, now: float | None = None, wall: float | None = None) -> None:
        """Sample the collectors that are due, then flush the sink if it is due. Never raises."""
        if now is None:
            now = time.monotonic()
        if wall is None:
            wall = time.time()
        for entry in self._enabled():
            if entry.next_tick_monotonic_s > now:
                continue
            # Sample on the grid next_tick_monotonic_s + k * period_s rather than sleeping period_s
            # after each sample, which would drift by the sampling time every round. A late tick
            # jumps to the first grid point after now and counts the points it passed as skipped,
            # instead of sampling them in a burst.
            missed = int((now - entry.next_tick_monotonic_s) // entry.period_s)
            if missed:
                log.debug(
                    f"Collector {entry.collector.name!r}: tick {now - entry.next_tick_monotonic_s:.2f}s late, "
                    f"{missed} sample(s) skipped."
                )
            entry.next_tick_monotonic_s += (missed + 1) * entry.period_s
            entry.skipped += missed
            self._sample(entry, now, sampled_at_unix_s=wall)  # the reads follow within this tick
        if self.sink and self.sink.next_flush_monotonic_s <= now:
            self._guard("sink", self.sink.flush)

    def _sample(self, entry: _Entry, now: float, sampled_at_unix_s: float) -> None:
        """Sample one collector, isolating its failures from the others."""
        self.sampling = entry.collector.name
        started = time.thread_time()
        try:
            rows = entry.collector.sample(now)
        except Exception as exc:
            entry.failures += 1
            if entry.collector.name not in self._logged:
                self._logged.add(entry.collector.name)
                log.exception(f"Collector {entry.collector.name!r} failed to sample.")
            else:
                log.debug(f"Collector {entry.collector.name!r} failed to sample ({entry.failures} in a row): {exc!r}")
            if entry.failures >= self.max_failures:
                log.exception(f"Collector {entry.collector.name!r} failed {entry.failures} times in a row, disabling it.")
                entry.collector.enabled = False
            return
        finally:
            entry.cpu_time += time.thread_time() - started
            self.sampling = None
        if entry.failures:
            log.debug(f"Collector {entry.collector.name!r} sampled again after {entry.failures} failure(s).")
        entry.failures = 0
        if rows and self.sink:
            self._guard("sink", self.sink.write, entry.collector, rows, sampled_at_unix_s)

    def _guard(self, name: str, fn: Callable[..., None], *args: Any) -> None:
        """Call fn, logging its first failure instead of raising: the sampler outlives a broken sink."""
        try:
            fn(*args)
        except Exception as exc:
            if name not in self._logged:
                self._logged.add(name)
                log.exception(f"{name} failed.")
            else:
                log.debug(f"{name} failed again: {exc!r}")

    def due(self) -> float:
        """The monotonic time of the next sample or sink flush, whichever comes first."""
        due = [entry.next_tick_monotonic_s for entry in self._enabled()]
        if self.sink:
            due.append(self.sink.next_flush_monotonic_s)
        return min(due, default=math.inf)

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
            if self._nvml_gpus:
                self._nvml_gpus.close()  # last: every collector's handles came from it
        else:
            log.warning(f"Collectors left open: the sampling thread is stuck in {self.sampling!r}.")
        if self.sink:
            self._guard("sink", self.sink.close)
        self._log_cpu()

    # TODO: Remove once the module is more mature: status.json already has each collector's cpu_s.
    def _log_cpu(self) -> None:
        """Log the CPU time the collectors used, in all and each."""
        if self._started is None:
            return  # never started: nothing was sampled
        elapsed = time.monotonic() - self._started
        entries = sorted(self._entries.values(), key=lambda entry: -entry.cpu_time)
        total = sum(entry.cpu_time for entry in entries)
        share = 100 * total / max(elapsed, 1e-9)
        each = ", ".join(f"{entry.collector.name} {entry.cpu_time:.3f}s" for entry in entries)
        log.info(f"CPU used over {elapsed:.1f}s: {total:.3f}s ({share:.2f}% of one core); {each}")
