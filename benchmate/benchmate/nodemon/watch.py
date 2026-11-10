"""Follow every node's samples live: `python -m benchmate.nodemon.watch DIR`.

Reads the TSVs a run is writing, incrementally, and prints one block per
refresh in three parts: readings (GPU clock and activity, link rates, CPU
clock, memory), findings (every rule that found something, on which node and device),
and the nodes ranked by score, the severities of their findings summed. With
--follow it stops once every sampler has finished, and on a terminal it
redraws the block in place instead of scrolling. Stdlib only, like the sampler.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO

from .collector import CollectorMetric, MetricKind
from .collectors import (
    COLLECTORS,
    CpuCollector,
    DcgmCollector,
    EccCollector,
    EdacCollector,
    GpmCollector,
    GpuCollector,
    IbCollector,
    MemCollector,
    NvlinkCollector,
    PcieCollector,
)
from .collectors.gpu import ClockReason
from .sink import TsvSink

# --- rules: what each kind of finding is, and how it is found ----------------------------------------------------------------------------


def by_device(
    window: Window, metric: CollectorMetric, counts: Callable[[dict[str, str]], bool] | None = None
) -> dict[tuple[str, str], float]:
    """Each device's median of one metric over the window, from the rows `counts` accepts (all, if None)."""
    samples: dict[tuple[str, str], list[float]] = {}
    for host, row in window.rows.get(metric.collector, []):
        value = number(row, metric.name)
        if value is not None and (counts is None or counts(row)):
            samples.setdefault((host, device_of(metric.collector, row)), []).append(value)
    return {key: median for key, values in samples.items() if (median := percentile(values, 0.5)) is not None}


@dataclass(frozen=True)
class Rule(ABC):
    """One kind of finding: what it costs a node's score, why it matters, and the test that finds it."""

    name: str
    #: Added to the score of the node it is found on, once per device it is found on; its colour follows.
    severity: float
    #: Why it matters, for whoever reads or tunes this; never printed.
    explanation: str

    @abstractmethod
    def check(self, window: Window) -> Iterator[Finding]:
        """Every finding in the window, at most one per device and what was seen."""


@dataclass(frozen=True)
class MetricRule(Rule, ABC):
    """A rule on one metric, read from the rows of the collector that writes it."""

    metric: CollectorMetric

    @property
    def collector(self) -> str:
        return self.metric.collector


@dataclass(frozen=True)
class CounterRule(MetricRule, ABC):
    """A rule on a cumulative counter."""

    def __post_init__(self) -> None:
        if self.metric.kind is not MetricKind.COUNTER:
            raise ValueError(f"{self.name}: {self.metric.name} is not a counter, so it has no rise")


@dataclass(frozen=True)
class Rise(CounterRule):
    """A cumulative counter that rose by more than `limit` since the sampler started."""

    limit: float = 0.0

    def check(self, window: Window) -> Iterator[Finding]:
        for (host, device), change in window.counters.risen(self.collector, self.metric.name).items():
            if change > self.limit:
                yield Finding(host, self, device, f"{self.metric.name} +{change:.0f}")


def rates(window: Window, metric: CollectorMetric) -> dict[tuple[str, str], float]:
    """Each device's average rise of a counter per second over the window; a device whose counter was reset has none."""
    ends: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for host, row in window.rows.get(metric.collector, []):
        value, stamp = number(row, metric.name), number(row, "time_ms")
        if value is not None and stamp is not None:
            seen = ends.setdefault((host, device_of(metric.collector, row)), [(stamp, value)] * 2)
            seen[1] = (stamp, value)
    return {
        device: (last - first) / ((last_ms - first_ms) / 1000)
        for device, ((first_ms, first), (last_ms, last)) in ends.items()
        if last_ms > first_ms and last >= first
    }


@dataclass(frozen=True)
class AbovePeers(CounterRule):
    """Devices whose counter rises at least `factor` times as fast as the median device's, and faster than `floor` a second.

    For a counter that always moves a little: a rise since the sampler started
    would flag every device, and what is a lot depends on the load. Only a device
    far above the others is worth a line. The floor keeps an idle median of zero
    from making every rise an outlier. One device alone is never flagged.
    """

    factor: float
    floor: float

    def check(self, window: Window) -> Iterator[Finding]:
        found = rates(window, self.metric)
        median = percentile(list(found.values()), 0.5)
        if median is None:
            return
        for (host, device), rate in found.items():
            if rate > self.floor and rate >= self.factor * median:
                yield Finding(host, self, device, f"{self.metric.name} {rate:,.0f}/s, median {median:,.0f}/s")


@dataclass(frozen=True)
class Flags(MetricRule):
    """A hex bit field with any bit of `mask` set."""

    mask: int

    def hit(self, row: dict[str, str]) -> bool:
        try:
            return bool(int(row.get(self.metric.name) or "0", 16) & self.mask)
        except ValueError:
            return False

    def check(self, window: Window) -> Iterator[Finding]:
        yield from {
            Finding(host, self, device_of(self.collector, row))
            for host, row in window.rows.get(self.collector, [])
            if self.hit(row)
        }


@dataclass(frozen=True)
class NonZero(MetricRule):
    """A column that is zero or empty when all is well: any other cell is the finding, as it is written.

    Text, not a number: several XIDs in one tick arrive as one cell, "79;45".
    """

    @staticmethod
    def raised(cell: str) -> bool:
        try:
            return float(cell) != 0
        except ValueError:
            return bool(cell)

    def check(self, window: Window) -> Iterator[Finding]:
        yield from {
            Finding(host, self, device_of(self.collector, row), f"{self.metric.name} {cell}")
            for host, row in window.rows.get(self.collector, [])
            for cell in [row.get(self.metric.name, "")]
            if self.raised(cell)
        }


@dataclass(frozen=True)
class BelowPeers(MetricRule):
    """Devices (a GPU, a CPU package) whose median of `metric` is at least `limit` percent below the median device's.

    Against the other devices, not against an earlier time: in a synchronous job
    every device carries the same load, so the one that is off is the straggler,
    whatever the load does to all of them. Each device is its own, so one bad GPU
    is not averaged into the rest of its node. Rows `counts` rejects are left
    out. One device alone is never flagged.
    """

    limit: float
    counts: Callable[[dict[str, str]], bool] | None = None

    def check(self, window: Window) -> Iterator[Finding]:
        values = by_device(window, self.metric, self.counts)
        if not values:
            return
        median = percentile(list(values.values()), 0.5)
        for (host, device), value in values.items():
            change = 100 * (value / median - 1) if median else 0.0
            if change <= self.limit:
                detail = f"{value:.0f} {self.metric.unit}, {change:.0f}% from its peers' median"
                yield Finding(host, self, device, detail)


@dataclass(frozen=True)
class Stale(Rule):
    """A sampler that stopped writing without closing; the status files are the evidence, not the window's rows."""

    def check(self, window: Window) -> Iterator[Finding]:
        yield from (Finding(host, self) for host in sorted(window.dead_hosts()))


CLOCK_REASONS = GpuCollector.metric("clocks_event_reasons")
SM_CLOCK = GpuCollector.metric("sm_clock_mhz")
CPU_BUSY_MHZ = CpuCollector.metric("cpu_mhz_busy_mean")


def working(row: dict[str, str]) -> bool:
    """A gpu row where the GPU was not idling: an idle GPU's clock is down by design, not slow."""
    return not ClockReason.GPU_IDLE & ClockReason.parse(row.get(CLOCK_REASONS.name))


#: The rules that read why a GPU's clock is down, which also colour the clock reading.
CLOCK_RULES = (
    Flags(
        "GPU hardware slowdown", 8,
        "The hardware itself cut the clock: HW_SLOWDOWN, HW_THERMAL_SLOWDOWN or HW_POWER_BRAKE_SLOWDOWN. A cooling "
        "or power supply fault; the node is a candidate to drain.",
        CLOCK_REASONS,
        ClockReason.HW_SLOWDOWN | ClockReason.HW_THERMAL_SLOWDOWN | ClockReason.HW_POWER_BRAKE_SLOWDOWN,
    ),
    Flags(
        "GPU thermal slowdown", 3,
        "The driver lowered the clock to stay under its temperature target (SW_THERMAL_SLOWDOWN): cooling is not "
        "keeping up.",
        CLOCK_REASONS,
        ClockReason.SW_THERMAL_SLOWDOWN,
    ),
)

STALE = Stale(
    "stale sampler", 8,
    "The sampler stopped writing without closing: killed, or the node hung. Its data ends there.",
)

#: Every rule, in no particular order: findings are reported by severity.
RULES = (
    *CLOCK_RULES,
    BelowPeers(
        "GPU slow", 3,
        "The SM clock of a working GPU runs well below the median GPU's. In a synchronous job every GPU carries the "
        "same load, so the one that lags is the straggler, whether or not the driver gives a reason for the lower "
        "clock.",
        SM_CLOCK, limit=-10.0, counts=working,
    ),
    BelowPeers(
        "CPU slow", 2,
        "The busy cores run well below the median node's. In a synchronous job every node carries the same load, so "
        "the one that lags is the straggler; against its own past the clock only follows the load.",
        CPU_BUSY_MHZ, limit=-15.0,
    ),
    # Both numbers are guesses, until a run on nodes the job has to itself shows what congestion looks like: the
    # unit of xmit_wait is not in the sysfs docs, and 10,000 a second is 0.004 % of the time waiting if a tick is 4 ns.
    AbovePeers(
        "IB congestion", 1,
        "xmit_wait: ticks a port had data but the fabric would not take it. Back-pressure, not a fault of the link, "
        "but it slows the collectives through that port. Found only on a port far above the others: every port waits "
        "a little, and a whole fabric congested alike is not a node to look at.",
        IbCollector.metric("xmit_wait"), factor=5.0, floor=1e4,
    ),
    Rise(
        "IB symbol errors", 6,
        "Symbol errors on a port: retransmits and stalls on it, a cable or transceiver to check.",
        IbCollector.metric("symbol_error"),
    ),
    Rise(
        "IB link down", 6,
        "A link that went down: the port stalls while it retrains, and one that keeps going down is a cable or "
        "transceiver to check.",
        IbCollector.metric("link_downed"),
    ),
    Rise(
        "CPU thermal throttle", 2,
        "The package hit its thermal limit and throttled since the sampler started.",
        CpuCollector.metric("thermal_throttle_count"),
    ),
    NonZero(
        "XID", 10,
        "The driver reported a GPU error event, from an application fault to a GPU that fell off the bus.",
        GpuCollector.metric("xid"),
    ),
    # States, not counters: non-zero is the GPU's condition now, whenever it started.
    NonZero(
        "row remap pending", 10,
        "A pending row remap needs a GPU reset to take effect.",
        EccCollector.metric("remap_pending"),
    ),
    NonZero(
        "row remap failed", 10,
        "A failed row remap: the GPU has no spare rows left.",
        EccCollector.metric("remap_failure"),
    ),
    Rise(
        "GPU ECC uncorrected", 10,
        "An uncorrectable GPU memory error: whatever read that memory got wrong data or crashed.",
        EccCollector.metric("ecc_uncorrected"),
    ),
    Rise(
        "host ECC uncorrected", 10,
        "An uncorrectable host memory error: the kernel or a process lost data.",
        EdacCollector.metric("ue_count"),
    ),
    Rise(
        "OOM kill", 6,
        "The kernel killed a process for memory; in a training job usually a rank, which leaves the others waiting.",
        MemCollector.metric("oom_kill"),
    ),
    Rise(
        "PCIe fatal error", 10,
        "A fatal uncorrectable PCIe error on a device (GPU, NIC, NVMe): the link cannot be trusted, and the driver "
        "resets it or the device falls off the bus.",
        PcieCollector.metric("aer_fatal"),
    ),
    STALE,
)


def slowdowns_of(row: dict[str, str]) -> list[Flags]:
    """The clock rules a gpu row's clock event reasons trip, in the order of CLOCK_RULES."""
    return [rule for rule in CLOCK_RULES if rule.hit(row)]


class Ansi(str, Enum):
    """Terminal styles: green for a normal reading, yellow for one worth a look, red for a fault, bold for a heading, purple for the summary."""

    # A str mixin rather than StrEnum (Python 3.11+), which benchmate can't require yet.
    BOLD = "\033[1m"
    PURPLE = "\033[35m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    RESET = "\033[0m"

    @classmethod
    def of(cls, severity: float) -> Ansi:
        """The colour of a severity, a rule's or a node's total: green at 0, red from RED_FROM."""
        if severity >= RED_FROM:
            return cls.RED
        return cls.YELLOW if severity > 0 else cls.GREEN


#: The level of a line that is a heading, drawn bold: a severity is never below 0.
HEADING = -1.0

#: The level of the line that sums up the nodes' health, drawn purple.
SUMMARY = -2.0

#: The styles of the levels that are not a severity.
STYLES = {HEADING: Ansi.BOLD, SUMMARY: Ansi.PURPLE}

#: The width of the readings' label column.
LABEL = 14

#: The severity a line turns red at. Rules are yellow below it (1-3) and red above (6-10); a node's total
#: crosses it with one fault or with enough small findings, which is the point of adding them up.
RED_FROM = 5


#: At 60 nodes a fault is often on many of them; name this many and count the rest.
FLAGGED_HOSTS = 8


@dataclass(frozen=True)
class Finding:
    """One rule's finding on one device of one node."""

    host: str
    rule: Rule
    #: Which device of the node, empty for the node itself: a GPU index, an HCA and port, a memory controller.
    device: str = ""
    #: What was seen, for the line: "xmit_wait +6", "1500 MHz, -38% from its peers' median".
    detail: str = ""

    def where(self) -> str:
        place = f"{self.host}:{self.device}" if self.device else self.host
        return f"{place} {self.detail}" if self.detail else place


class Tail:
    """One TSV read forward only: the header once, then whatever rows have been appended.

    Read as bytes, because the offsets are byte offsets: a text stream's tell()
    is an opaque cookie that cannot be seeked to arithmetically. The first read
    starts near the end rather than at the beginning, so watching a run that
    started hours ago does not pull its whole history into memory to throw all
    but the last window away.

    The file stays open between reads, which only seek and read: on a network
    file system every open is a round trip to the metadata server, and a follow
    would otherwise make one per file per refresh. close() releases it.
    """

    def __init__(self, path: Path, tail_bytes: int) -> None:
        self.path = path
        self.tail_bytes = tail_bytes
        self.columns: list[str] = []
        self.offset = 0
        #: The rows of the file's first sample, when the first read skipped ahead past them.
        self.first: list[dict[str, str]] = []
        self._stream: BinaryIO | None = None

    def rows(self) -> list[dict[str, str]]:
        """The rows appended since the last call; a last line still being written is left for next time."""
        try:
            if self._stream is None:
                self._stream = self.path.open("rb")
            stream = self._stream
            if not self.columns:
                stream.seek(0)  # a header still incomplete last time is read again from the top
                if not self._start(stream):
                    return []
            # SEEK_END asks the file system for the size now rather than the one cached at open:
            # Lustre and BeeGFS refresh it there, and SEEK_SET alone does not. Nothing new, no read.
            if stream.seek(0, os.SEEK_END) <= self.offset:
                return []
            stream.seek(self.offset)
            chunk = stream.read()
        except OSError:
            self.close()  # the handle may have gone bad; the next read opens the file afresh
            return []
        text = chunk.decode(errors="replace")
        complete, newline, _ = text.rpartition("\n")
        if not newline:
            return []
        self.offset += len(complete.encode()) + 1
        return [
            dict(zip(self.columns, line.split("\t")))
            for line in complete.split("\n")
            if line
        ]

    def close(self) -> None:
        """Release the file; a later rows() opens it again. Safe to call twice."""
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    def _start(self, stream) -> bool:
        """Take the header, then skip to the last `tail_bytes` of what is already there.

        False, and nothing taken, until the header is a whole line: a file can be
        there before its header is, and columns read from an empty or half-written
        line would be kept for good, every row after them unreadable.
        """
        header = stream.readline()
        if not header.endswith(b"\n"):
            return False
        self.columns = header.decode(errors="replace").rstrip("\n").split("\t")
        self.offset = stream.tell()
        end = stream.seek(0, 2)
        if end - self.offset <= self.tail_bytes:
            return True
        stream.seek(self.offset)
        self.first = self._first_sample(stream)
        stream.seek(end - self.tail_bytes)
        stream.readline()  # the row this landed inside is half a row
        self.offset = stream.tell()
        return True

    def _first_sample(self, stream) -> list[dict[str, str]]:
        """The rows sharing the first time_ms: one per device, the counters' values when the sampler started."""
        rows: list[dict[str, str]] = []
        while True:
            line = stream.readline()
            if not line.endswith(b"\n"):
                return rows
            row = dict(zip(self.columns, line.decode(errors="replace").rstrip("\n").split("\t")))
            if rows and row.get("time_ms") != rows[0].get("time_ms"):
                return rows
            rows.append(row)


def number(row: dict[str, str], column: str) -> float | None:
    """One cell as a float; an empty cell means unavailable, not zero."""
    text = row.get(column, "")
    try:
        return float(text) if text else None
    except ValueError:
        return None


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def host_of(path: Path) -> str:
    return path.name.split(".")[0]


def human(value: float | None, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def percent(fraction: float) -> str:
    """A 0 to 1 fraction as a whole percent: 0.94 as "94"."""
    return f"{fraction * 100:.0f}"


def rate(value: float | None) -> str:
    """Bytes per second as GB/s, the unit the links are specified in."""
    return "-" if value is None else f"{value / 1e9:.2f} GB/s"


def device_of(collector: str, row: dict[str, str]) -> str:
    """Which device of its node a row is for, from the collector's device columns: "0", "mlx5_4:1", or ""."""
    collected = COLLECTORS.get(collector)
    return ":".join(row.get(column, "?") for column in collected.device_columns) if collected else ""


class Counters:
    """How far each cumulative counter has risen since the sampler's first sample.

    They count since boot or driver load, so a value an earlier job left is not
    this run's. Kept per (collector, column, host, device) as the rows arrive, so
    a rise outlives the window the rows fell out of.
    """

    def __init__(self) -> None:
        #: Per key, the last value and the rise so far.
        self._seen: dict[tuple[str, str, str, str], tuple[float, float]] = {}

    def update(self, key: tuple[str, str, str, str], value: float) -> None:
        last, risen = self._seen.get(key, (value, 0.0))
        # A counter that went down was reset (a driver reload): what it holds now was counted since.
        self._seen[key] = (value, risen + (value - last if value >= last else value))

    def risen(self, collector: str, column: str) -> dict[tuple[str, str], float]:
        """Per (host, device), the rise of one counter, for those that rose."""
        return {
            (host, device): risen
            for (found, name, host, device), (_, risen) in self._seen.items()
            if found == collector and name == column and risen > 0
        }


class Window:
    """What every node reported in the last `window` seconds, and what its sampler said about itself."""

    #: Bytes of an existing file the first read goes back over, per second of the window.
    #: The busiest file measured writes about 450 B/s, so this covers the window with room over.
    bytes_per_second = 600
    #: Never read back less than this, however short the window.
    min_tail_bytes = 16384

    def __init__(self, window_s: float) -> None:
        self.window_s = window_s
        self.tail_bytes = max(self.min_tail_bytes, int(window_s * self.bytes_per_second))
        self.rows: dict[str, list[tuple[str, dict[str, str]]]] = {}
        self.counters = Counters()
        self.last_seen: dict[str, float] = {}
        #: Each host's status file, and the time the window ends at: set by refresh().
        self.status: dict[str, dict[str, Any]] = {}
        self.now_s = 0.0

    def add(self, collector: str, host: str, rows: list[dict[str, str]]) -> None:
        kept = self.rows.setdefault(collector, [])
        for row in rows:
            kept.append((host, row))
            stamp = number(row, "time_ms")
            if stamp:
                self.last_seen[host] = max(self.last_seen.get(host, 0.0), stamp / 1000)
        self.count(collector, host, rows)

    def count(self, collector: str, host: str, rows: Iterable[dict[str, str]]) -> None:
        """Feed the counters the collector declares from these rows."""
        collected = COLLECTORS.get(collector)
        if collected is None:
            return
        for row in rows:
            device = device_of(collector, row)
            for column in collected.counters:
                value = number(row, column)
                if value is not None:
                    self.counters.update((collector, column, host, device), value)

    def refresh(self, status: dict[str, dict[str, Any]], now_s: float) -> None:
        """Take the samplers' status as of `now_s`, and drop what fell out of the window.

        Once every sampler has written its end, the window closes at the last of them: a finished
        run is reported as it was, not as the empty stretch between its end and now.
        """
        self.status = status
        ends = [record.get("end") or 0 for record in status.values()]
        if status and all(ends):
            now_s = max(ends) / 1000
        self.now_s = now_s
        cutoff = (now_s - self.window_s) * 1000
        for collector, rows in self.rows.items():
            self.rows[collector] = [(host, row) for host, row in rows if (number(row, "time_ms") or 0) >= cutoff]

    def dead_hosts(self) -> set[str]:
        """Samplers whose status says running but that stopped writing: those nodes' data ends there."""
        dead = set()
        for host, record in self.status.items():
            if record.get("state") != "running":
                continue
            # Before its first flush a sampler has no rows yet: count from its start.
            last = self.last_seen.get(host, (record.get("start") or 0) / 1000)
            limit = 2 * (record.get("flush_every") or 10.0)
            if self.now_s - last > limit:
                dead.add(host)
        return dead

    def values(self, collector: str, column: str) -> list[float]:
        return [
            value
            for _, row in self.rows.get(collector, [])
            for value in [number(row, column)]
            if value is not None
        ]


def read(directory: Path, tails: dict[Path, Tail], window: Window) -> None:
    """Pull every TSV forward into the window; the rows are a level down, beside the status files."""
    rows = sorted((directory / TsvSink.rows_dir).glob("*.tsv")) or sorted(directory.glob("*.tsv"))
    for path in rows:
        parts = path.name.split(".")
        if len(parts) != 3:
            continue
        tail = tails.setdefault(path, Tail(path, window.tail_bytes))
        rows = tail.rows()
        # The counters start at the sampler's first sample, even when the first read skipped past it.
        window.count(parts[1], parts[0], tail.first)
        tail.first = []
        window.add(parts[1], parts[0], rows)


def statuses(directory: Path) -> dict[str, dict[str, Any]]:
    """Each host's status file, so a sampler that died can be told from one that stopped."""
    found = {}
    for path in sorted(directory.glob("*.status.json")):
        try:
            found[host_of(path)] = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
    return found


if TYPE_CHECKING:
    #: One output line and the severity it is coloured by; None leaves it plain.
    #: Only ever an annotation, so it is not evaluated on the older Pythons benchmate still supports.
    Line = tuple[float | None, str]


# --- findings: every rule that found something, on which node and device ---------------------------------------------------


def findings(window: Window) -> list[Finding]:
    """Every rule's findings in the window, on every node and device."""
    return [finding for rule in RULES for finding in rule.check(window)]


def ranked(found: list[Finding], hosts: Iterable[str] = ()) -> list[tuple[float, str, list[Finding]]]:
    """Every node, worst first: its score, the severities of its findings summed, and the findings.

    `hosts` adds the nodes found clean, at 0.
    """
    by_host: dict[str, list[Finding]] = {host: [] for host in hosts}
    for finding in found:
        by_host.setdefault(finding.host, []).append(finding)
    # Rounded: tenths summed in binary floating point leave 0.30000000000000004 behind.
    scored = [(round(sum(f.rule.severity for f in kept), 3), host, kept) for host, kept in by_host.items()]
    return sorted(scored, key=lambda item: (-item[0], item[1]))


# --- readings: how the run is doing, green when no rule found anything --------------------------------------


def slowest_clock(window: Window) -> tuple[float, str]:
    """The GPU with the lowest SM clock in the window, and the worst severity of the reasons it was slowed for."""
    worst = None
    for host, row in window.rows.get(SM_CLOCK.collector, []):
        clock = number(row, SM_CLOCK.name)
        if clock is not None and (worst is None or clock < worst[0]):
            worst = (clock, f"{host}:{device_of(SM_CLOCK.collector, row)}", row)
    if worst is None:
        return 0, "no samples"
    level = max((rule.severity for rule in slowdowns_of(worst[2])), default=0)
    reasons = ClockReason.parse(worst[2].get(CLOCK_REASONS.name)).names()
    return level, f"min {worst[0]:.0f} {SM_CLOCK.unit} on {worst[1]} ({reasons})"


def peaks(window: Window, metric: CollectorMetric) -> dict[str, float]:
    """The highest reading of a metric in the window, per GPU."""
    peak: dict[str, float] = {}
    for host, row in window.rows.get(metric.collector, []):
        value = number(row, metric.name)
        if value is not None:
            key = f"{host}:{device_of(metric.collector, row)}"
            peak[key] = max(peak.get(key, 0.0), value)
    return peak


def busiest(window: Window, metric: CollectorMetric) -> str:
    """The three busiest GPUs, by their busiest reading in the window."""
    busy = peaks(window, metric)
    if not busy:
        return "no samples"
    top = ", ".join(f"{name} {percent(value)}%" for name, value in sorted(busy.items(), key=lambda item: -item[1])[:3])
    return f"busiest {top} ({metric.name})"


def hottest(window: Window, metric: CollectorMetric) -> str:
    """The GPU with the highest temperature in the window; nothing without samples."""
    heat = peaks(window, metric)
    if not heat:
        return ""
    name = max(heat, key=lambda key: heat[key])
    return f"hottest {name} {heat[name]:.0f} {metric.unit}"


def readings(window: Window) -> list[Line]:
    """A table of the readings: p50, p90 and max of each over the window, and a note where one helps.

    The GPU rows are always there, so that no GPU samples is seen; the others only with samples.
    GPU busy is the duty cycle; the tensor and SM rows are what GPM and DCGM measure, when either answers.
    """
    duty, point = GpuCollector.metric("util_mean"), GpuCollector.metric("util")
    busy = duty if window.values(duty.collector, duty.name) else point
    memory = MemCollector.metric("mem_used_gib")
    power, temp, cpu_busy = GpuCollector.metric("power_w"), GpuCollector.metric("temp_c"), CpuCollector.metric("cpu_util")
    clock_level, clock_note = slowest_clock(window)
    #: Per row: its label, the metrics pooled for it, how a value is written, its severity, and its note.
    rows = (
        ("GPU clock", (SM_CLOCK,), lambda value: f"{value:.0f} {SM_CLOCK.unit}", clock_level, clock_note),
        ("GPU busy %", (busy,), percent, 0, busiest(window, busy)),
        ("GPU tensor %", (GpmCollector.metric("tensor_util"),), percent, 0, ""),
        ("GPU SM active", (DcgmCollector.metric("sm_active"),), lambda value: f"{value:.2f}", 0, ""),
        ("GPU power", (power,), lambda value: f"{value:.0f} {power.unit}", 0, ""),
        ("GPU temp", (temp,), lambda value: f"{value:.0f} {temp.unit}", 0, hottest(window, temp)),
        ("NVLink", (NvlinkCollector.metric("nvlink_tx_bytes_per_s"), NvlinkCollector.metric("nvlink_rx_bytes_per_s")),
         rate, 0, ""),
        ("IB", (IbCollector.metric("ib_tx_bytes_per_s"), IbCollector.metric("ib_rx_bytes_per_s")), rate, 0, ""),
        ("CPU clock", (CPU_BUSY_MHZ,), lambda value: f"{value:.0f} {CPU_BUSY_MHZ.unit}", 0, ""),
        ("CPU busy", (cpu_busy,), lambda value: f"{value:.2f}", 0, ""),
        ("Host memory", (memory,), lambda value: f"{value:.1f} {memory.unit}", 0, ""),
    )
    table = []
    for label, metrics, write, level, note in rows:
        values = [value for metric in metrics for value in window.values(metric.collector, metric.name)]
        if values or note:
            cells = [write(percentile(values, 0.5)), write(percentile(values, 0.9)), write(max(values))] if values else ["-"] * 3
            table.append((label, cells, level, note))
    width = max([3] + [len(cell) for _, cells, _, _ in table for cell in cells])
    lines: list[Line] = [(HEADING, f"{'':<{LABEL}} {'p50':>{width}}  {'p90':>{width}}  {'max':>{width}}  Note")] if table else []
    for label, cells, level, note in table:
        columns = "  ".join(f"{cell:>{width}}" for cell in cells)
        lines.append((level or None, f"{label:<{LABEL}} {columns}  {note}".rstrip()))
    return lines


# --- rendering --------------------------------------------------------------------------------------------------------


def nodes(count: int) -> str:
    """A count of nodes, in words: "1 node", "9 nodes"."""
    return f"{count} node" + ("" if count == 1 else "s")


def listed(items: list[str], limit: int) -> str:
    """The first `limit` items and a count of the rest: on 60 nodes one finding can be on all of them."""
    shown = ", ".join(items[:limit])
    rest = len(items) - limit
    return f"{shown} and {rest} more" if rest > 0 else shown


def report(found: list[Finding], flagged_hosts: int) -> list[Line]:
    """One line per rule that found something, worst first, naming where."""
    by_rule: dict[Rule, list[Finding]] = {}
    for finding in found:
        by_rule.setdefault(finding.rule, []).append(finding)
    lines: list[Line] = []
    for rule in sorted(by_rule, key=lambda r: (-r.severity, r.name)):
        places = sorted(finding.where() for finding in by_rule[rule])
        label = "" if lines else "Findings"
        lines.append((rule.severity, f"{label:<12} {rule.name}: {listed(places, flagged_hosts)}"))
    return lines


def ranking(found: list[Finding], hosts: Iterable[str], flagged_hosts: int) -> list[Line]:
    """A table of the nodes with a score, worst first, each row coloured by its score.

    Columns: the score, the node, and the rules that found something on it.
    """
    flagged = [(score, host, kept) for score, host, kept in ranked(found, hosts) if score]
    if not flagged:
        return []
    shown = flagged[:flagged_hosts]
    width = max(len(host) for _, host, _ in shown)
    lines: list[Line] = [(HEADING, f"{'Score':>5}  {'Node':<{width}}  Reasons")]
    for score, host, kept in shown:
        reasons = ", ".join(sorted({finding.rule.name for finding in kept}))
        lines.append((score, f"{score:>5g}  {host:<{width}}  {reasons}"))
    rest = len(flagged) - flagged_hosts
    if rest > 0:
        lines.append((None, f"{'':>5}  and {nodes(rest)} more with a score"))
    return lines


def healthy(found: list[Finding], hosts: Iterable[str], covered: str) -> list[Line]:
    """How many nodes have no score over the time covered: "all 9 nodes healthy, ...", or "2/3 nodes healthy, ..."."""
    scored = ranked(found, hosts)
    clean = sum(1 for score, _, _ in scored if not score)
    if clean == len(scored):
        return [(SUMMARY, f"all {nodes(clean)} healthy, {covered}")]
    return [(SUMMARY, f"{clean}/{len(scored)} nodes healthy, {covered}")]


def finished(window: Window) -> Line | None:
    """Once every sampler has written how it ended, the line that says so; None while one is still running.

    A stale sampler keeps the follow going: stale is a guess from the rows, and a
    slow file system can make a live sampler look like one. Its status still says
    running, so it never gets here; Ctrl-C ends the follow of a run whose sampler was killed.
    """
    status = window.status.values()
    if not status or any(record.get("state") == "running" for record in status):
        return None
    ends: dict[str, int] = {}
    for record in status:
        state = str(record.get("state"))
        ends[state] = ends.get(state, 0) + 1
    summary = ", ".join(f"{count} {state}" for state, count in sorted(ends.items()))
    level = 0 if set(ends) == {"closed"} else STALE.severity
    return level, f"== every sampler has finished: {summary}"


def clip(line: str, width: int) -> str:
    """A line cut to `width` visible characters, its colour codes kept."""
    parts = re.split(r"(\033\[[0-9;]*m)", line)
    seen = 0
    for index in range(0, len(parts), 2):
        parts[index] = parts[index][: max(width - seen, 0)]
        seen += len(parts[index])
    return "".join(parts)


def redraw(text: str, first: bool) -> str:
    """A block as a terminal screen drawn over the last one: the first clears the screen, later ones start at its top.

    Each line clears what the last block left to its right, and what is below the
    block is cleared. Lines past the terminal's width or height are cut, since a
    line that wraps or a block taller than the screen would scroll the heading
    off the top and garble the next.
    """
    size = shutil.get_terminal_size()
    rows = max(size.lines - 1, 2)
    lines = [clip(line, size.columns) for line in text.splitlines()]
    if len(lines) > rows:
        lines = [*lines[: rows - 1], f"... {len(lines) - rows + 1} more line(s): the terminal is too short"]
    return ("\033[2J" if first else "") + "\033[H" + "\n".join(line + "\033[K" for line in lines) + "\033[J"


def paint(lines: list[Line], color: bool) -> str:
    """The lines as text, each in the colour of its severity (or bold, for a heading) when `color` is set."""
    return "\n".join(
        f"{(STYLES.get(level) or Ansi.of(level)).value}{text}{Ansi.RESET.value}"
        if color and level is not None else text
        for level, text in lines
    )


def span(window: Window) -> str:
    """The time from the oldest row in the window to the newest, as clock times; with a date when it crosses midnight."""
    stamps = [stamp for rows in window.rows.values() for _, row in rows for stamp in [number(row, "time_ms")] if stamp]
    if not stamps:
        return ""
    first, last = (datetime.fromtimestamp(stamp / 1000) for stamp in (min(stamps), max(stamps)))
    pattern = "%H:%M:%S" if first.date() == last.date() else "%m-%d %H:%M:%S"
    return f"{first.strftime(pattern)} to {last.strftime(pattern)}"


def block(window: Window, color: bool = False, flagged_hosts: int = FLAGGED_HOSTS, stamp: bool = True) -> str:
    """One refresh: the readings, every rule that found something, and the nodes ranked by score, a blank line apart.

    It starts with a line giving the time and the node count, unless `stamp` is off: a block redrawn in place is
    always the latest, and the time says nothing a clock doesn't. A blank line starts it then.
    """
    hosts = {host for rows in window.rows.values() for host, _ in rows} | set(window.status)
    times = span(window)
    covered = f"{times} (last {window.window_s:.0f}s)" if times else f"last {window.window_s:.0f}s"
    found = findings(window)
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    sections = [
        [(HEADING, f"== {now}  {nodes(len(hosts))}")] if stamp else [],
        readings(window),
        healthy(found, hosts, covered),
        report(found, flagged_hosts),
        ranking(found, hosts, flagged_hosts),
    ]
    lines: list[Line] = []
    for section in filter(None, sections):
        lines += [(None, ""), *section] if lines else section
    return paint(lines if stamp else [(None, ""), *lines], color)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m benchmate.nodemon.watch",
        description="Follow the node samples of a run: clocks, activity, links, CPU, failures.",
    )
    parser.add_argument("directory", type=Path, help="A sampler output directory (BENCHMATE_NODEMON_DIR)")
    parser.add_argument("--follow", type=float, metavar="SECONDS",
                        help="Refresh every SECONDS instead of once, until every sampler has finished")
    parser.add_argument("--window", type=float, default=60.0, metavar="SECONDS",
                        help="How much history each refresh reports on (default: 60)")
    parser.add_argument("--flagged-hosts", type=int, default=FLAGGED_HOSTS, metavar="N",
                        help="Hosts or devices to name on a line before counting the rest (default: %(default)s)")
    args = parser.parse_args(argv)
    if args.flagged_hosts < 1:
        parser.error("--flagged-hosts must be at least 1")
    # With --follow the directory may not exist yet: the job that writes it can still be queued.
    if not args.follow and not args.directory.is_dir():
        print(f"No such directory: {args.directory}", file=sys.stderr)
        return 2
    color = sys.stdout.isatty() and "NO_COLOR" not in os.environ
    try:
        return watch(args.directory, args.window, args.follow, color, args.flagged_hosts)
    except KeyboardInterrupt:
        # Ctrl-C is how a follow is usually ended: no traceback, the shell's usual status for it.
        print(file=sys.stderr)
        return 130


def watch(directory: Path, window_s: float, follow: float | None, color: bool, flagged_hosts: int) -> int:
    """Print a block now, or every `follow` seconds until every sampler has finished.

    Nothing is printed before a sampler has written its status file: until then
    there is nothing to report on, and a block of "no samples" would only look
    like a fault.

    A follow on a terminal redraws the block in place; anywhere else (a pipe, a
    file such as a job's output) each block is appended.
    """
    tails: dict[Path, Tail] = {}
    window = Window(window_s)
    waiting = False
    in_place = bool(follow) and sys.stdout.isatty() and os.environ.get("TERM") != "dumb"
    drawn = False
    try:
        while True:
            status = statuses(directory)
            if not status:
                if not follow:
                    print(f"No sampler has started in {directory} yet", file=sys.stderr)
                    return 1
                if not waiting:
                    print(f"Waiting for a sampler to start in {directory} ...", file=sys.stderr, flush=True)
                    waiting = True
                time.sleep(follow)
                continue
            read(directory, tails, window)
            window.refresh(status, time.time())
            text = block(window, color, flagged_hosts, stamp=not in_place)
            print(redraw(text, first=not drawn) if in_place else text, flush=True)
            drawn = True
            if not follow:
                return 0
            end = finished(window)
            if end:
                print(paint([end], color), flush=True)
                return 0
            time.sleep(follow)
    finally:
        for tail in tails.values():
            tail.close()


if __name__ == "__main__":
    raise SystemExit(main())
