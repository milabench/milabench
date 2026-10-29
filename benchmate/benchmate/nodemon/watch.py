"""Follow every node's samples live: `python -m benchmate.nodemon.watch DIR`.

Reads the TSVs a run is writing, incrementally, and prints one block per
refresh in three parts: readings (GPU clock and activity, link rates, CPU
clock, memory), findings (every Threshold crossed, on which node and device),
and the nodes ranked by score, the severities of their findings summed. With
--follow it stops once every sampler has finished. Stdlib only, like the
sampler.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .collector import CollectorMetric, MetricKind
from .collectors import (
    COLLECTORS,
    CpuCollector,
    EccCollector,
    EdacCollector,
    GpuCollector,
    IbCollector,
    MemCollector,
    NvlinkCollector,
    PcieCollector,
)
from .collectors.gpu import ClockReason
from .sink import TsvSink


class Ansi(str, Enum):
    """Terminal colours: green for a normal reading, yellow for one worth a look, red for a fault."""

    # A str mixin rather than StrEnum (Python 3.11+), which benchmate can't require yet.
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    RESET = "\033[0m"

    @classmethod
    def of(cls, severity: float) -> Ansi:
        """The colour of a severity, a threshold's or a node's total: green at 0, red from RED_FROM."""
        if severity >= RED_FROM:
            return cls.RED
        return cls.YELLOW if severity > 0 else cls.GREEN


#: The severity a line turns red at. Thresholds are yellow below it (1-3) and red above (6-10); a node's total
#: crosses it with one fault or with enough small findings, which is the point of adding them up.
RED_FROM = 5


@dataclass(frozen=True)
class Threshold:
    """One kind of finding, and what it costs a node's score; its colour follows. A Rule says how it is found."""

    name: str
    #: Added to the score of the node it is found on, once per device it is found on.
    severity: float
    #: Why it matters, for whoever reads or tunes this; never printed.
    explanation: str


GPU_POWER_CAP = Threshold(
    "power cap", 0,
    "The driver held the SM clock under the board power limit (SW_POWER_CAP). Usual at full load and worth a few "
    "percent at most, so it is listed in green and adds nothing to a node's score; when it slows a GPU well below "
    "its peers, GPU slow reports that.",
)
GPU_SLOW = Threshold(
    "GPU slow", 3,
    "The SM clock of a working GPU runs well below the median GPU's. In a synchronous job every GPU carries the same "
    "load, so the one that lags is the straggler, whether or not the driver gives a reason for the lower clock.",
)
GPU_THERMAL = Threshold(
    "thermal slowdown", 3,
    "The driver lowered the clock to stay under its temperature target (SW_THERMAL_SLOWDOWN): cooling is not "
    "keeping up.",
)
GPU_HARDWARE = Threshold(
    "hardware slowdown", 8,
    "The hardware itself cut the clock: HW_SLOWDOWN, HW_THERMAL_SLOWDOWN or HW_POWER_BRAKE_SLOWDOWN. A cooling "
    "or power supply fault; the node is a candidate to drain.",
)
XID = Threshold(
    "XID", 10,
    "The driver reported a GPU error event, from an application fault to a GPU that fell off the bus.",
)
ROW_REMAP = Threshold(
    "row remap", 10,
    "A pending remap needs a GPU reset to take effect; a failed one means the GPU has no spare rows left.",
)
GPU_ECC = Threshold(
    "GPU ECC uncorrected", 10,
    "An uncorrectable GPU memory error: whatever read that memory got wrong data or crashed.",
)
HOST_ECC_CORRECTED = Threshold(
    "host ECC corrected", 1,
    "Host memory corrected an error. Harmless alone; a rising count predicts an uncorrectable one.",
)
HOST_ECC_UNCORRECTED = Threshold(
    "host ECC uncorrected", 10,
    "An uncorrectable host memory error: the kernel or a process lost data.",
)
OOM_KILL = Threshold(
    "OOM kill", 6,
    "The kernel killed a process for memory; in a training job usually a rank, which leaves the others waiting.",
)
IB_LINK_ERROR = Threshold(
    "IB link error", 6,
    "Symbol errors or a link that went down: retransmits and stalls on that port, a cable or transceiver to check.",
)
IB_CONGESTION = Threshold(
    "IB congestion", 1,
    "xmit_wait: ticks a port had data but the fabric would not take it. Back-pressure, not a fault of the link, "
    "but it slows the collectives through that port.",
)
CPU_SLOW = Threshold(
    "CPU slow", 2,
    "The busy cores run well below the median node's. In a synchronous job every node carries the same load, so "
    "the one that lags is the straggler; against its own past the clock only follows the load.",
)
CPU_THERMAL = Threshold(
    "CPU thermal throttle", 2,
    "The package hit its thermal limit and throttled since the sampler started.",
)
PCIE_AER_FATAL = Threshold(
    "PCIe fatal error", 10,
    "A fatal uncorrectable PCIe error on a device (GPU, NIC, NVMe): the link cannot be trusted, and the driver "
    "resets it or the device falls off the bus.",
)
STALE = Threshold(
    "stale sampler", 8,
    "The sampler stopped writing without closing: killed, or the node hung. Its data ends there.",
)


#: At 60 nodes a fault is often on many of them; name this many and count the rest.
FLAGGED_HOSTS = 6


@dataclass(frozen=True)
class Finding:
    """One threshold crossed on one device of one node."""

    host: str
    threshold: Threshold
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
    """

    def __init__(self, path: Path, tail_bytes: int) -> None:
        self.path = path
        self.tail_bytes = tail_bytes
        self.columns: list[str] = []
        self.offset = 0
        #: The rows of the file's first sample, when the first read skipped ahead past them.
        self.first: list[dict[str, str]] = []

    def rows(self) -> list[dict[str, str]]:
        """The rows appended since the last call; a last line still being written is left for next time."""
        try:
            with self.path.open("rb") as stream:
                if not self.columns and not self._start(stream):
                    return []
                stream.seek(self.offset)
                chunk = stream.read()
        except OSError:
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
        """Take the samplers' status as of `now_s`, and drop what fell out of the window."""
        self.status = status
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


# --- rules: how each threshold is found ----------------------------------------------------------------------------


def every_row(row: dict[str, str]) -> bool:
    return True


def by_device(
    window: Window, metric: CollectorMetric, counts: Callable[[dict[str, str]], bool] = every_row
) -> dict[tuple[str, str], float]:
    """Each device's median of one metric over the window, from the rows `counts` accepts."""
    samples: dict[tuple[str, str], list[float]] = {}
    for host, row in window.rows.get(metric.collector, []):
        value = number(row, metric.name)
        if value is not None and counts(row):
            samples.setdefault((host, device_of(metric.collector, row)), []).append(value)
    return {key: percentile(values, 0.5) for key, values in samples.items()}


@dataclass(frozen=True)
class Rule(ABC):
    """How one threshold is found: the test it puts the window to."""

    threshold: Threshold

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
class Rise(MetricRule):
    """A cumulative counter that rose by more than `limit` since the sampler started."""

    limit: float = 0.0

    def __post_init__(self) -> None:
        if self.metric.kind is not MetricKind.COUNTER:
            raise ValueError(f"{self.threshold.name}: {self.metric.name} is not a counter, so it has no rise")

    def check(self, window: Window) -> Iterator[Finding]:
        for (host, device), change in window.counters.risen(self.collector, self.metric.name).items():
            if change > self.limit:
                yield Finding(host, self.threshold, device, f"{self.metric.name} +{change:.0f}")


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
            Finding(host, self.threshold, device_of(self.collector, row))
            for host, row in window.rows.get(self.collector, [])
            if self.hit(row)
        }


@dataclass(frozen=True)
class NonZero(MetricRule):
    """A column that is zero or empty when all is well: any other value is the finding."""

    def check(self, window: Window) -> Iterator[Finding]:
        yield from {
            Finding(host, self.threshold, device_of(self.collector, row), f"{self.metric.name} {value:g}")
            for host, row in window.rows.get(self.collector, [])
            for value in [number(row, self.metric.name)]
            if value
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
    counts: Callable[[dict[str, str]], bool] = every_row

    def check(self, window: Window) -> Iterator[Finding]:
        values = by_device(window, self.metric, self.counts)
        if not values:
            return
        median = percentile(list(values.values()), 0.5)
        for (host, device), value in values.items():
            change = 100 * (value / median - 1) if median else 0.0
            if change <= self.limit:
                detail = f"{value:.0f} {self.metric.unit}, {change:.0f}% from its peers' median"
                yield Finding(host, self.threshold, device, detail)


@dataclass(frozen=True)
class Stale(Rule):
    """A sampler that stopped writing without closing; the status files are the evidence, not the window's rows."""

    def check(self, window: Window) -> Iterator[Finding]:
        yield from (Finding(host, self.threshold) for host in sorted(window.dead_hosts()))


CLOCK_REASONS = GpuCollector.metric("clocks_event_reasons")
SM_CLOCK = GpuCollector.metric("sm_clock_mhz")
CPU_BUSY_MHZ = CpuCollector.metric("cpu_mhz_busy_mean")


def working(row: dict[str, str]) -> bool:
    """A gpu row where the GPU was not idling: an idle GPU's clock is down by design, not slow."""
    return not ClockReason.GPU_IDLE & ClockReason.parse(row.get(CLOCK_REASONS.name))


#: The rules that read why a GPU's clock is down, which also colour the clock reading.
CLOCK_RULES = (
    Flags(GPU_HARDWARE, CLOCK_REASONS,
          ClockReason.HW_SLOWDOWN | ClockReason.HW_THERMAL_SLOWDOWN | ClockReason.HW_POWER_BRAKE_SLOWDOWN),
    Flags(GPU_THERMAL, CLOCK_REASONS, ClockReason.SW_THERMAL_SLOWDOWN),
    Flags(GPU_POWER_CAP, CLOCK_REASONS, ClockReason.SW_POWER_CAP),
)

#: Every rule, in no particular order: findings are reported by severity. A threshold can be found by several.
RULES = (
    *CLOCK_RULES,
    NonZero(XID, GpuCollector.metric("xid")),
    # States, not counters: non-zero is the GPU's condition now, whenever it started.
    NonZero(ROW_REMAP, EccCollector.metric("remap_pending")),
    NonZero(ROW_REMAP, EccCollector.metric("remap_failure")),
    Rise(GPU_ECC, EccCollector.metric("ecc_uncorrected")),
    Rise(HOST_ECC_CORRECTED, EdacCollector.metric("ce_count")),
    Rise(HOST_ECC_UNCORRECTED, EdacCollector.metric("ue_count")),
    Rise(OOM_KILL, MemCollector.metric("oom_kill")),
    Rise(IB_LINK_ERROR, IbCollector.metric("symbol_error")),
    Rise(IB_LINK_ERROR, IbCollector.metric("link_downed")),
    Rise(IB_CONGESTION, IbCollector.metric("xmit_wait")),
    Rise(CPU_THERMAL, CpuCollector.metric("thermal_throttle_count")),
    Rise(PCIE_AER_FATAL, PcieCollector.metric("aer_fatal")),
    BelowPeers(CPU_SLOW, CPU_BUSY_MHZ, limit=-15.0),
    BelowPeers(GPU_SLOW, SM_CLOCK, limit=-10.0, counts=working),
    Stale(STALE),
)


def slowdowns_of(row: dict[str, str]) -> list[Threshold]:
    """The thresholds a gpu row's clock event reasons cross, in the order of CLOCK_RULES."""
    return [rule.threshold for rule in CLOCK_RULES if rule.hit(row)]


# --- findings: every threshold crossed, on which node and device ---------------------------------------------------


def findings(window: Window) -> list[Finding]:
    """Every threshold crossed in the window, on every node and device it was crossed on."""
    return [finding for rule in RULES for finding in rule.check(window)]


def ranked(found: list[Finding], hosts: Iterable[str] = ()) -> list[tuple[float, str, list[Finding]]]:
    """Every node, worst first: its score, the severities of its findings summed, and the findings.

    `hosts` adds the nodes found clean, at 0.
    """
    by_host: dict[str, list[Finding]] = {host: [] for host in hosts}
    for finding in found:
        by_host.setdefault(finding.host, []).append(finding)
    # Rounded: tenths summed in binary floating point leave 0.30000000000000004 behind.
    scored = [(round(sum(f.threshold.severity for f in kept), 3), host, kept) for host, kept in by_host.items()]
    return sorted(scored, key=lambda item: (-item[0], item[1]))


# --- readings: how the run is doing, green when nothing crossed a threshold --------------------------------------


def slowest_clock(window: Window) -> Line:
    """The GPU with the lowest SM clock in the window, coloured by the worst reason it was slowed for."""
    worst = None
    for host, row in window.rows.get(SM_CLOCK.collector, []):
        clock = number(row, SM_CLOCK.name)
        if clock is not None and (worst is None or clock < worst[0]):
            worst = (clock, f"{host}:{device_of(SM_CLOCK.collector, row)}", row)
    if worst is None:
        return None, "GPU clock    no samples"
    level = max((threshold.severity for threshold in slowdowns_of(worst[2])), default=0)
    reasons = ClockReason.parse(worst[2].get(CLOCK_REASONS.name)).names()
    return level, f"GPU clock    min {worst[0]:.0f} {SM_CLOCK.unit} on {worst[1]} ({reasons})"


def busiest(window: Window) -> Line:
    """How busy the GPUs are, preferring the duty cycle over the point sample."""
    duty, point = GpuCollector.metric("util_mean"), GpuCollector.metric("util")
    column = duty.name if window.values(duty.collector, duty.name) else point.name
    busy: dict[str, float] = {}
    for host, row in window.rows.get(duty.collector, []):
        value = number(row, column)
        if value is not None:
            key = f"{host}:{device_of(duty.collector, row)}"
            busy[key] = max(busy.get(key, 0.0), value)
    if not busy:
        return None, "GPU busy     no samples"
    ranking = sorted(busy.items(), key=lambda item: item[1], reverse=True)
    top = ", ".join(f"{name} {value:.2f}" for name, value in ranking[:3])
    return 0, f"GPU busy     median {human(percentile(list(busy.values()), 0.5))} ({column}); busiest {top}"


def readings(window: Window) -> list[Line]:
    """GPU clock and activity, link rates, CPU clock and host memory."""
    lines = [slowest_clock(window), busiest(window)]
    for label, collected, columns in (
        ("NVLink", NvlinkCollector, ("nvlink_tx_bytes_per_s", "nvlink_rx_bytes_per_s")),
        ("IB", IbCollector, ("ib_tx_bytes_per_s", "ib_rx_bytes_per_s")),
    ):
        values = [
            value
            for metric in map(collected.metric, columns)
            for value in window.values(metric.collector, metric.name)
        ]
        if values:
            lines.append((0, (
                f"{label:<12} p50 {rate(percentile(values, 0.5))}  "
                f"p90 {rate(percentile(values, 0.9))}  max {rate(max(values))}"
            )))
    clocks = by_device(window, CPU_BUSY_MHZ)
    if clocks:
        lines.append((0, f"{'CPU clock':<12} median {human(percentile(list(clocks.values()), 0.5), 0)} {CPU_BUSY_MHZ.unit}"))
    peak = MemCollector.metric("mem_used_peak_gib")
    peaks = window.values(peak.collector, peak.name)
    if peaks:
        lines.append((0, f"{'Host memory':<12} peak {human(max(peaks), 1)} {peak.unit}"))
    return lines


# --- rendering --------------------------------------------------------------------------------------------------------


def listed(items: list[str]) -> str:
    """The first few items and a count of the rest: on 60 nodes one finding can be on all of them."""
    shown = ", ".join(items[:FLAGGED_HOSTS])
    rest = len(items) - FLAGGED_HOSTS
    return f"{shown} and {rest} more" if rest > 0 else shown


def report(found: list[Finding]) -> list[Line]:
    """One line per threshold crossed, worst first, naming where."""
    by_threshold: dict[Threshold, list[Finding]] = {}
    for finding in found:
        by_threshold.setdefault(finding.threshold, []).append(finding)
    lines: list[Line] = []
    for threshold in sorted(by_threshold, key=lambda t: (-t.severity, t.name)):
        places = sorted(finding.where() for finding in by_threshold[threshold])
        lines.append((threshold.severity, f"{'':<12} {threshold.name}: {listed(places)}"))
    return lines


def ranking(found: list[Finding], hosts: Iterable[str]) -> list[Line]:
    """The nodes with a score, worst first, each coloured by its score; the clean ones counted."""
    scored = ranked(found, hosts)
    flagged = [(score, host, kept) for score, host, kept in scored if score]
    if not flagged:
        return [(0, f"{'Score':<12} 0 on all {len(scored)} node(s)")]
    lines: list[Line] = []
    for score, host, kept in flagged[:FLAGGED_HOSTS]:
        kinds = ", ".join(sorted({finding.threshold.name for finding in kept}))
        lines.append((score, f"{'Score':<12} {score:>5g}  {host}  {kinds}"))
    rest = len(flagged) - FLAGGED_HOSTS
    if rest > 0:
        lines.append((None, f"{'':<12} and {rest} more node(s) with a score"))
    clean = len(scored) - len(flagged)
    if clean:
        lines.append((0, f"{'':<12} {clean} node(s) at 0"))
    return lines


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


def paint(lines: list[Line], color: bool) -> str:
    """The lines as text, each in the colour of its severity when `color` is set."""
    return "\n".join(
        f"{Ansi.of(level).value}{text}{Ansi.RESET.value}" if color and level is not None else text
        for level, text in lines
    )


def block(window: Window, color: bool = False) -> str:
    """One refresh: the readings, every threshold crossed, and the nodes ranked by score."""
    hosts = {host for rows in window.rows.values() for host, _ in rows} | set(window.status)
    header = f"== {time.strftime('%H:%M:%S')}  {len(hosts)} node(s), last {window.window_s:.0f}s"
    found = findings(window)
    lines = [(None, header), *readings(window), *report(found), *ranking(found, hosts)]
    return paint(lines, color)


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
    args = parser.parse_args(argv)
    # With --follow the directory may not exist yet: the job that writes it can still be queued.
    if not args.follow and not args.directory.is_dir():
        print(f"No such directory: {args.directory}", file=sys.stderr)
        return 2
    color = sys.stdout.isatty() and "NO_COLOR" not in os.environ
    try:
        return watch(args.directory, args.window, args.follow, color)
    except KeyboardInterrupt:
        # Ctrl-C is how a follow is usually ended: no traceback, the shell's usual status for it.
        print(file=sys.stderr)
        return 130


def watch(directory: Path, window_s: float, follow: float | None, color: bool) -> int:
    """Print a block now, or every `follow` seconds until every sampler has finished.

    Nothing is printed before a sampler has written its status file: until then
    there is nothing to report on, and a block of "no samples" would only look
    like a fault.
    """
    tails: dict[Path, Tail] = {}
    window = Window(window_s)
    waiting = False
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
        print(block(window, color), flush=True)
        if not follow:
            return 0
        end = finished(window)
        if end:
            print(paint([end], color), flush=True)
            return 0
        time.sleep(follow)


if __name__ == "__main__":
    raise SystemExit(main())
