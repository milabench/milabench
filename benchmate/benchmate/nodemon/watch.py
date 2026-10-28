"""Follow every node's samples live: `python -m benchmate.nodemon.watch DIR`.

Reads the TSVs a run is writing, incrementally, and prints one block per
refresh: the worst GPU clock, how busy the GPUs are, link rates and errors,
CPU clocks that have fallen, memory, and any node whose sampler has died.
Stdlib only, like the sampler.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable

from .sink import TsvSink

#: Flag a node whose CPU clock has fallen at least this far since the run's first window.
CPU_DROP_FLAG = -15.0
#: At 60 nodes a fault is often on many of them; name this many and count the rest.
FLAGGED_HOSTS = 6


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

    def rows(self) -> list[dict[str, str]]:
        """The rows appended since the last call; a last line still being written is left for next time."""
        try:
            with self.path.open("rb") as stream:
                if not self.columns:
                    self._start(stream)
                stream.seek(self.offset)
                chunk = stream.read()
        except OSError:
            return []
        text = chunk.decode(errors="replace")
        complete, newline, partial = text.rpartition("\n")
        if not newline:
            return []
        self.offset += len(complete.encode()) + 1
        return [
            dict(zip(self.columns, line.split("\t")))
            for line in complete.split("\n")
            if line
        ]

    def _start(self, stream) -> None:
        """Take the header, then skip to the last `tail_bytes` of what is already there."""
        self.columns = stream.readline().decode(errors="replace").rstrip("\n").split("\t")
        self.offset = stream.tell()
        end = stream.seek(0, 2)
        if end - self.offset <= self.tail_bytes:
            return
        stream.seek(end - self.tail_bytes)
        stream.readline()  # the row this landed inside is half a row
        self.offset = stream.tell()


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


def names(hosts: set, limit: int = 6) -> str:
    """A few host names: on 60 nodes a fault can be on all of them, and the line still has to be read."""
    listed = sorted(hosts)
    if len(listed) <= limit:
        return ", ".join(listed)
    return ", ".join(listed[:limit]) + f" and {len(listed) - limit} more"


def human(value: float | None, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def rate(value: float | None) -> str:
    """Bytes per second as GB/s, the unit the links are specified in."""
    return "-" if value is None else f"{value / 1e9:.2f} GB/s"


class Window:
    """What every node reported in the last `window` seconds."""

    #: Bytes of an existing file the first read goes back over, per second of the window.
    #: The busiest file measured writes about 450 B/s, so this covers the window with room over.
    bytes_per_second = 600
    #: Never read back less than this, however short the window.
    min_tail_bytes = 16384

    def __init__(self, window_s: float) -> None:
        self.window_s = window_s
        self.tail_bytes = max(self.min_tail_bytes, int(window_s * self.bytes_per_second))
        self.rows: dict[str, list[tuple[str, dict[str, str]]]] = {}
        self.first_cpu: dict[str, float] = {}
        self.last_seen: dict[str, float] = {}

    def add(self, collector: str, host: str, rows: Iterable[dict[str, str]]) -> None:
        kept = self.rows.setdefault(collector, [])
        for row in rows:
            kept.append((host, row))
            stamp = number(row, "time_ms")
            if stamp:
                self.last_seen[host] = max(self.last_seen.get(host, 0.0), stamp / 1000)

    def trim(self, now_s: float) -> None:
        """Drop what fell out of the window, keeping each host's first CPU clock as the baseline."""
        for host, row in self.rows.get("cpu", []):
            busy = number(row, "cpu_mhz_busy_mean")
            if busy and host not in self.first_cpu:
                self.first_cpu[host] = busy
        cutoff = (now_s - self.window_s) * 1000
        for collector, rows in self.rows.items():
            self.rows[collector] = [(host, row) for host, row in rows if (number(row, "time_ms") or 0) >= cutoff]

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
        window.add(parts[1], parts[0], tail.rows())


def statuses(directory: Path) -> dict[str, dict[str, Any]]:
    """Each host's status file, so a sampler that died can be told from one that stopped."""
    found = {}
    for path in sorted(directory.glob("*.status.json")):
        try:
            found[host_of(path)] = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
    return found


def slowest_clock(window: Window) -> str:
    """The GPU with the lowest SM clock in the window, and why it was slowed."""
    worst = None
    for host, row in window.rows.get("gpu", []):
        clock = number(row, "sm_clock_mhz")
        if clock is not None and (worst is None or clock < worst[0]):
            worst = (clock, f"{host}:{row.get('gpu', '?')}", row.get("clocks_event_reasons", ""))
    if worst is None:
        return "GPU clock    no samples"
    return f"GPU clock    min {worst[0]:.0f} MHz on {worst[1]} (reasons {worst[2] or '-'})"


def busiest(window: Window) -> str:
    """How busy the GPUs are, preferring the duty cycle over the point sample."""
    column = "util_mean" if window.values("gpu", "util_mean") else "util"
    busy: dict[str, float] = {}
    for host, row in window.rows.get("gpu", []):
        value = number(row, column)
        if value is not None:
            key = f"{host}:{row.get('gpu', '?')}"
            busy[key] = max(busy.get(key, 0.0), value)
    if not busy:
        return "GPU busy     no samples"
    ranked = sorted(busy.items(), key=lambda item: item[1], reverse=True)
    top = ", ".join(f"{name} {value:.2f}" for name, value in ranked[:3])
    return f"GPU busy     median {human(percentile(list(busy.values()), 0.5))} ({column}); busiest {top}"


def links(window: Window) -> list[str]:
    """NVLink and InfiniBand rates, and any error counter that is not zero."""
    lines = []
    for collector, label, columns in (
        ("nvlink", "NVLink", ("nvlink_tx_bytes_per_s", "nvlink_rx_bytes_per_s")),
        ("ib", "IB", ("ib_tx_bytes_per_s", "ib_rx_bytes_per_s")),
    ):
        values = window.values(collector, columns[0]) + window.values(collector, columns[1])
        if values:
            lines.append(
                f"{label:<12} p50 {rate(percentile(values, 0.5))}  "
                f"p90 {rate(percentile(values, 0.9))}  max {rate(max(values))}"
            )
    faults = []
    for column in ("xmit_wait", "symbol_error", "link_downed"):
        hosts = {host for host, row in window.rows.get("ib", []) if (number(row, column) or 0) > 0}
        if hosts:
            faults.append(f"{column} on {names(hosts)}")
    if faults:
        lines.append(f"{'IB errors':<12} {'; '.join(faults)}")
    return lines


def cpu_clocks(window: Window) -> list[str]:
    """Per node, the busy-core clock and how far it has fallen since the first window."""
    now: dict[str, float] = {}
    for host, row in window.rows.get("cpu", []):
        busy = number(row, "cpu_mhz_busy_mean")
        if busy is not None:
            now[host] = min(now.get(host, busy), busy)
    if not now:
        return []
    lines = [f"{'CPU clock':<12} median {human(percentile(list(now.values()), 0.5), 0)} MHz"]
    fallen = []
    for host, value in now.items():
        first = window.first_cpu.get(host)
        if not first:
            continue
        change = 100 * (value / first - 1)
        if change <= CPU_DROP_FLAG:
            fallen.append((change, host, value))
    fallen.sort()
    for change, host, value in fallen[:FLAGGED_HOSTS]:
        lines.append(f"{'':<12} FELL  {host} {value:.0f} MHz, {change:.0f}% since the first window")
    if len(fallen) > FLAGGED_HOSTS:
        lines.append(f"{'':<12} FELL  and {len(fallen) - FLAGGED_HOSTS} more node(s)")
    return lines


def failures(window: Window) -> list[str]:
    """Anything that should never be non-zero: XIDs, ECC, EDAC, OOM kills."""
    lines = []
    for host, row in window.rows.get("gpu", []):
        if row.get("xid"):
            lines.append(f"{'XID':<12} {host}:{row.get('gpu', '?')} xid {row['xid']}")
    for collector, columns in (
        ("ecc", ("ecc_uncorrected", "remap_pending", "remap_failure")),
        ("edac", ("ce_count", "ue_count")),
        ("mem", ("oom_kill",)),
    ):
        for column in columns:
            hosts = {host for host, row in window.rows.get(collector, []) if (number(row, column) or 0) > 0}
            if hosts:
                lines.append(f"{column:<12} non-zero on {names(hosts)}")
    return lines


def memory(window: Window) -> list[str]:
    peaks = window.values("mem", "mem_used_peak_gib")
    return [f"{'Host memory':<12} peak {human(max(peaks), 1)} GiB"] if peaks else []


def stale(window: Window, status: dict[str, dict[str, Any]], now_s: float) -> list[str]:
    """Samplers that should still be writing but stopped: those nodes' data ends there."""
    dead = set()
    for host, record in status.items():
        if record.get("state") != "running":
            continue
        last = window.last_seen.get(host)
        limit = 2 * (record.get("flush_every") or 10.0)
        if last is None or now_s - last > limit:
            dead.add(host)
    if not dead:
        return []
    return [f"{'STALE':<12} {len(dead)} sampler(s) still 'running' with no recent row: {names(dead)}"]


def block(window: Window, status: dict[str, dict[str, Any]], now_s: float) -> str:
    """One refresh, as plain lines."""
    hosts = len({host for rows in window.rows.values() for host, _ in rows})
    header = f"== {time.strftime('%H:%M:%S')}  {hosts} node(s), last {window.window_s:.0f}s"
    lines = [header, slowest_clock(window), busiest(window)]
    lines += links(window) + cpu_clocks(window) + memory(window) + failures(window)
    lines += stale(window, status, now_s)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m benchmate.nodemon.watch",
        description="Follow the node samples of a run: clocks, activity, links, CPU, failures.",
    )
    parser.add_argument("directory", type=Path, help="A sampler output directory (BENCHMATE_NODEMON_DIR)")
    parser.add_argument("--follow", type=float, metavar="SECONDS", help="Refresh every SECONDS instead of once")
    parser.add_argument("--window", type=float, default=60.0, metavar="SECONDS",
                        help="How much history each refresh reports on (default: 60)")
    args = parser.parse_args(argv)
    if not args.directory.is_dir():
        print(f"No such directory: {args.directory}", file=sys.stderr)
        return 2

    tails: dict[Path, Tail] = {}
    window = Window(args.window)
    while True:
        read(args.directory, tails, window)
        now = time.time()
        window.trim(now)
        print(block(window, statuses(args.directory), now), flush=True)
        if not args.follow:
            return 0
        time.sleep(args.follow)


if __name__ == "__main__":
    raise SystemExit(main())
