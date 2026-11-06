"""Per-package runs of a low busy-core clock, timed from the real sample gaps.

``cpu_mhz_busy_mean`` is one row per package. An empty cell means no core was
busy, not 0 MHz, so it is not a low sample. Duration is the time from the first
low sample of a run to the next sample that is not low. A run that is still low
at the last sample has no following gap: several low samples span from the first
to the last, and a single trailing sample has no measured duration.
"""

from __future__ import annotations

from typing import Any


def cpu_clock_episodes(
    samples: list[dict[str, Any]],
    *,
    below_mhz: float = 500,
    column: str = "cpu_mhz_busy_mean",
) -> list[dict[str, Any]]:
    """One summary per (host, package), using `time_ms` gaps rather than a fixed period.

    `samples` are dicts with `time_ms`, `package`, and the clock column. `host`
    is optional. Samples of one package are sorted by `time_ms`.
    """
    grouped: dict[tuple[str, str], list[tuple[int, float | None]]] = {}
    for sample in samples:
        host = "" if sample.get("host") is None else str(sample.get("host"))
        package = "" if sample.get("package") is None else str(sample.get("package"))
        stamp = sample.get("time_ms")
        if stamp is None:
            continue
        grouped.setdefault((host, package), []).append((int(stamp), _clock(sample.get(column))))
    return [_summarize(host, package, rows, below_mhz) for (host, package), rows in sorted(grouped.items())]


def _clock(value: Any) -> float | None:
    """A clock reading; empty, not a number, is missing rather than zero."""
    if value is None or value == "":
        return None
    try:
        clock = float(value)
    except (TypeError, ValueError):
        return None
    if clock != clock:  # NaN
        return None
    return clock


def _summarize(host: str, package: str, rows: list[tuple[int, float | None]], below_mhz: float) -> dict[str, Any]:
    rows = sorted(rows)
    durations: list[float] = []
    episodes = 0
    start: int | None = None
    last_low: int | None = None
    for stamp, clock in rows:
        low = clock is not None and clock < below_mhz
        if low:
            if start is None:
                start = stamp
                episodes += 1
            last_low = stamp
            continue
        if start is not None:
            durations.append((stamp - start) / 1000)
            start = None
    # Still low at the last sample: the gaps between those samples are known, and
    # there is no sample after them. One trailing sample has no gap to measure.
    if start is not None and last_low is not None and last_low > start:
        durations.append((last_low - start) / 1000)
    span_s = (rows[-1][0] - rows[0][0]) / 1000 if len(rows) > 1 else 0.0
    low_time_s = sum(durations)
    return {
        "host": host,
        "package": package,
        "episodes": episodes,
        "span_s": span_s,
        "low_time_s": low_time_s,
        "low_share": low_time_s / span_s if span_s else None,
        "per_minute": episodes / (span_s / 60) if span_s else None,
        "duration_p90_s": _percentile(durations, 0.9),
        "durations_s": durations,
    }


def _percentile(values: list[float], fraction: float) -> float | None:
    """The linear-interpolated percentile; None when nothing was measured."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[lower + 1] * weight
