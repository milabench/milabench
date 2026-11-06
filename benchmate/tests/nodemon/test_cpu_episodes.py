"""Busy-core clock episodes are timed from the sample gaps, one package at a time."""

import importlib.util
from pathlib import Path


def _load():
    path = Path(__file__).resolve().parents[3] / "scripts" / "analysis" / "tt_analysis" / "cpu_episodes.py"
    spec = importlib.util.spec_from_file_location("cpu_episodes", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


episodes = _load().cpu_clock_episodes


def test_a_low_run_lasts_until_the_next_sample_and_packages_stay_apart():
    samples = [
        {"host": "trig0134", "package": 0, "time_ms": 0, "cpu_mhz_busy_mean": 3000},
        {"host": "trig0134", "package": 0, "time_ms": 2000, "cpu_mhz_busy_mean": 200},
        {"host": "trig0134", "package": 0, "time_ms": 4000, "cpu_mhz_busy_mean": 200},
        {"host": "trig0134", "package": 0, "time_ms": 6000, "cpu_mhz_busy_mean": 3000},
        {"host": "trig0134", "package": 0, "time_ms": 8000, "cpu_mhz_busy_mean": 3000},
        # The other package is low on the ticks the first is high. Merging them
        # would invent sub-sample episodes; they are separate series.
        {"host": "trig0134", "package": 1, "time_ms": 0, "cpu_mhz_busy_mean": 200},
        {"host": "trig0134", "package": 1, "time_ms": 2000, "cpu_mhz_busy_mean": 3000},
        {"host": "trig0134", "package": 1, "time_ms": 4000, "cpu_mhz_busy_mean": 200},
    ]

    by_package = {row["package"]: row for row in episodes(samples)}

    assert by_package["0"]["durations_s"] == [4.0]
    assert by_package["0"]["episodes"] == 1
    assert by_package["0"]["low_share"] == 0.5
    assert by_package["0"]["per_minute"] == 7.5
    assert by_package["1"]["durations_s"] == [2.0]  # the last low sample has no following gap
    assert by_package["1"]["episodes"] == 2


def test_two_samples_at_the_same_time_with_an_empty_clock_do_not_crash_the_sort():
    samples = [
        {"package": 0, "time_ms": 2000, "cpu_mhz_busy_mean": 3000},
        {"package": 0, "time_ms": 2000, "cpu_mhz_busy_mean": None},
        {"package": 0, "time_ms": 0, "cpu_mhz_busy_mean": 200},
    ]

    (row,) = episodes(samples)

    assert row["episodes"] == 1
    assert row["durations_s"] == [2.0]


def test_an_empty_busy_clock_is_not_a_low_sample():
    samples = [
        {"package": 0, "time_ms": 0, "cpu_mhz_busy_mean": 200},
        {"package": 0, "time_ms": 2000, "cpu_mhz_busy_mean": None},
        {"package": 0, "time_ms": 4000, "cpu_mhz_busy_mean": ""},
        {"package": 0, "time_ms": 6000, "cpu_mhz_busy_mean": 200},
        {"package": 0, "time_ms": 8000, "cpu_mhz_busy_mean": 3200},
    ]

    (row,) = episodes(samples)

    assert row["episodes"] == 2
    assert row["durations_s"] == [2.0, 2.0]
