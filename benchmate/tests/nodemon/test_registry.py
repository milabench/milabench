import math

import pytest

from benchmate.nodemon import (
    CollectorMetric,
    CollectorRegistry,
    MetricKind,
    NodeCollector,
)


class FakeCollector(NodeCollector):
    metrics = (CollectorMetric("value", MetricKind.SAMPLE, "", "calls so far"),)

    def __init__(self, name, fail=False):
        self.name = name
        self.fail = fail
        self.samples = 0

    def sample(self, now_s):
        self.samples += 1
        if self.fail:
            raise RuntimeError("broken")
        return [{"value": self.samples}]


class ListSink:
    def __init__(self):
        self.rows = []

    def write(self, collector, rows, sampled_at_unix_s):
        self.rows.extend((collector.name, sampled_at_unix_s) for _ in rows)

    schema = "test"
    flush_every_s = 0.0
    next_flush_monotonic_s = math.inf

    def flush(self):
        pass

    def close(self):
        pass


def test_each_collector_keeps_its_period_and_failures_stay_isolated():
    sink = ListSink()
    registry = CollectorRegistry(sink)
    fast = FakeCollector("fast")
    broken = FakeCollector("broken", fail=True)
    registry.register(fast, period_s=1.0)
    registry.register(FakeCollector("slow"), period_s=5.0)
    registry.register(broken, period_s=1.0)

    registry.start(now=0.0, wall=1000.0)
    for second in range(10):
        registry.tick(now=float(second), wall=1000.0 + second)

    assert fast.samples == 10
    assert [wall for name, wall in sink.rows if name == "slow"] == [1000.0, 1005.0]
    assert broken.samples == registry.max_failures
    assert not broken.enabled
    assert fast.enabled


def test_first_sample_lands_on_a_multiple_of_the_period():
    registry = CollectorRegistry(ListSink())
    collector = FakeCollector("c")
    registry.register(collector, period_s=5.0)

    registry.start(now=100.0, wall=1002.5)  # the next multiple of 5 s is 2.5 s away

    assert registry.due() == 102.5
    registry.tick(now=102.4, wall=1004.9)
    assert collector.samples == 0
    registry.tick(now=102.5, wall=1005.0)
    assert collector.samples == 1


def test_a_late_tick_is_stamped_with_its_read_time_and_the_skipped_ones_are_counted():
    sink = ListSink()
    registry = CollectorRegistry(sink)
    registry.register(FakeCollector("c"), period_s=5.0)
    registry.start(now=100.0, wall=1002.5)  # first due instant is monotonic 102.5, wall 1005

    registry.tick(now=112.6, wall=1015.1)  # due instants 102.5 and 107.5 passed unsampled

    assert sink.rows == [("c", 1015.1)]  # the read time, not the due instant 1012.5
    assert registry.describe()["c"]["skipped"] == 2


def test_every_collector_the_default_registry_samples_can_be_read_back_by_watch():
    from benchmate.nodemon import default_registry
    from benchmate.nodemon.collectors import COLLECTORS

    # watch finds a collector's device columns and counters in COLLECTORS, and ignores a file it does not list.
    assert set(default_registry().describe()) <= set(COLLECTORS)


def test_start_runs_once():
    registry = CollectorRegistry(ListSink())
    registry.register(FakeCollector("fast"), period_s=1.0)
    registry.start(now=0.0, wall=1000.0)
    with pytest.raises(RuntimeError, match="already called"):
        registry.start(now=1.0, wall=1001.0)
