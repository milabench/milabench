import math

from benchmate.nodemon import CollectorRegistry, NodeCollector


class FakeCollector(NodeCollector):
    columns = ("value",)

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

    def write(self, collector, rows, wall):
        self.rows.extend((collector.name, wall) for _ in rows)

    next_flush = math.inf

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
