import json

from benchmate.nodemon import CollectorRegistry, NodeCollector
from benchmate.nodemon.collectors import GpuCollector
from benchmate.nodemon.collectors.nvml import Gpu


class DescribingCollector(NodeCollector):
    name = "described"
    columns = ("value",)

    def sample(self, now_s):
        return [{"value": 1}]

    def describe(self):
        return {"extra": "kept"}


class BrokenDescribe(DescribingCollector):
    name = "broken"

    def describe(self):
        raise RuntimeError("no")


def test_a_collectors_own_fields_reach_the_status_and_a_broken_one_does_not_sink_it():
    registry = CollectorRegistry()
    registry.register(DescribingCollector(), 1.0)
    registry.register(BrokenDescribe(), 1.0)

    described = registry.describe()

    assert described["described"]["extra"] == "kept"
    assert described["described"]["period_s"] == 1.0
    assert described["broken"] == {"enabled": True, "period_s": 1.0, "cpu_s": 0.0}


def test_the_gpu_uuid_map_survives_close_since_the_status_is_written_after_it():
    collector = GpuCollector()
    collector._uuid_of = {1: "GPU-b", 0: "GPU-a"}
    collector._gpus = [Gpu(0, object(), "GPU-a")]

    collector.close()

    assert collector.describe() == {"gpus": {"0": "GPU-a", "1": "GPU-b"}}
    assert json.dumps(collector.describe())  # the status file must be able to hold it
