"""The GPU collector's own arithmetic, against a stub NVML: no driver needed, and it is the
part that could be wrong without crashing."""

from types import SimpleNamespace

import pytest

from benchmate.nodemon.collectors import GpuCollector
from benchmate.nodemon.collectors.nvml import Gpu


class FakeNvmlError(Exception):
    def __init__(self, value=0):
        self.value = value


def sample(timestamp, value):
    return SimpleNamespace(timeStamp=timestamp, sampleValue=SimpleNamespace(uiVal=value))


def fake_nvml(samples=(), events=()):
    """Just the NVML surface the collector touches."""
    remaining = list(events)

    def wait(event_set, timeout_ms):
        if not remaining:
            raise FakeNvmlError(7)  # NVML_ERROR_TIMEOUT
        return remaining.pop(0)

    def get_samples(handle, kind, since):
        newer = [s for s in samples if s.timeStamp > since]
        if not newer:
            raise FakeNvmlError(6)  # NVML_ERROR_NOT_FOUND
        return (0, newer)

    return SimpleNamespace(
        NVMLError=FakeNvmlError,
        NVML_ERROR_NOT_FOUND=6,
        NVML_ERROR_NOT_SUPPORTED=3,
        NVML_GPU_UTILIZATION_SAMPLES=1,
        nvmlEventTypeXidCriticalError=0x8,
        nvmlDeviceGetSamples=get_samples,
        nvmlEventSetWait=wait,
        nvmlDeviceGetIndex=lambda device: device,
    )


def test_the_duty_cycle_averages_the_buffered_samples_and_asks_only_for_newer_ones():
    collector = GpuCollector()
    collector._nvml = fake_nvml(samples=[sample(10, 100), sample(20, 50), sample(30, 90)])
    gpu = Gpu(0, object(), "GPU-a")

    first = collector._duty_cycle(gpu)
    second = collector._duty_cycle(gpu)

    assert first == {"util_mean": pytest.approx(0.8), "util_min": 0.5, "util_samples": 3}
    assert collector._seen[0] == 30  # the next call asks for samples after this one
    assert second == {"util_mean": None, "util_min": None, "util_samples": None}


def test_xids_are_collected_per_gpu_and_an_empty_queue_is_the_normal_case():
    collector = GpuCollector()
    collector._nvml = fake_nvml(events=[
        SimpleNamespace(device=0, eventType=0x8, eventData=79),
        SimpleNamespace(device=1, eventType=0x8, eventData=48),
        SimpleNamespace(device=0, eventType=0x8, eventData=13),
        SimpleNamespace(device=0, eventType=0x1, eventData=0),  # not an XID
    ])
    collector._events = object()

    assert collector._xids() == {0: ["79", "13"], 1: ["48"]}
    assert collector._xids() == {}
