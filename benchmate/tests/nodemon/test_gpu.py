"""The GPU collector's own arithmetic, against a stub NVML: no driver needed, and it is the
part that could be wrong without crashing."""

from types import SimpleNamespace

import pynvml
import pytest

from benchmate.nodemon.collectors import GpuCollector
from benchmate.nodemon.collectors import nvml as nvml_module
from benchmate.nodemon.collectors.nvml import Gpu


def sample(timestamp, value):
    return SimpleNamespace(timeStamp=timestamp, sampleValue=SimpleNamespace(uiVal=value))


def fake_nvml(samples=(), events=()):
    """The NVML calls, plus the constants the collector reads off the same module."""
    remaining = list(events)

    def wait(event_set, timeout_ms):
        if not remaining:
            raise pynvml.NVMLError(pynvml.NVML_ERROR_TIMEOUT)
        return remaining.pop(0)

    def get_samples(handle, kind, since):
        newer = [s for s in samples if s.timeStamp > since]
        if not newer:
            raise pynvml.NVMLError(pynvml.NVML_ERROR_NOT_FOUND)
        return (0, newer)

    return SimpleNamespace(
        NVMLError=pynvml.NVMLError,
        NVML_ERROR_NOT_FOUND=pynvml.NVML_ERROR_NOT_FOUND,
        NVML_GPU_UTILIZATION_SAMPLES=pynvml.NVML_GPU_UTILIZATION_SAMPLES,
        nvmlEventTypeXidCriticalError=pynvml.nvmlEventTypeXidCriticalError,
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
        SimpleNamespace(device=0, eventType=pynvml.nvmlEventTypeXidCriticalError, eventData=79),
        SimpleNamespace(device=1, eventType=pynvml.nvmlEventTypeXidCriticalError, eventData=48),
        SimpleNamespace(device=0, eventType=pynvml.nvmlEventTypeXidCriticalError, eventData=13),
        SimpleNamespace(device=0, eventType=0x1, eventData=0),  # not an XID
    ])
    collector._events = object()

    assert collector._xids() == {0: ["79", "13"], 1: ["48"]}
    assert collector._xids() == {}


def _one_gpu(monkeypatch):
    """NVML answers one GPU and counts init and shutdown, without a driver."""
    calls = {"init": 0, "shutdown": 0}

    def nvml_init():
        calls["init"] += 1

    def nvml_shutdown():
        calls["shutdown"] += 1

    def no_events():
        raise pynvml.NVMLError(pynvml.NVML_ERROR_NOT_SUPPORTED)

    monkeypatch.setattr(nvml_module.pynvml, "nvmlInit", nvml_init)
    monkeypatch.setattr(nvml_module.pynvml, "nvmlShutdown", nvml_shutdown)
    monkeypatch.setattr(nvml_module.pynvml, "nvmlDeviceGetCount", lambda: 1)
    monkeypatch.setattr(nvml_module.pynvml, "nvmlDeviceGetHandleByIndex", lambda index: object())
    monkeypatch.setattr(nvml_module.pynvml, "nvmlDeviceGetUUID", lambda handle: "GPU-a")
    monkeypatch.setattr(nvml_module.pynvml, "nvmlEventSetCreate", no_events)
    return calls


def test_nvml_starts_once_and_the_last_close_stops_it(monkeypatch):
    from benchmate.nodemon.registry import CollectorRegistry

    class OtherGpu(GpuCollector):
        name = "other"

    calls = _one_gpu(monkeypatch)
    registry = CollectorRegistry()
    registry.register(GpuCollector(), 1.0)
    registry.register(OtherGpu(), 1.0)

    registry.start(now=0.0, wall=0.0)
    assert calls == {"init": 1, "shutdown": 0}
    registry.close()
    assert calls["shutdown"] == 1

    nvml_module.pynvml.nvmlInit()  # the session ended, so this starts NVML again
    assert calls["init"] == 2
    nvml_module.pynvml.nvmlShutdown()


def test_a_refused_nvml_init_is_not_asked_again(monkeypatch):
    from benchmate.nodemon.registry import CollectorRegistry

    class OtherGpu(GpuCollector):
        name = "other"

    calls = {"init": 0, "shutdown": 0}

    def nvml_init():
        calls["init"] += 1
        raise pynvml.NVMLError(pynvml.NVML_ERROR_UNKNOWN)

    def nvml_shutdown():
        calls["shutdown"] += 1

    def no_devices():
        raise pynvml.NVMLError(pynvml.NVML_ERROR_UNINITIALIZED)

    monkeypatch.setattr(nvml_module.pynvml, "nvmlInit", nvml_init)
    monkeypatch.setattr(nvml_module.pynvml, "nvmlShutdown", nvml_shutdown)
    monkeypatch.setattr(nvml_module.pynvml, "nvmlDeviceGetCount", no_devices)
    registry = CollectorRegistry()
    registry.register(GpuCollector(), 1.0)
    registry.register(OtherGpu(), 1.0)

    registry.start(now=0.0, wall=0.0)
    registry.close()

    assert calls == {"init": 1, "shutdown": 0}
