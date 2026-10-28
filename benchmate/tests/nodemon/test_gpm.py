"""The GPM collector against a stub NVML: the sample pairing and the unit conversion, which are
the parts that could be wrong without crashing."""

from types import SimpleNamespace

from benchmate.nodemon.collectors import GpmCollector
from benchmate.nodemon.collectors.nvml import Gpu


class FakeNvmlError(Exception):
    def __init__(self, value=0):
        self.value = value


class FakeMetricsGet:
    """Stands in for the ctypes request struct: the fields the collector writes and reads."""

    def __init__(self):
        self.version = 0
        self.numMetrics = 0
        self.sample1 = None
        self.sample2 = None
        self.metrics = [SimpleNamespace(metricId=0, nvmlReturn=0, value=0.0) for _ in range(16)]


def fake_nvml(percentages, freed, pairs):
    """Just the GPM surface the collector touches; `percentages` is what NVML reports per metric id."""

    def metrics_get(request):
        pairs.append((request.sample1, request.sample2))
        for metric in request.metrics[: request.numMetrics]:
            metric.value = percentages[metric.metricId]
        return request

    return SimpleNamespace(
        NVMLError=FakeNvmlError,
        NVML_SUCCESS=0,
        NVML_ERROR_NOT_SUPPORTED=3,
        NVML_GPM_METRIC_SM_UTIL=2,
        NVML_GPM_METRIC_SM_OCCUPANCY=3,
        NVML_GPM_METRIC_ANY_TENSOR_UTIL=5,
        NVML_GPM_METRIC_DRAM_BW_UTIL=10,
        NVML_GPM_METRICS_GET_VERSION=1,
        c_nvmlGpmMetricsGet_t=FakeMetricsGet,
        nvmlGpmQueryDeviceSupport=lambda handle: SimpleNamespace(isSupportedDevice=1),
        nvmlGpmSampleAlloc=lambda: object(),
        nvmlGpmSampleGet=lambda handle, buffer: buffer,
        nvmlGpmSampleFree=freed.append,
        nvmlGpmMetricsGet=metrics_get,
        nvmlShutdown=lambda: None,
    )


def one_gpu(percentages=None):
    """A collector holding one supported GPU, with the buffers NVML freed and the pairs it was asked about."""
    freed = []
    pairs = []
    collector = GpmCollector()
    collector._nvml = fake_nvml(percentages or {2: 0.0, 3: 0.0, 5: 0.0, 10: 0.0}, freed, pairs)
    gpu = Gpu(0, object(), "GPU-a")
    assert collector._init()
    assert collector._open(gpu)
    collector._gpus = [gpu]
    return collector, freed, pairs


def test_the_first_tick_has_nothing_to_compare_and_the_next_reports_fractions():
    collector, _, _ = one_gpu({2: 80.0, 3: 45.5, 5: 60.0, 10: 12.0})

    first = collector.sample(0.0)
    second = collector.sample(1.0)

    assert first == [{"gpu": 0, "sm_util": None, "sm_occupancy": None,
                      "tensor_util": None, "dram_bw_util": None}]
    assert second == [{"gpu": 0, "sm_util": 0.8, "sm_occupancy": 0.455,
                       "tensor_util": 0.6, "dram_bw_util": 0.12}]


def test_each_tick_is_compared_with_the_one_before_it_not_with_itself():
    collector, _, pairs = one_gpu()
    first, second = collector._pairs[0].buffers

    collector.sample(0.0)
    collector.sample(1.0)
    collector.sample(2.0)

    # The third tick reused the first buffer, which the second tick's sample had to survive.
    assert pairs == [(first, second), (second, first)]


def test_close_gives_every_sample_buffer_back_to_nvml():
    collector, freed, _ = one_gpu()
    buffers = list(collector._pairs[0].buffers)

    collector.close()

    assert freed == buffers
    assert collector._pairs == {}


def test_a_gpu_without_gpm_support_is_not_kept():
    collector, _, _ = one_gpu()
    collector._nvml.nvmlGpmQueryDeviceSupport = lambda handle: SimpleNamespace(isSupportedDevice=0)

    assert not collector._open(Gpu(1, object(), "GPU-b"))
    assert 1 not in collector._pairs


def test_a_gpu_whose_second_buffer_nvml_refuses_gives_the_first_one_back():
    freed = []
    collector = GpmCollector()
    collector._nvml = fake_nvml({}, freed, [])
    given = []

    def alloc_once():
        if given:
            raise FakeNvmlError()
        given.append(object())
        return given[-1]

    collector._nvml.nvmlGpmSampleAlloc = alloc_once
    assert collector._init()

    assert not collector._open(Gpu(0, object(), "GPU-a"))
    assert freed == given  # the one NVML did give back is handed back
