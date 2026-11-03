"""The GPM collector against a stub NVML: the sample pairing and the unit conversion, which are
the parts that could be wrong without crashing."""

from types import SimpleNamespace

import pynvml

from benchmate.nodemon.collectors import GpmCollector
from benchmate.nodemon.collectors.nvml import Gpu


class FakeMetricsGet:
    """Stands in for the ctypes request struct: the fields the collector writes and reads."""

    def __init__(self):
        self.version = 0
        self.numMetrics = 0
        self.sample1 = None
        self.sample2 = None
        self.metrics = [
            SimpleNamespace(metricId=0, nvmlReturn=pynvml.NVML_SUCCESS, value=0.0) for _ in range(32)
        ]


def _metric_id(constant):
    """The id the installed bindings give this counter, None when they do not name it."""
    return getattr(pynvml, constant, None)


def fake_nvml(percentages, freed, pairs):
    """The GPM calls, plus the constants the collector reads off the same module."""

    def metrics_get(request):
        pairs.append((request.sample1, request.sample2))
        for metric in request.metrics[: request.numMetrics]:
            metric.nvmlReturn = pynvml.NVML_SUCCESS
            metric.value = percentages[metric.metricId]
        return request

    stub = SimpleNamespace(
        NVMLError=pynvml.NVMLError,
        NVML_SUCCESS=pynvml.NVML_SUCCESS,
        NVML_GPM_METRICS_GET_VERSION=pynvml.NVML_GPM_METRICS_GET_VERSION,
        c_nvmlGpmMetricsGet_t=FakeMetricsGet,
        nvmlGpmQueryDeviceSupport=lambda handle: SimpleNamespace(isSupportedDevice=1),
        nvmlGpmSampleAlloc=lambda: object(),
        nvmlGpmSampleGet=lambda handle, buffer: buffer,
        nvmlGpmSampleFree=freed.append,
        nvmlGpmMetricsGet=metrics_get,
        nvmlShutdown=lambda: None,
    )
    for constant in GpmCollector.metric_names.values():
        metric_id = _metric_id(constant)
        if metric_id is not None:
            setattr(stub, constant, metric_id)
    return stub


def reported(by_column):
    """NVML's raw numbers keyed by metric id. A column the bindings do not name is absent."""
    values = {}
    for column, constant in GpmCollector.metric_names.items():
        metric_id = _metric_id(constant)
        if metric_id is None:
            continue
        values[metric_id] = by_column.get(column, 0.0)
    return values


def expected(by_column, *, empty=False):
    """The row those raw numbers become: percentages as fractions, MiB/s left alone."""
    row = {"gpu": 0}
    for column, constant in GpmCollector.metric_names.items():
        if empty or _metric_id(constant) is None:
            row[column] = None
            continue
        raw = by_column.get(column, 0.0)
        row[column] = raw if column in GpmCollector.rates else raw / 100
    return row


def one_gpu(by_column=None):
    """A collector holding one supported GPU, with the buffers NVML freed and the pairs it was asked about."""
    freed = []
    pairs = []
    collector = GpmCollector()
    collector._nvml = fake_nvml(reported(by_column or {}), freed, pairs)
    gpu = Gpu(0, object(), "GPU-a")
    assert collector._init()
    assert collector._open(gpu)
    collector._gpus = [gpu]
    return collector, freed, pairs


def test_the_declared_columns_are_the_ones_sampled():
    assert GpmCollector.columns[1:] == tuple(GpmCollector.metric_names)


def test_the_first_tick_has_nothing_to_compare_and_the_next_reports_fractions():
    readings = {
        "sm_util": 80.0,
        "sm_occupancy": 45.5,
        "tensor_util": 60.0,
        "dram_bw_util": 12.0,
        "fp32_util": 25.0,
        "hmma_tensor_util": 40.0,
        "pcie_tx_mib_s": 12.5,
    }
    collector, _, _ = one_gpu(readings)

    first = collector.sample(0.0)
    second = collector.sample(1.0)

    assert first == [expected({}, empty=True)]
    assert second == [expected(readings)]
    assert second[0]["sm_util"] == 0.8
    assert second[0]["pcie_tx_mib_s"] == 12.5


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
            raise pynvml.NVMLError(pynvml.NVML_ERROR_UNKNOWN)
        given.append(object())
        return given[-1]

    collector._nvml.nvmlGpmSampleAlloc = alloc_once
    assert collector._init()

    assert not collector._open(Gpu(0, object(), "GPU-a"))
    assert freed == given  # the one NVML did give back is handed back


def test_a_constant_the_bindings_omit_leaves_that_cell_empty(monkeypatch):
    monkeypatch.delattr(pynvml, "NVML_GPM_METRIC_FP32_UTIL")
    collector, _, _ = one_gpu({"fp32_util": 50.0, "sm_util": 80.0, "pcie_tx_mib_s": 12.5})

    collector.sample(0.0)
    row = collector.sample(1.0)[0]

    assert "fp32_util" not in collector._metric_id
    assert row["fp32_util"] is None
    assert row["sm_util"] == 0.8
    assert row["pcie_tx_mib_s"] == 12.5


def test_one_missing_counter_does_not_turn_the_collector_off(monkeypatch):
    monkeypatch.delattr(pynvml, "NVML_GPM_METRIC_SM_UTIL")
    collector, _, _ = one_gpu({"sm_occupancy": 45.5})

    collector.sample(0.0)
    row = collector.sample(1.0)[0]

    assert "sm_util" not in collector._metric_id
    assert row["sm_util"] is None
    assert row["sm_occupancy"] == 0.455


def test_bindings_that_name_no_counter_do_not_collect(monkeypatch, caplog):
    for constant in GpmCollector.metric_names.values():
        monkeypatch.delattr(pynvml, constant, raising=False)
    collector = GpmCollector()
    collector._nvml = fake_nvml({}, [], [])

    with caplog.at_level("DEBUG"):
        assert not collector._init()

    assert any(
        record.levelname == "DEBUG" and "none of the counters" in record.message
        for record in caplog.records
    )
