"""Tests for pure-logic functions in milabench.sizer."""

import time
from collections import defaultdict
from unittest.mock import MagicMock

import pytest
import yaml
from cantilever.core.statstream import StatStream

from milabench.sizer import (
    BenchStats,
    MemoryUsageExtractor,
    Sizer,
    SizerOptions,
    arch_to_device,
    broadcast,
    compact_dump,
    deduplicate_observation,
    observation_memory,
    to_octet,
    _fixed_overhead,
    _fit_torchmem_vs_batch,
    _obs_pair_octets,
<<<<<<< HEAD
    _torch_backend_info,
=======
    _per_gpu_memory_mib,
>>>>>>> b3c6727 (New torchtitan benchmark concept)
)


# ---------------------------------------------------------------------------
# to_octet: metric-prefix byte string → float
# ---------------------------------------------------------------------------

class TestToOctet:
    """Tests for to_octet(), the byte-string parser."""

    @pytest.mark.parametrize(
        "value, expected",
        [
            ("1 GiB", 1 * 1024**3),
            ("2 GiB", 2 * 1024**3),
            ("0.5 GiB", 0.5 * 1024**3),
        ],
    )
    def test_gibibytes(self, value, expected):
        # to_octet doesn't handle spaces — strip them as the callers do
        assert to_octet(value.replace(" ", "")) == expected

    @pytest.mark.parametrize(
        "value, expected",
        [
            ("1GB", 1 * 10**9),
            ("2GB", 2 * 10**9),
            ("0.5GB", 0.5 * 10**9),
        ],
    )
    def test_gigabytes_si(self, value, expected):
        assert to_octet(value) == expected

    @pytest.mark.parametrize(
        "value, expected",
        [
            ("12Go", 12 * 10**9),
            ("48Go", 48 * 10**9),
        ],
    )
    def test_gigaoctets(self, value, expected):
        assert to_octet(value) == expected

    @pytest.mark.parametrize(
        "value, expected",
        [
            ("1024MiB", 1024 * 1024**2),
            ("41920MiB", 41920 * 1024**2),
            ("512MiB", 512 * 1024**2),
        ],
    )
    def test_mebibytes(self, value, expected):
        assert to_octet(value) == expected

    @pytest.mark.parametrize(
        "value, expected",
        [
            ("500MB", 500 * 10**6),
            ("1MB", 1 * 10**6),
        ],
    )
    def test_megabytes_si(self, value, expected):
        assert to_octet(value) == expected

    def test_terabytes(self):
        assert to_octet("1TB") == 10**12

    def test_tebibytes(self):
        assert to_octet("1TiB") == 1024**4

    def test_kilobytes(self):
        assert to_octet("4kB") == 4 * 10**3

    def test_kibibytes(self):
        assert to_octet("4kiB") == 4 * 1024**1

    def test_plain_octets_with_suffix(self):
        assert to_octet("1000o") == 1000.0

    def test_plain_ioctets(self):
        assert to_octet("2048io") == 2048.0

    def test_bare_number(self):
        assert to_octet("42") == 42.0

    def test_zero(self):
        assert to_octet("0") == 0.0


# ---------------------------------------------------------------------------
# arch_to_device: arch string → device type
# ---------------------------------------------------------------------------

class TestArchToDevice:
    @pytest.mark.parametrize(
        "arch, expected",
        [
            ("cuda", "cuda"),
            ("cpu", "cpu"),
            ("rocm", "cuda"),
            ("xpu", "xpu"),
            ("hpu", "hpu"),
            ("mps", "mps"),
            ("meta", "meta"),
        ],
    )
    def test_known_architectures(self, arch, expected):
        assert arch_to_device(arch) == expected

    def test_unknown_arch_defaults_to_cpu(self):
        assert arch_to_device("nonexistent_arch") == "cpu"

    def test_empty_string_defaults_to_cpu(self):
        assert arch_to_device("") == "cpu"


# ---------------------------------------------------------------------------
# broadcast: call a list of delegates with error resilience
# ---------------------------------------------------------------------------

class TestBroadcast:
    def test_calls_all_delegates(self):
        results = []
        broadcast([lambda x: results.append(x), lambda x: results.append(x * 2)], 5)
        assert results == [5, 10]

    def test_empty_delegates(self):
        broadcast([], 1, 2, 3)

    def test_error_in_one_does_not_stop_others(self, capsys):
        results = []

        def bad(_):
            raise ValueError("boom")

        def good(x):
            results.append(x)

        broadcast([bad, good], 42)
        assert results == [42]
        captured = capsys.readouterr()
        assert "boom" in captured.out

    def test_passes_kwargs(self):
        received = {}

        def capture(**kw):
            received.update(kw)

        broadcast([capture], key="val")
        assert received == {"key": "val"}


# ---------------------------------------------------------------------------
# BenchStats: dataclass helpers
# ---------------------------------------------------------------------------

class TestBenchStats:
    def test_initial_state(self):
        bs = BenchStats("test_bench")
        assert bs.benchname == "test_bench"
        assert bs.active_count == 0
        assert bs.rc == []
        assert bs.early_stopped == []

    def test_max_memory_usage_prefers_torchmem(self):
        bs = BenchStats("b")
        bs.torchmem_usage += 500
        bs.max_usage += 100
        assert bs.max_memory_usage() == 500

    def test_max_memory_usage_falls_back_to_max_usage(self):
        bs = BenchStats("b")
        bs.max_usage += 300
        bs.max_usage += 400
        assert bs.max_memory_usage() == 400

    def test_max_memory_usage_no_data(self):
        bs = BenchStats("b")
        assert bs.max_memory_usage() == float("-inf")

    def test_has_stopped_early_false_when_empty(self):
        bs = BenchStats("b")
        assert bs.has_stopped_early() is False

    def test_has_stopped_early_true(self):
        bs = BenchStats("b")
        bs.early_stopped.append(True)
        assert bs.has_stopped_early() is True

    def test_has_stopped_early_last_false(self):
        bs = BenchStats("b")
        bs.early_stopped.append(True)
        bs.early_stopped.append(False)
        assert bs.has_stopped_early() is False

    def test_statstream_fields_accumulate(self):
        bs = BenchStats("b")
        bs.perf += 10.0
        bs.perf += 20.0
        assert bs.perf.avg == 15.0
        assert bs.perf.current_count == 2


class TestPerGpuMemoryMib:
    def test_none_on_empty(self):
        assert _per_gpu_memory_mib(None) is None
        assert _per_gpu_memory_mib({}) is None

    def test_single_device(self):
        assert _per_gpu_memory_mib({"0": {"memory": [31240, 192000]}}) == 31240

    def test_multi_device_is_peak_not_sum(self):
        gpudata = {
            str(i): {"memory": [31240 + i, 192000]} for i in range(8)
        }
        # Sum would be ~249960; we keep the per-GPU peak.
        assert _per_gpu_memory_mib(gpudata) == 31247
        assert _per_gpu_memory_mib(gpudata) != sum(
            d["memory"][0] for d in gpudata.values()
        )

    def test_min_load_skips_idle_ddp_init_spike(self):
        # Real spike shape from run.out: ~249k on every GPU at load ~0.1
        spike = {
            str(i): {"memory": [246000 + i * 100, 294896], "load": 0.1}
            for i in range(8)
        }
        assert _per_gpu_memory_mib(spike, min_load=0.3) is None

        steady = {
            str(i): {"memory": [32000 + i, 294896], "load": 0.95}
            for i in range(8)
        }
        assert _per_gpu_memory_mib(steady, min_load=0.3) == 32007


class TestMemoryUsageExtractorPerGpu:
    def test_gpudata_records_per_gpu_peak_not_cluster_sum(self, tmp_path, monkeypatch):
        save = tmp_path / "scaling.yaml"
        save.write_text("version: 2.0\n")

        opts = MagicMock()
        opts.save = str(save)
        opts.config = str(save)
        monkeypatch.setattr(SizerOptions, "instance", staticmethod(lambda: opts))

        layer = MemoryUsageExtractor()
        layer.filepath = str(save)

        pack = MagicMock()
        pack.config = {"name": "resnet152-ddp-gpus"}

        gpudata = {
            str(i): {"memory": [31240, 192000], "load": 0.95} for i in range(8)
        }
        entry = MagicMock()
        entry.pack = pack
        entry.data = {"gpudata": gpudata}

        layer.on_start(entry)
        layer.on_data(entry)

        # Idle multi-GPU SMI spike must not raise the peak.
        spike = {
            str(i): {"memory": [249924, 294896], "load": 0.08} for i in range(8)
        }
        entry.data = {"gpudata": spike}
        layer.on_data(entry)
        layer.on_batch_size_set(pack, None, 256)
        layer.on_cpu_count_set(pack, None, 8)
        entry.data = {"rate": 13000.0}
        layer.on_data(entry)
        entry.data = {"return_code": 0}
        layer.on_end(entry)

        stats = layer._benchstat["resnet152-ddp-gpus"]
        assert stats.max_usage.current_count == 1
        assert int(stats.max_usage.max) == 31240
        # Cluster sum of one poll — must not be what we store.
        assert int(stats.max_usage.max) != 31240 * 8

        obs = layer.memory["resnet152-ddp-gpus"]["observations"][-1]
        assert obs["memory"] == "31240 MiB"

    def test_has_stopped_early_false_when_empty(self):
        bs = BenchStats("b")
        assert bs.has_stopped_early() is False

    def test_has_stopped_early_true(self):
        bs = BenchStats("b")
        bs.early_stopped.append(True)
        assert bs.has_stopped_early() is True

    def test_has_stopped_early_last_false(self):
        bs = BenchStats("b")
        bs.early_stopped.append(True)
        bs.early_stopped.append(False)
        assert bs.has_stopped_early() is False

    def test_statstream_fields_accumulate(self):
        bs = BenchStats("b")
        bs.perf += 10.0
        bs.perf += 20.0
        assert bs.perf.avg == 15.0
        assert bs.perf.current_count == 2


# ---------------------------------------------------------------------------
# deduplicate_observation: the core deduplication logic (0% coverage)
# ---------------------------------------------------------------------------

def _obs(batch_size, cpu, memory_mib, perf, t=None):
    """Helper to build an observation dict."""
    return {
        "batch_size": batch_size,
        "cpu": cpu,
        "memory": f"{memory_mib} MiB",
        "perf": perf,
        "time": t or int(time.time()),
    }


class TestDeduplicateObservation:

    def test_empty_input(self):
        assert deduplicate_observation({}) == {}

    def test_version_key_preserved(self):
        result = deduplicate_observation({"version": 2.0})
        assert result["version"] == 2.0

    def test_bench_with_no_observations(self):
        scaling = {"mybench": {"observations": []}}
        result = deduplicate_observation(scaling)
        assert result["mybench"]["observations"] == []

    def test_single_observation_kept(self):
        obs = _obs(32, 4, 1000, 100.0)
        scaling = {"bench": {"observations": [obs]}}
        result = deduplicate_observation(scaling)
        assert len(result["bench"]["observations"]) == 1
        assert result["bench"]["observations"][0]["batch_size"] == 32

    def test_single_observation_zero_perf_dropped(self):
        obs = _obs(32, 4, 1000, 0)
        scaling = {"bench": {"observations": [obs]}}
        result = deduplicate_observation(scaling)
        assert result["bench"]["observations"] == []

    def test_unique_observations_all_kept(self):
        scaling = {
            "bench": {
                "observations": [
                    _obs(16, 4, 500, 50.0),
                    _obs(32, 4, 1000, 100.0),
                    _obs(64, 8, 2000, 200.0),
                ]
            }
        }
        result = deduplicate_observation(scaling)
        sizes = [o["batch_size"] for o in result["bench"]["observations"]]
        assert sorted(sizes) == [16, 32, 64]

    def test_duplicates_merged_when_similar(self):
        t = int(time.time())
        scaling = {
            "bench": {
                "observations": [
                    _obs(32, 4, 1000, 100.0, t),
                    _obs(32, 4, 1005, 101.0, t + 10),
                ]
            }
        }
        result = deduplicate_observation(scaling)
        obs = result["bench"]["observations"]
        assert len(obs) == 1
        assert obs[0]["batch_size"] == 32
        assert obs[0]["time"] == t + 10

    def test_duplicates_not_merged_when_very_different(self):
        t = int(time.time())
        scaling = {
            "bench": {
                "observations": [
                    _obs(32, 4, 1000, 100.0, t),
                    _obs(32, 4, 5000, 500.0, t + 10),
                ]
            }
        }
        result = deduplicate_observation(scaling)
        obs = result["bench"]["observations"]
        assert len(obs) == 2

    def test_zero_perf_entries_excluded_from_merge(self):
        t = int(time.time())
        scaling = {
            "bench": {
                "observations": [
                    _obs(32, 4, 1000, 100.0, t),
                    _obs(32, 4, 1000, 0, t + 1),
                ]
            }
        }
        result = deduplicate_observation(scaling)
        obs = result["bench"]["observations"]
        # Only the perf>0 entry remains (single valid → should_generate_single)
        assert len(obs) == 1
        assert obs[0]["perf"] > 0

    def test_all_duplicate_entries_zero_perf_dropped(self):
        t = int(time.time())
        scaling = {
            "bench": {
                "observations": [
                    _obs(32, 4, 1000, 0, t),
                    _obs(32, 4, 1000, 0, t + 1),
                ]
            }
        }
        result = deduplicate_observation(scaling)
        assert result["bench"]["observations"] == []

    def test_output_sorted_by_batch_size(self):
        scaling = {
            "bench": {
                "observations": [
                    _obs(64, 4, 2000, 200.0),
                    _obs(16, 4, 500, 50.0),
                    _obs(32, 4, 1000, 100.0),
                ]
            }
        }
        result = deduplicate_observation(scaling)
        sizes = [o["batch_size"] for o in result["bench"]["observations"]]
        assert sizes == sorted(sizes)

    def test_multiple_benchmarks_handled_independently(self):
        scaling = {
            "version": 2.0,
            "bench_a": {"observations": [_obs(32, 4, 1000, 100.0)]},
            "bench_b": {"observations": [_obs(64, 8, 2000, 200.0)]},
        }
        result = deduplicate_observation(scaling)
        assert "bench_a" in result
        assert "bench_b" in result
        assert result["version"] == 2.0

    def test_three_similar_duplicates_merged(self):
        t = int(time.time())
        scaling = {
            "bench": {
                "observations": [
                    _obs(32, 4, 1000, 100.0, t),
                    _obs(32, 4, 1002, 100.5, t + 5),
                    _obs(32, 4, 1001, 100.2, t + 10),
                ]
            }
        }
        result = deduplicate_observation(scaling)
        obs = result["bench"]["observations"]
        assert len(obs) == 1
        assert obs[0]["batch_size"] == 32
        assert obs[0]["time"] == t + 10

    def test_mixed_unique_and_duplicate(self):
        t = int(time.time())
        scaling = {
            "bench": {
                "observations": [
                    _obs(16, 4, 500, 50.0, t),
                    _obs(32, 4, 1000, 100.0, t),
                    _obs(32, 4, 1003, 101.0, t + 10),
                    _obs(64, 8, 2000, 200.0, t),
                ]
            }
        }
        result = deduplicate_observation(scaling)
        obs = result["bench"]["observations"]
        sizes = [o["batch_size"] for o in obs]
        assert sorted(sizes) == [16, 32, 64]

    def test_merged_observation_memory_format(self):
        t = int(time.time())
        scaling = {
            "bench": {
                "observations": [
                    _obs(32, 4, 1000, 100.0, t),
                    _obs(32, 4, 1010, 101.0, t + 10),
                ]
            }
        }
        result = deduplicate_observation(scaling)
        obs = result["bench"]["observations"][0]
        assert "MiB" in obs["memory"]
        assert isinstance(obs["perf"], float)


# ---------------------------------------------------------------------------
# compact_dump: YAML formatting
# ---------------------------------------------------------------------------

class TestCompactDump:
    def test_returns_dumper_class(self):
        dumper = compact_dump()
        assert issubclass(dumper, yaml.SafeDumper)

    def test_roundtrip_preserves_data(self):
        data = {
            "bench": {
                "observations": [
                    {"batch_size": 32, "memory": "1000 MiB", "perf": 100.0},
                    {"batch_size": 64, "memory": "2000 MiB", "perf": 200.0},
                ]
            }
        }
        dumped = yaml.dump(data, Dumper=compact_dump())
        reloaded = yaml.safe_load(dumped)
        assert reloaded == data


# ---------------------------------------------------------------------------
# fixed + torchmem helpers
# ---------------------------------------------------------------------------

def _pair_obs(batch_size, memory_mib, torchmem_mib, perf=100.0):
    return {
        "batch_size": batch_size,
        "cpu": 8,
        "memory": f"{memory_mib} MiB",
        "torchmem": f"{torchmem_mib} MiB",
        "perf": perf,
        "time": 0,
    }


class TestObsPairOctets:
    def test_returns_both(self):
        pair = _obs_pair_octets(_pair_obs(32, 4000, 2500))
        assert pair is not None
        mem, alloc = pair
        assert mem == to_octet("4000MiB")
        assert alloc == to_octet("2500MiB")

    def test_missing_torchmem(self):
        assert _obs_pair_octets({"memory": "1000 MiB", "batch_size": 1}) is None

    def test_zero_torchmem_treated_as_missing(self):
        assert _obs_pair_octets(_pair_obs(32, 4000, 0)) is None

    def test_jaxmem_fallback(self):
        obs = {
            "memory": "3000 MiB",
            "jaxmem": "2000 MiB",
            "batch_size": 8,
        }
        pair = _obs_pair_octets(obs)
        assert pair is not None
        assert pair[1] == to_octet("2000MiB")


class TestFixedOverhead:
    def test_median_gap(self):
        obs = [
            _pair_obs(16, 3600, 1600),  # gap 2000
            _pair_obs(32, 5200, 3200),  # gap 2000
            _pair_obs(64, 8400, 6400),  # gap 2000
        ]
        fixed = _fixed_overhead(obs)
        assert fixed == pytest.approx(to_octet("2000MiB"))

    def test_clamps_negative_gap(self):
        obs = [_pair_obs(16, 1000, 1500)]  # torchmem > nvml
        assert _fixed_overhead(obs) == 0.0

    def test_none_without_pairs(self):
        assert _fixed_overhead([{"batch_size": 1, "memory": "1000 MiB"}]) is None


class TestFitTorchmemVsBatch:
    def test_linear_through_origin(self):
        # torchmem = 100 MiB * batch
        obs = [
            _pair_obs(16, 3600, 1600),
            _pair_obs(32, 5200, 3200),
            _pair_obs(64, 8400, 6400),
        ]
        alpha, beta = _fit_torchmem_vs_batch(obs)
        assert alpha == pytest.approx(to_octet("100MiB"), rel=1e-6)
        assert beta == pytest.approx(0.0, abs=to_octet("1MiB"))

    def test_none_when_non_positive_slope(self):
        obs = [
            _pair_obs(16, 5000, 4000),
            _pair_obs(32, 4500, 3000),  # torchmem shrinks with batch
        ]
        assert _fit_torchmem_vs_batch(obs) is None

    def test_none_with_single_point(self):
        assert _fit_torchmem_vs_batch([_pair_obs(16, 3600, 1600)]) is None

    def test_none_when_all_torchmem_zero(self):
        obs = [
            _pair_obs(16, 3600, 0),
            _pair_obs(32, 5200, 0),
            _pair_obs(64, 8400, 0),
        ]
        assert _fit_torchmem_vs_batch(obs) is None


class TestObservationMemoryZeroTorchmem:
    def test_skips_zero_torchmem_uses_nvml(self):
        obs = _pair_obs(64, 8400, 0)
        assert observation_memory(obs) == to_octet("8400MiB")

    def test_prefers_positive_torchmem(self):
        obs = _pair_obs(64, 8400, 6400)
        assert observation_memory(obs) == to_octet("6400MiB")


class TestAutoSizeZeroTorchmem:
    def test_falls_back_to_nvml_fit(self, tmp_path):
        """All-zero torchmem must not crash; NVML memory still sizes the batch."""
        profile = tmp_path / "scaling.yaml"
        profile.write_text(
            yaml.dump(
                {
                    "resnet152-ddp-gpus": {
                        "observations": [
                            _pair_obs(64, 14760, 0, perf=100),
                            _pair_obs(128, 21110, 0, perf=200),
                            _pair_obs(256, 33213, 0, perf=300),
                        ]
                    }
                }
            )
        )
        sizer = Sizer(
            sizer=SizerOptions(auto=True),
            config=str(profile),
        )
        size = sizer.auto_size("resnet152-ddp-gpus", "40000 MiB")
        assert size is not None
        assert size > 1


# ---------------------------------------------------------------------------
# torch / backend version stamping on scaling observations
# ---------------------------------------------------------------------------

class TestTorchBackendInfo:
    def setup_method(self):
        _torch_backend_info.cache_clear()

    def teardown_method(self):
        _torch_backend_info.cache_clear()

    def test_cuda_backend(self, monkeypatch):
        import sys
        import types

        fake = types.ModuleType("torch")
        fake.__version__ = "2.7.0+cu128"
        fake.version = types.SimpleNamespace(cuda="12.8", hip=None)
        monkeypatch.setitem(sys.modules, "torch", fake)

        info = _torch_backend_info()
        assert info == {
            "torch": "2.7.0+cu128",
            "backend": "cuda",
            "backend_version": "12.8",
        }

    def test_rocm_backend(self, monkeypatch):
        import sys
        import types

        fake = types.ModuleType("torch")
        fake.__version__ = "2.7.0+rocm6.3"
        fake.version = types.SimpleNamespace(cuda=None, hip="6.3.42134")
        monkeypatch.setitem(sys.modules, "torch", fake)

        info = _torch_backend_info()
        assert info["backend"] == "rocm"
        assert info["backend_version"] == "6.3.42134"
        assert info["torch"] == "2.7.0+rocm6.3"

    def test_missing_torch(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def _raise(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("no torch")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _raise)
        assert _torch_backend_info() == {}


class TestPushObservationVersions:
    def setup_method(self):
        _torch_backend_info.cache_clear()

    def teardown_method(self):
        _torch_backend_info.cache_clear()

    def test_observation_includes_backend_fields(self, tmp_path, monkeypatch):
        import sys
        import types

        fake = types.ModuleType("torch")
        fake.__version__ = "2.6.0+cu124"
        fake.version = types.SimpleNamespace(cuda="12.4", hip=None)
        monkeypatch.setitem(sys.modules, "torch", fake)

        save_path = tmp_path / "out.yaml"
        monkeypatch.setattr(
            SizerOptions,
            "instance",
            classmethod(lambda cls: SizerOptions(save=str(save_path), config=None)),
        )

        extractor = MemoryUsageExtractor()
        extractor.filepath = str(save_path)
        extractor.memory = {"version": 2.0}

        stats = BenchStats("bert-fp16")
        stats.cpu += 8
        stats.batch_size += 16
        stats.perf += 100.0
        stats.max_usage += 4000

        extractor.push_observation(stats)
        obs = extractor.memory["bert-fp16"]["observations"][0]
        assert obs["torch"] == "2.6.0+cu124"
        assert obs["backend"] == "cuda"
        assert obs["backend_version"] == "12.4"
        assert "revision" not in obs


# ---------------------------------------------------------------------------
# StatStream integration (used heavily in sizer)
# ---------------------------------------------------------------------------

class TestStatStreamUsage:
    """Verify StatStream behaves as sizer.py assumes."""

    def test_basic_accumulation(self):
        s = StatStream(drop_first_obs=0)
        s += 10
        s += 20
        assert s.current_count == 2
        assert s.avg == 15.0
        assert s.max == 20
        assert s.min == 10

    def test_empty_stream_defaults(self):
        s = StatStream(drop_first_obs=0)
        assert s.current_count == 0
        assert s.max == float("-inf")

    def test_single_value(self):
        s = StatStream(drop_first_obs=0)
        s += 42
        assert s.avg == 42.0
        assert s.current_count == 1
        assert s.sd == 0.0
