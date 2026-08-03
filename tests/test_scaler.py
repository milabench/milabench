import pytest
import yaml

from milabench.sizer import Sizer, SizerOptions, sizer_global


def test_scaler_use_override(multipack, config):
    sizer = Sizer(SizerOptions(batch_size=64, auto=False), config("scaling"))
    for k, pack in multipack.packs.items():
        assert sizer.size(pack, "48Go") == 64


def test_scaler_use_optimized(multipack, config):
    sizer = Sizer(
        SizerOptions(
            batch_size=None,
            auto=False,
            optimized=True,
        ),
        config("scaling"),
    )
    for k, pack in multipack.packs.items():
        assert sizer.size(pack, "48Go") == 138


_values = [
    ("5Go", 27),  # Not a multiple of 8
    ("6Go", 32),
    ("12Go", 64),
    ("18Go", 96),
    ("24Go", 128),
    ("30Go", 160),
    ("48Go", 256),
    ("72Go", 384),
]


@pytest.mark.parametrize("capacity,expected", _values)
def test_scaler_autoscaler_lerp(multipack, config, capacity, expected):
    sizer = Sizer(SizerOptions(batch_size=None, auto=True, multiple=None), config("scaling"))
    for k, pack in multipack.packs.items():
        assert sizer.size(pack, capacity) == expected


_values_2 = [
    ("5Go", 24),  # a multiple of 8
    ("6Go", 32),
]


@pytest.mark.parametrize("capacity,expected", _values_2)
def test_scaler_autoscaler_lerp_multiple(multipack, config, capacity, expected):
    sizer = Sizer(
        SizerOptions(
            batch_size=None,
            auto=True,
            multiple=8,
        ),
        config("scaling"),
    )
    for k, pack in multipack.packs.items():
        assert sizer.size(pack, capacity) == expected


def test_scaler_disabled(multipack):
    for k, pack in multipack.packs.items():
        assert pack.argv == ["--batch_size", "auto_batch(12)"]


def fakeexec(pack):
    from milabench.sizer import resolve_argv

    final_args = resolve_argv(pack, pack.argv)
    return final_args


def test_scaler_enabled(multipack, config):
    from milabench.system import system_global
    from milabench.system import apply_system

    conf = {
        "gpu": {
            "capacity": "41920 MiB"
        },
        "sizer": {
            "multiple": 8
        },
    }

    for k, pack in multipack.packs.items():
        assert fakeexec(pack) == ["--batch_size", "12"]

    with apply_system(conf):
        for k, pack in multipack.packs.items():
            result = fakeexec(pack)
            assert result[0] == "--batch_size"
            assert int(result[1]) >= 12

    for k, pack in multipack.packs.items():
        assert fakeexec(pack) == ["--batch_size", "12"]


def _write_scaling(tmp_path, observations, name="bench"):
    path = tmp_path / "scaling_profile.yaml"
    path.write_text(yaml.safe_dump({name: {"observations": observations}}))
    return path


def _obs(batch_size, memory_mib, torchmem_mib=None, perf=100.0):
    row = {
        "batch_size": batch_size,
        "cpu": 8,
        "memory": f"{memory_mib} MiB",
        "perf": perf,
        "time": 0,
    }
    if torchmem_mib is not None:
        row["torchmem"] = f"{torchmem_mib} MiB"
    return row


class TestAutoSizeFixedTorchmem:
    """fixed = memory−torchmem; torchmem ≈ α·BS + β; fill NVML capacity."""

    def test_predicts_from_fixed_plus_torchmem(self, tmp_path):
        # fixed=2000 MiB, torchmem=100 MiB * BS  →  B* = (C - 2000) / 100
        observations = [
            _obs(16, 3600, 1600),
            _obs(32, 5200, 3200),
            _obs(64, 8400, 6400),
        ]
        path = _write_scaling(tmp_path, observations)
        sizer = Sizer(
            SizerOptions(batch_size=None, auto=True, multiple=None),
            path,
        )
        # C = 10000 MiB → B* = 80
        assert sizer.size("bench", "10000MiB") == 80

    def test_applies_multiple(self, tmp_path):
        observations = [
            _obs(16, 3600, 1600),
            _obs(32, 5200, 3200),
            _obs(64, 8400, 6400),
        ]
        path = _write_scaling(tmp_path, observations)
        sizer = Sizer(
            SizerOptions(batch_size=None, auto=True, multiple=32),
            path,
        )
        # 80 → floored to multiple of 32 → 64
        assert sizer.size("bench", "10000MiB") == 64

    def test_fallback_without_torchmem(self, tmp_path):
        # Legacy: batch_size ≈ a·memory + b using NVML only (MiB throughout)
        observations = [
            _obs(64, 12000),
            _obs(128, 24000),
            _obs(256, 48000),
        ]
        path = _write_scaling(tmp_path, observations)
        sizer = Sizer(
            SizerOptions(batch_size=None, auto=True, multiple=None),
            path,
        )
        assert sizer.size("bench", "48000MiB") == 256

    def test_fallback_when_torchmem_slope_non_positive(self, tmp_path):
        observations = [
            _obs(16, 5000, 4000, perf=50),
            _obs(32, 4500, 3000, perf=40),
            _obs(64, 4000, 2000, perf=30),
        ]
        path = _write_scaling(tmp_path, observations)
        sizer = Sizer(
            SizerOptions(batch_size=None, auto=True, multiple=None),
            path,
        )
        # Falls back to legacy fit on observation_memory (torchmem preferred).
        assert sizer.size("bench", "3500MiB") >= 1

    def test_optimized_prefers_nvml_memory(self, tmp_path):
        # torchmem fits capacity but NVML does not for the faster row
        observations = [
            _obs(128, 9000, 2000, perf=200),  # nvml too big
            _obs(64, 5000, 1500, perf=100),   # nvml fits
        ]
        path = _write_scaling(tmp_path, observations)
        sizer = Sizer(
            SizerOptions(batch_size=None, auto=False, optimized=True),
            path,
        )
        assert sizer.size("bench", "6000MiB") == 64
