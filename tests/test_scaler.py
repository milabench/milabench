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


def test_scaler_optimized_missing_benchmark(tmp_path):
    path = tmp_path / "scaling.yaml"
    path.write_text(yaml.safe_dump({"other-bench": {"observations": []}}))
    sizer = Sizer(
        SizerOptions(batch_size=None, auto=False, optimized=True),
        path,
    )
    assert sizer.optimized("resnet50", "48Go") is None


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


#
# Envelope (config/scaling/default.yaml): the lower bound of the GPUs we measured
#
def _gpu_profile(path, alpha_mib, beta_mib, batches, gpus=None, oom=None):
    """A scaling profile measured on one GPU: torchmem = alpha * B + beta, NVML 2 GiB above."""
    obs = []
    for b in batches:
        torchmem = alpha_mib * b + beta_mib
        row = {"batch_size": b, "perf": 100.0, "memory": f"{torchmem + 2048} MiB", "torchmem": f"{torchmem} MiB"}
        if gpus is not None:
            row["gpus"] = gpus
        obs.append(row)
    if oom is not None:
        row = {"batch_size": oom, "oom": True}
        if gpus is not None:
            row["gpus"] = gpus
        obs.append(row)
    path.write_text(yaml.safe_dump({"bench": {"observations": obs}}))
    return str(path)


def _envelope_sizer(tmp_path, files, **kwargs):
    from milabench.sizer import build_envelope_scaling, write_envelope_scaling

    profile, report = build_envelope_scaling(files, **kwargs)
    out = tmp_path / "default.yaml"
    write_envelope_scaling(profile, str(out), files=files, report=report)
    options = SizerOptions(batch_size=None, auto=True, multiple=8, overhead_override="10%")
    return Sizer(options, str(out)), profile


@pytest.mark.parametrize("capacity", ["40GiB", "80GiB", "180GiB", "288GiB"])
def test_envelope_never_above_any_measured_gpu(tmp_path, capacity):
    a = _gpu_profile(tmp_path / "GPUA.yaml", alpha_mib=300, beta_mib=4000, batches=[16, 64, 128])
    b = _gpu_profile(tmp_path / "GPUB.yaml", alpha_mib=360, beta_mib=2000, batches=[16, 64, 128])
    envelope, _ = _envelope_sizer(tmp_path, [a, b], cap_factor=100)

    options = SizerOptions(batch_size=None, auto=True, multiple=8, overhead_override="10%")
    bench = {"name": "bench", "devices": [0]}
    picked = envelope.auto_size(bench, capacity)
    assert picked is not None and picked > 0
    for own in (a, b):
        assert picked <= Sizer(options, own).auto_size(bench, capacity)


def test_envelope_oom_lowers_the_batch(tmp_path):
    # the linear model says bs432 fits 183 GiB, it ran out of memory: the envelope gives at most
    # the middle between the largest batch that completed (232) and the OOM one
    a = _gpu_profile(tmp_path / "B200.yaml", alpha_mib=374, beta_mib=1900, batches=[32, 232], oom=432)
    envelope, profile = _envelope_sizer(tmp_path, [a], single_gpu_margin=1.0, cap_factor=100)
    assert "alpha_raised_by" in profile["bench"]["envelope"][0]
    assert envelope.auto_size({"name": "bench", "devices": [0]}, "183GiB") <= 332


def test_envelope_matches_the_gpu_count(tmp_path):
    one = _gpu_profile(tmp_path / "GPUA.yaml", alpha_mib=100, beta_mib=1000, batches=[8, 64], gpus=8)
    other = _gpu_profile(tmp_path / "GPUB.yaml", alpha_mib=400, beta_mib=1000, batches=[8, 64], gpus=16)
    envelope, profile = _envelope_sizer(tmp_path, [one, other], single_gpu_margin=1.0, cap_factor=100)
    assert [e["gpus"] for e in profile["bench"]["envelope"]] == [8, 16]

    eight = envelope.auto_size({"name": "bench", "devices": list(range(8))}, "80GiB")
    sixteen = envelope.auto_size({"name": "bench", "devices": list(range(8)), "num_machines": 2}, "80GiB")
    assert eight > sixteen
    # no envelope for that GPU count, none without a count: no resize
    assert envelope.auto_size({"name": "bench", "devices": [0]}, "80GiB") is None


def test_envelope_caps_the_extrapolation(tmp_path):
    a = _gpu_profile(tmp_path / "GPUA.yaml", alpha_mib=1, beta_mib=1000, batches=[8, 16])
    envelope, _ = _envelope_sizer(tmp_path, [a], cap_factor=2.0)
    assert envelope.auto_size({"name": "bench", "devices": [0]}, "288GiB") <= 32


def test_envelope_single_gpu_margin(tmp_path):
    a = _gpu_profile(tmp_path / "GPUA.yaml", alpha_mib=400, beta_mib=1000, batches=[8, 64])
    _, profile = _envelope_sizer(tmp_path, [a], single_gpu_margin=1.25)
    alpha = float(profile["bench"]["envelope"][0]["alpha"].split()[0])
    assert 500.0 <= alpha <= 500.2  # 400 x 1.25, rounded up


def test_gpu_count():
    from milabench.sizer import gpu_count

    assert gpu_count({"devices": [3]}) == 1
    assert gpu_count({"devices": list(range(8))}) == 8
    assert gpu_count({"devices": list(range(8)), "num_machines": 2}) == 16
    assert gpu_count({"name": "x"}) is None
    assert gpu_count("bench") is None
