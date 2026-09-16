import json
from pathlib import Path

import pytest

from milabench.diagnostic import analyze_pack, analyze_run, load_pack_series, plot_pack_series


def _gpu_line(t, mem, load=1.0, device=0, power=300.0):
    return {
        "event": "data",
        "data": {
            "time": t,
            "task": "main",
            "gpudata": {
                str(device): {"memory": [mem, 81920.0], "load": load, "power": power},
            },
        },
    }


def _rate_line(t, rate=100.0):
    return {
        "event": "data",
        "data": {"time": t, "task": "train", "rate": rate, "units": "items/s"},
    }


def _write_data(path: Path, lines):
    with open(path, "w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")


def test_detects_load_dip_with_flat_memory(tmp_path):
    path = tmp_path / "resnet50-noio.D0.data"
    t0 = 1_000_000.0
    lines = []
    for i in range(40):
        lines.append(_gpu_line(t0 + i * 0.25, 27000, load=0.99))
    for i in range(40, 120):
        lines.append(_gpu_line(t0 + i * 0.25, 27000, load=0.05))
    for i in range(120, 160):
        lines.append(_gpu_line(t0 + i * 0.25, 27000, load=0.99))
    _write_data(path, lines)

    report = analyze_pack(path)
    kinds = {i.kind for i in report.issues}
    assert "LOAD_DIP" in kinds


def test_detects_teardown_not_mid_cliff(tmp_path):
    path = tmp_path / "lightning.D0.data"
    t0 = 2_000_000.0
    lines = []
    for i in range(360):
        lines.append(_gpu_line(t0 + i * 0.25, 26500, load=1.0))
    lines.append(_gpu_line(t0 + 360 * 0.25, 1200, load=0.1))
    _write_data(path, lines)

    report = analyze_pack(path)
    kinds = {i.kind for i in report.issues}
    assert "TEARDOWN" in kinds
    assert "MEM_CLIFF_MID" not in kinds


def test_load_pack_series_aligns_rate_to_gpu_time(tmp_path):
    path = tmp_path / "bench.D0.data"
    t0 = 1_000_000.0
    lines = [_gpu_line(t0 + i * 0.25, 10000, load=0.9, power=250.0) for i in range(20)]
    lines.extend(_rate_line(t0 + 5.0 + i * 0.4, rate=100.0 + i) for i in range(10))
    _write_data(path, lines)

    series = load_pack_series(path)
    assert len(series.gpu_t) == 20
    assert len(series.rate_t) == 10
    assert series.rate_t[0] == pytest.approx(5.0)
    assert series.power_w[0] == pytest.approx(250.0)


def test_load_pack_series_parses_all_devices(tmp_path):
    path = tmp_path / "lightning-gpus.0.data"
    t0 = 4_000_000.0
    lines = []
    for i in range(5):
        gpudata = {
            str(d): {"memory": [10000 + d * 100, 81920.0], "load": 0.9, "power": 200.0 + d}
            for d in range(8)
        }
        lines.append({"event": "data", "data": {"time": t0 + i * 0.25, "task": "main", "gpudata": gpudata}})
    _write_data(path, lines)

    series = load_pack_series(path)
    assert len(series.devices) == 8
    assert series.devices["3"].mem_mib[0] == pytest.approx(10300.0)


def test_plot_pack_series_writes_png(tmp_path):
    path = tmp_path / "bench.D0.data"
    t0 = 3_000_000.0
    lines = [_gpu_line(t0 + i * 0.25, 5000, power=200.0) for i in range(5)]
    lines.append(_rate_line(t0 + 1.0, rate=42.0))
    _write_data(path, lines)

    out = tmp_path / "out.png"
    plot_pack_series(load_pack_series(path), out)
    assert out.is_file() and out.stat().st_size > 0


def test_cli_on_bomelugu():
    run = Path("/home/delaunap/work/milabench_dev/data/1x8xA100_run_055c0b64/runs/bomelugu.2026-09-16_10-30-19")
    if not run.is_dir():
        pytest.skip("bomelugu run data not available")

    report = analyze_run(run, select="lightning.D0")
    assert len(report.packs) == 1
    assert report.packs[0].n_gpudata > 0
