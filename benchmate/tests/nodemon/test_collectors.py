import time
from pathlib import Path

import pytest
from benchmate.nodemon.collectors import (
    CpuCollector,
    DcgmCollector,
    EdacCollector,
    IbCollector,
    PcieCollector,
)


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def ib_port(root: Path, hca: str, port: int, state: str, link_layer: str, xmit_lanes: int) -> Path:
    directory = root / hca / "ports" / str(port)
    write(directory / "state", state)
    write(directory / "link_layer", link_layer)
    write(directory / "rate", "100 Gb/sec (2X HDR)")
    for name in ("port_rcv_data", "port_xmit_wait", "symbol_error", "link_downed"):
        write(directory / "counters" / name, "0")
    write(directory / "counters" / "port_xmit_data", str(xmit_lanes))
    return directory


def test_ib_counts_bytes_from_lanes_on_active_infiniband_ports_only(tmp_path):
    active = ib_port(tmp_path, "mlx5_0", 1, "4: ACTIVE", "InfiniBand", 1000)
    ib_port(tmp_path, "mlx5_0", 2, "1: DOWN", "InfiniBand", 5)
    ib_port(tmp_path, "mlx5_1", 1, "4: ACTIVE", "Ethernet", 5)
    collector = IbCollector(tmp_path)

    assert collector.probe()
    collector.sample(now_s=0.0)
    write(active / "counters" / "port_xmit_data", "1250")  # 250 more lanes over 1 s
    write(active / "counters" / "port_xmit_wait", str((1 << 32) - 1))  # stuck at its 32-bit maximum
    (row,) = collector.sample(now_s=1.0)

    assert (row["hca"], row["port"]) == ("mlx5_0", "1")
    assert row["tx_bytes"] == 1250 * 4
    assert row["ib_tx_bytes_per_s"] == 1000.0
    assert row["ib_utilization"] == pytest.approx(1000.0 / 12.5e9)
    assert row["xmit_wait"] is None
    assert row["symbol_error"] == 0
    collector.close()


def proc_stat(ticks: dict[int, tuple[int, int]]) -> str:
    """A /proc/stat with each CPU's cumulative (user, idle) ticks."""
    lines = ["cpu  0 0 0 0 0 0 0 0 0 0"]
    for cpu, (user, idle) in ticks.items():
        lines.append(f"cpu{cpu} {user} 0 0 {idle} 0 0 0 0 0 0")
    return "\n".join(lines) + "\n"


def test_cpu_busy_core_clock_is_averaged_per_package(tmp_path):
    proc = tmp_path / "proc"
    cpu_root = tmp_path / "cpu"
    clocks = {0: 3700.0, 1: 2000.0, 2: 2600.0, 3: 2400.0}
    write(proc / "cpuinfo", "".join(f"processor\t: {cpu}\ncpu MHz\t\t: {mhz}\n\n" for cpu, mhz in clocks.items()))
    write(proc / "stat", proc_stat({cpu: (0, 100) for cpu in clocks}))
    for cpu, package in {0: 0, 1: 0, 2: 1, 3: 1}.items():
        write(cpu_root / f"cpu{cpu}" / "topology" / "physical_package_id", str(package))
    collector = CpuCollector(proc, cpu_root, tmp_path / "hwmon")

    assert collector.probe()
    # 100 more ticks each since probe(): cpu0, cpu2 and cpu3 were busy over them, cpu1 mostly idle.
    write(proc / "stat", proc_stat({0: (90, 110), 1: (10, 190), 2: (60, 140), 3: (70, 130)}))
    rows = {row["package"]: row for row in collector.sample(now_s=time.monotonic() + 1.0)}  # the first sample

    assert rows[0]["busy_cores"] == 1
    assert rows[0]["cpu_mhz_busy_mean"] == 3700.0
    assert rows[0]["cpu_mhz_mean"] == 2850.0
    assert rows[1]["busy_cores"] == 2
    assert rows[1]["cpu_mhz_busy_mean"] == 2500.0
    collector.close()


def test_pcie_walks_from_the_gpu_up_to_the_root_port(tmp_path):
    root_port = tmp_path / "devices" / "pci0000:00" / "0000:00:01.0"
    switch = root_port / "0000:01:00.0"
    gpu = switch / "0000:02:00.0"
    write(gpu / "current_link_width", "8")
    write(gpu / "max_link_width", "16")
    pci = tmp_path / "bus" / "pci"
    nvidia = pci / "drivers" / "nvidia"
    nvidia.mkdir(parents=True)
    (pci / "devices").mkdir()
    for device in (root_port, switch, gpu):
        (pci / "devices" / device.name).symlink_to(device)
    (nvidia / gpu.name).symlink_to(gpu)
    (gpu / "driver").symlink_to(nvidia)
    collector = PcieCollector(pci)

    assert collector.probe()
    rows = {row["device"]: row for row in collector.sample(now_s=0.0)}

    assert sorted(rows) == ["0000:00:01.0", "0000:01:00.0", "0000:02:00.0"]
    assert rows["0000:02:00.0"]["driver"] == "nvidia"
    assert (rows["0000:02:00.0"]["link_width"], rows["0000:02:00.0"]["max_link_width"]) == (8, 16)
    collector.close()


def test_edac_reads_each_controller_and_leaves_a_missing_counter_empty(tmp_path):
    write(tmp_path / "mc0" / "mc_name", "F17h_M30h")
    write(tmp_path / "mc0" / "ce_count", "7")
    write(tmp_path / "mc0" / "ue_count", "0")
    write(tmp_path / "mc1" / "mc_name", "F17h_M30h")
    write(tmp_path / "mc1" / "ce_count", "0")  # this controller exposes no ue_count
    collector = EdacCollector(tmp_path)

    assert collector.probe()
    first, second = collector.sample(now_s=0.0)
    collector.close()

    assert (first["mc"], first["mc_name"], first["ce_count"], first["ue_count"]) == ("mc0", "F17h_M30h", 7, 0)
    assert second["ue_count"] is None


EXPORTER_METRICS = """\
# HELP DCGM_FI_PROF_SM_ACTIVE The ratio of cycles an SM has at least 1 warp assigned.
DCGM_FI_PROF_SM_ACTIVE{gpu="0",UUID="GPU-a",Hostname="cn-l041"} 0.812
DCGM_FI_PROF_SM_ACTIVE{gpu="1",UUID="GPU-b",Hostname="cn-l041"} 0.004
DCGM_FI_PROF_PIPE_TENSOR_ACTIVE{gpu="0",UUID="GPU-a",Hostname="cn-l041"} 0.271
DCGM_FI_DEV_GPU_TEMP{gpu="0",UUID="GPU-a",Hostname="cn-l041"} 62
"""


def test_dcgm_reads_both_profiling_series_per_gpu_and_ignores_the_rest(monkeypatch):
    collector = DcgmCollector()
    monkeypatch.setattr(collector, "_scrape", lambda: EXPORTER_METRICS)

    first, second = collector.sample(now_s=0.0)

    assert first == {"gpu": 0, "sm_active": 0.812, "tensor_active": 0.271}
    assert second == {"gpu": 1, "sm_active": 0.004}  # the exporter published no tensor series for it


def test_dcgm_records_the_exporters_own_gpu_numbering_whatever_the_label_order(monkeypatch):
    collector = DcgmCollector()
    reordered = EXPORTER_METRICS.replace(
        'DCGM_FI_PROF_SM_ACTIVE{gpu="1",UUID="GPU-b"', 'DCGM_FI_PROF_SM_ACTIVE{UUID="GPU-b",gpu="1"'
    )
    monkeypatch.setattr(collector, "_scrape", lambda: reordered)

    collector.sample(now_s=0.0)

    # Under its own key, not "gpus": the analysis checks it against NVML's map rather than merging them.
    assert collector.describe()["exporter_gpus"] == {"0": "GPU-a", "1": "GPU-b"}


def test_dcgm_is_disabled_when_no_exporter_answers(monkeypatch):
    collector = DcgmCollector()
    monkeypatch.setattr(collector, "_scrape", lambda: None)

    assert not collector.probe()
    assert collector.sample(now_s=0.0) == []
