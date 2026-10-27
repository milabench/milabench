import time
from pathlib import Path

import pytest
from benchmate.nodemon.collectors import CpuCollector, IbCollector, PcieCollector


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
