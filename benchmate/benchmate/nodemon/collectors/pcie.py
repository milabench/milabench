from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..collector import CollectorMetric, MetricKind, NodeCollector, PolledFile


def parse_speed(text: str | None) -> float | None:
    """GT/s from a sysfs link speed such as "16.0 GT/s PCIe"."""
    try:
        return float(text.split()[0]) if text else None
    except (ValueError, IndexError):
        return None


def aer_total(text: str | None, field: str) -> int | None:
    """The value of one "<field> <value>" line of an aer_dev_* file."""
    for line in (text or "").splitlines():
        name, _, value = line.partition(" ")
        if name == field:
            try:
                return int(value)
            except ValueError:
                return None
    return None


class _PciDevice:
    """One PCI device's open link and AER files."""

    def __init__(self, path: Path, driver: str) -> None:
        self.address = path.name
        self.driver = driver
        self.files = {
            name: PolledFile(path / name)
            for name in (
                "current_link_width",
                "max_link_width",
                "current_link_speed",
                "max_link_speed",
                "aer_dev_fatal",
                "aer_dev_nonfatal",
                "aer_dev_correctable",
            )
        }


class PcieCollector(NodeCollector):
    name = "pcie"
    metrics = (
        CollectorMetric("device", MetricKind.DEVICE, "", "PCI address (0000:3b:00.0)"),
        CollectorMetric("driver", MetricKind.SAMPLE, "", "The driver bound to the device; empty for a bridge"),
        CollectorMetric("link_width", MetricKind.SAMPLE, "lanes", "Current link width"),
        CollectorMetric("max_link_width", MetricKind.SAMPLE, "lanes", "Widest link the device and its slot support"),
        CollectorMetric("link_speed", MetricKind.SAMPLE, "GT/s", "Current link speed per lane"),
        CollectorMetric("max_link_speed", MetricKind.SAMPLE, "GT/s", "Fastest link speed the device and its slot support"),
        CollectorMetric("aer_fatal", MetricKind.COUNTER, "errors", "Fatal uncorrectable AER errors since boot"),
        CollectorMetric("aer_nonfatal", MetricKind.COUNTER, "errors", "Non-fatal uncorrectable AER errors since boot"),
        CollectorMetric("aer_correctable", MetricKind.COUNTER, "errors", "Correctable AER errors since boot"),
    )
    #: The devices whose links matter for training: GPUs and InfiniBand HCAs.
    drivers = ("nvidia", "mlx5_core")
    address = re.compile(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]")

    def __init__(self, root: Path = Path("/sys/bus/pci")) -> None:
        self.root = Path(root)
        self._devices: list[_PciDevice] = []

    def probe(self) -> bool:
        """Find the GPUs and HCAs, and every bridge on their path up to the root port."""
        found: dict[str, Path] = {}
        for driver in self.drivers:
            for link in sorted((self.root / "drivers" / driver).glob("*:*")):
                if not self.address.fullmatch(link.name):
                    continue
                path = (self.root / "devices" / link.name).resolve()
                while self.address.fullmatch(path.name):  # stops at the host bridge, pci0000:00
                    found.setdefault(path.name, path)
                    path = path.parent
        self._devices = [
            _PciDevice(found[name], self._driver_of(found[name]))
            for name
            in sorted(found)
        ]
        return bool(self._devices)

    def _driver_of(self, path: Path) -> str:
        driver = path / "driver"
        return driver.resolve().name if driver.exists() else ""

    def sample(self, now_s: float) -> list[dict[str, Any]]:
        """One row per device."""
        rows = []
        for device in self._devices:
            files = device.files
            rows.append(
                {
                    "device": device.address,
                    "driver": device.driver,
                    "link_width": files["current_link_width"].read_int(),
                    "max_link_width": files["max_link_width"].read_int(),
                    "link_speed": parse_speed(files["current_link_speed"].read()),
                    "max_link_speed": parse_speed(files["max_link_speed"].read()),
                    "aer_fatal": aer_total(files["aer_dev_fatal"].read(), "TOTAL_ERR_FATAL"),
                    "aer_nonfatal": aer_total(files["aer_dev_nonfatal"].read(), "TOTAL_ERR_NONFATAL"),
                    "aer_correctable": aer_total(files["aer_dev_correctable"].read(), "TOTAL_ERR_COR"),
                }
            )
        return rows

    def close(self) -> None:
        for device in self._devices:
            for polled in device.files.values():
                polled.close()
        self._devices = []
