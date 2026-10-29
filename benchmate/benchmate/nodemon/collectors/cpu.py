from __future__ import annotations

import time
from pathlib import Path
from statistics import fmean
from typing import Any

from ..collector import (
    CollectorMetric,
    MetricKind,
    NodeCollector,
    PolledFile,
    read_attr,
)


def parse_stat(text: str) -> dict[int, tuple[int, int]]:
    """Each CPU's cumulative (busy, total) ticks from /proc/stat."""
    ticks = {}
    for line in text.splitlines():
        if not line.startswith("cpu") or not line[3:4].isdigit():
            continue
        fields = line.split()
        # guest and guest_nice (past the 8th) are already counted in user and nice.
        values = [int(value) for value in fields[1:9]]
        idle = sum(values[3:5])  # idle + iowait
        total = sum(values)
        ticks[int(fields[0][3:])] = (total - idle, total)
    return ticks


def cpu_mhz(text: str) -> dict[int, float]:
    """Each CPU's clock from /proc/cpuinfo, which has one block per CPU separated by a blank line."""
    clocks = {}
    # Searching each block for its two lines, rather than testing every line, is ~3x faster.
    for block in text.split("\n\n"):
        start = block.find("cpu MHz")
        if not block.startswith("processor") or start < 0:
            continue
        processor = block.partition("\n")[0].partition(":")[2]
        mhz = block[start:].partition("\n")[0].partition(":")[2]
        clocks[int(processor)] = float(mhz)
    return clocks


def package_temperatures(hwmon_root: Path) -> dict[int, Path]:
    """Each package's temperature file: coretemp's "Package id N", or one k10temp per package."""
    temps = {}
    k10temp = []
    for hwmon in sorted(hwmon_root.glob("hwmon*")):
        name = read_attr(hwmon / "name")
        if name == "coretemp":
            for label in hwmon.glob("temp*_label"):
                words = (read_attr(label) or "").split()
                if words[:2] == ["Package", "id"] and words[-1].isdigit():
                    temps[int(words[-1])] = _input_of(label)
        elif name == "k10temp":
            k10temp.append(hwmon)
    # AMD has one k10temp device per package; their PCI order is the package order.
    for package, hwmon in enumerate(sorted(k10temp, key=lambda path: (path / "device").resolve())):
        labels = {
            read_attr(label): label
            for label
            in hwmon.glob("temp*_label")
        }
        label = labels.get("Tdie") or labels.get("Tctl")  # Tctl can carry a fan-control offset
        temps[package] = _input_of(label) if label else hwmon / "temp1_input"
    return temps


def _input_of(label: Path) -> Path:
    return label.with_name(label.name.replace("_label", "_input"))


class _Package:
    """One package's CPUs and open files."""

    def __init__(self, package: int, first_cpu: Path, temp: Path | None) -> None:
        self.package = package
        self.cpus: list[int] = []
        self.max_freq: list[PolledFile] = []
        self.governor = PolledFile(first_cpu / "cpufreq" / "scaling_governor")
        self.throttle = PolledFile(first_cpu / "thermal_throttle" / "package_throttle_count")
        #: None when the host has no temperature sensor for this package.
        self.temp = PolledFile(temp) if temp else None

    def close(self) -> None:
        for polled in [*self.max_freq, self.governor, self.throttle, *filter(None, (self.temp,))]:
            polled.close()


class CpuCollector(NodeCollector):
    name = "cpu"
    metrics = (
        CollectorMetric("package", MetricKind.DEVICE, "", "Physical package (socket) id"),
        CollectorMetric("busy_cores", MetricKind.SAMPLE, "count", "Cores at or above the busy utilization over the last tick"),
        CollectorMetric(
            "cpu_mhz_busy_mean", MetricKind.SAMPLE, "MHz",
            "Mean clock of the busy cores. Usually the column to watch: the mean over all cores hides a busy core "
            "that slowed down, since idle cores report their nominal clock. cpu_mhz_cap, cpu_temp_c and "
            "thermal_throttle_count give the cause.",
        ),
        CollectorMetric("cpu_mhz_min", MetricKind.SAMPLE, "MHz", "Lowest core clock in the package"),
        CollectorMetric("cpu_mhz_max", MetricKind.SAMPLE, "MHz", "Highest core clock in the package"),
        CollectorMetric("cpu_mhz_mean", MetricKind.SAMPLE, "MHz", "Mean clock over every core, idle ones included"),
        CollectorMetric("cpu_mhz_cap", MetricKind.SAMPLE, "MHz", "The tightest cpufreq cap (scaling_max_freq) on any of the package's cores"),
        CollectorMetric("cpu_util", MetricKind.SAMPLE, "fraction", "Mean core utilization over the last tick"),
        CollectorMetric("cpu_temp_c", MetricKind.SAMPLE, "C", "Package temperature; empty without a sensor"),
        CollectorMetric(
            "thermal_throttle_count", MetricKind.COUNTER, "count",
            "Times the package hit its thermal limit and throttled since boot",
        ),
        CollectorMetric("governor", MetricKind.SAMPLE, "", "The cpufreq scaling governor"),
    )
    #: A core at or above this utilization over the last tick counts as busy.
    busy = 0.5
    #: Shortest span, in seconds, utilization is measured over: at the usual 100 jiffies
    #: a second, a shorter one would make it coarse. A shorter tick leaves it to the next.
    min_span = 0.1

    def __init__(
        self,
        proc: Path = Path("/proc"),
        cpu_root: Path = Path("/sys/devices/system/cpu"),
        hwmon_root: Path = Path("/sys/class/hwmon"),
    ) -> None:
        self.proc = Path(proc)
        self.cpu_root = Path(cpu_root)
        self.hwmon_root = Path(hwmon_root)
        self._packages: list[_Package] = []
        self._stat: PolledFile | None = None
        self._cpuinfo: PolledFile | None = None
        self._last_stat: dict[int, tuple[int, int]] = {}
        self._last_time_s = 0.0

    def probe(self) -> bool:
        """Group the online CPUs by package and open each package's files."""
        self._stat = PolledFile(self.proc / "stat", max_bytes=None)
        self._cpuinfo = PolledFile(self.proc / "cpuinfo", max_bytes=None)
        stat = self._stat.read()
        if not stat or not self._cpuinfo.available:
            return False
        temps = package_temperatures(self.hwmon_root)
        packages: dict[int, _Package] = {}
        for cpu in sorted(parse_stat(stat)):
            directory = self.cpu_root / f"cpu{cpu}"
            package_id = int(read_attr(directory / "topology" / "physical_package_id") or 0)
            if package_id not in packages:
                packages[package_id] = _Package(package_id, directory, temps.get(package_id))
            package = packages[package_id]
            package.cpus.append(cpu)
            package.max_freq.append(PolledFile(directory / "cpufreq" / "scaling_max_freq"))
        self._packages = [packages[key] for key in sorted(packages)]
        # The baseline for the first tick's utilization, so its row is not left empty.
        self._last_stat = parse_stat(stat)
        self._last_time_s = time.monotonic()
        return bool(self._packages)

    def sample(self, now_s: float) -> list[dict[str, Any]]:
        """One row per package."""
        if not self._stat or not self._cpuinfo:  # closed
            return []
        stat = parse_stat(self._stat.read() or "")
        clocks = cpu_mhz(self._cpuinfo.read() or "")
        util = {}
        if now_s - self._last_time_s >= self.min_span:
            for cpu, (busy, total) in stat.items():
                previous = self._last_stat.get(cpu)
                if previous and total > previous[1]:
                    util[cpu] = (busy - previous[0]) / (total - previous[1])
            self._last_stat = stat
            self._last_time_s = now_s
        return [self._row(package, util, clocks) for package in self._packages]

    def _row(self, package: _Package, util: dict[int, float], clocks: dict[int, float]) -> dict[str, Any]:
        """One package's row, from this tick's utilization and clocks."""
        utils = [util[cpu] for cpu in package.cpus if cpu in util]
        busy = [cpu for cpu in package.cpus if util.get(cpu, 0.0) >= self.busy]
        mhz = [clocks[cpu] for cpu in package.cpus if cpu in clocks]
        max_khz = [value for value in map(PolledFile.read_int, package.max_freq) if value is not None]
        temp = package.temp.read_int() if package.temp else None
        governor = package.governor.read()
        return {
            "package": package.package,
            # Unknown on a tick too soon after the last measurement.
            "busy_cores": len(busy) if utils else None,
            "cpu_mhz_busy_mean": _round(_mean([clocks[cpu] for cpu in busy if cpu in clocks])),
            "cpu_mhz_min": _round(min(mhz)) if mhz else None,
            "cpu_mhz_max": _round(max(mhz)) if mhz else None,
            "cpu_mhz_mean": _round(_mean(mhz)),
            # The tightest cpufreq cap (scaling_max_freq) on any of the package's cores.
            "cpu_mhz_cap": _round(min(max_khz) / 1000) if max_khz else None,
            "cpu_util": _mean(utils),
            "cpu_temp_c": None if temp is None else temp / 1000,
            "thermal_throttle_count": package.throttle.read_int(),
            "governor": governor.strip() if governor else None,
        }

    def close(self) -> None:
        """Close every package's files and the /proc files."""
        for package in self._packages:
            package.close()
        for polled in filter(None, (self._stat, self._cpuinfo)):
            polled.close()
        self._packages = []
        self._stat = None
        self._cpuinfo = None


def _mean(values: list[float]) -> float | None:
    return fmean(values) if values else None


def _round(mhz: float | None) -> int | None:
    # Whole MHz: the clock moves in steps far larger than a fraction of one.
    return None if mhz is None else round(mhz)
