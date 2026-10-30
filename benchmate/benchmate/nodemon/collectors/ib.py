from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from ..collector import (
    CollectorMetric,
    MetricKind,
    NodeCollector,
    PolledFile,
    Rate,
    read_attr,
)


def _carries_traffic(port: Path) -> bool:
    """Whether a sysfs port is an active InfiniBand link.

    Only rejects a port that *says* it is down or not InfiniBand (e.g. RoCE):
    a missing attribute means unknown, not inactive.
    """
    state = read_attr(port / "state")
    if state and "active" not in state.lower():
        return False
    link_layer = read_attr(port / "link_layer")
    return not link_layer or link_layer.lower() == "infiniband"


def _parse_rate_gbps(text: str | None) -> float | None:
    """The link rate in Gb/s from a sysfs `rate` such as "100 Gb/sec (2X HDR)"."""
    try:
        return float(text.split()[0]) if text else None
    except (ValueError, IndexError):
        return None


class _Port:
    """One port's open counter files."""

    counters_read = (
        "port_xmit_data",
        "port_rcv_data",
        "port_xmit_wait",
        "symbol_error",
        "link_downed"
    )

    def __init__(self, port: Path) -> None:
        self.hca = port.parents[1].name
        self.port = port.name
        self.rate = PolledFile(port / "rate")
        # The narrow PortCounters value sticks at 2^32-1. ConnectX publishes the same count as a
        # 64-bit PPCNT counter under hw_counters, which keeps counting past that.
        extended = port / "hw_counters" / "port_xmit_wait"
        self.xmit_wait_extended = extended.is_file()
        self.counters = {
            name: PolledFile(
                extended if name == "port_xmit_wait" and self.xmit_wait_extended else port / "counters" / name
            )
            for name
            in self.counters_read
        }

    def close(self) -> None:
        for polled in [self.rate, *self.counters.values()]:
            polled.close()


class IbCollector(NodeCollector):
    name = "ib"
    metrics = (
        CollectorMetric("hca", MetricKind.DEVICE, "", "Host channel adapter, as sysfs names it (mlx5_0)"),
        CollectorMetric("port", MetricKind.DEVICE, "", "Port number on the adapter"),
        CollectorMetric("rate_gbps", MetricKind.SAMPLE, "Gb/s", "The link's rate in each direction"),
        CollectorMetric("tx_bytes", MetricKind.COUNTER, "B", "Bytes sent since the counter was last reset"),
        CollectorMetric("rx_bytes", MetricKind.COUNTER, "B", "Bytes received since the counter was last reset"),
        CollectorMetric("ib_tx_bytes_per_s", MetricKind.SAMPLE, "B/s", "Send rate over the interval"),
        CollectorMetric("ib_rx_bytes_per_s", MetricKind.SAMPLE, "B/s", "Receive rate over the interval"),
        CollectorMetric("ib_utilization", MetricKind.SAMPLE, "fraction", "The busier direction's share of the link rate"),
        CollectorMetric(
            "xmit_wait", MetricKind.COUNTER, "ticks",
            "Ticks the port had data to send but the fabric would not take it: back-pressure",
        ),
        CollectorMetric("symbol_error", MetricKind.COUNTER, "errors", "Symbol errors on the link"),
        CollectorMetric("link_downed", MetricKind.COUNTER, "count", "Times the link went down"),
    )
    #: port_{xmit,rcv}_data count 4-byte lanes, not bytes: missing this understates traffic 4x.
    lane_bytes = 4
    #: Where the narrow IBA PortCounters stick instead of wrapping. A stuck counter no
    #: longer counts, so it is reported as unknown rather than as a flat line.
    saturation: ClassVar[dict[str, int]] = {
        "port_xmit_wait": (1 << 32) - 1,
        "symbol_error": (1 << 16) - 1,
        "link_downed": (1 << 8) - 1,
    }

    def __init__(self, root: Path = Path("/sys/class/infiniband")) -> None:
        self.root = Path(root)
        self._ports: list[_Port] = []
        self._rates = Rate()

    def probe(self) -> bool:
        """Find the active InfiniBand ports; False when there are none."""
        self._ports = [
            _Port(port)
            for port
            in sorted(self.root.glob("*/ports/*"))
            if _carries_traffic(port)
        ]
        return bool(self._ports)

    def sample(self, now_s: float) -> list[dict[str, Any]]:
        """One row per port: cumulative bytes, rates, and the cumulative error counters."""
        rows = []
        for port in self._ports:
            values = {
                name: self._counter(name, polled.read_int(), port.xmit_wait_extended)
                for name, polled
                in port.counters.items()
            }
            tx = self._bytes(values["port_xmit_data"])
            rx = self._bytes(values["port_rcv_data"])
            tx_rate = self._rates.update((port.hca, port.port, "tx"), now_s, tx)
            rx_rate = self._rates.update((port.hca, port.port, "rx"), now_s, rx)
            rate_gbps = _parse_rate_gbps(port.rate.read())
            rows.append(
                {
                    "hca": port.hca,
                    "port": port.port,
                    "rate_gbps": rate_gbps,
                    "tx_bytes": tx,
                    "rx_bytes": rx,
                    "ib_tx_bytes_per_s": tx_rate,
                    "ib_rx_bytes_per_s": rx_rate,
                    "ib_utilization": _utilization(tx_rate, rx_rate, rate_gbps),
                    "xmit_wait": values["port_xmit_wait"],
                    "symbol_error": values["symbol_error"],
                    "link_downed": values["link_downed"],
                }
            )
        return rows

    def close(self) -> None:
        for port in self._ports:
            port.close()
        self._ports = []

    def _bytes(self, lanes: int | None) -> int | None:
        return None if lanes is None else lanes * self.lane_bytes

    def _counter(self, name: str, value: int | None, extended: bool) -> int | None:
        # The 64-bit counter does not stick at the 32-bit maximum, so that value is a real reading there.
        if name == "port_xmit_wait" and extended:
            return value
        return None if value == self.saturation.get(name) else value


def _utilization(tx_rate: float | None, rx_rate: float | None, rate_gbps: float | None) -> float | None:
    """The busier direction's share of the link rate, which applies to each direction."""
    if tx_rate is None or rx_rate is None or not rate_gbps:
        return None
    return max(tx_rate, rx_rate) / (rate_gbps * 1e9 / 8)
