from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from ..collector import NodeCollector, PolledFile, Rate, read_attr


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
        self.counters = {
            name: PolledFile(port / "counters" / name)
            for name
            in self.counters_read
        }

    def close(self) -> None:
        for polled in [self.rate, *self.counters.values()]:
            polled.close()


class IbCollector(NodeCollector):
    name = "ib"
    columns = (
        "hca",
        "port",
        "rate_gbps",
        "tx_bytes",
        "rx_bytes",
        "ib_tx_bytes_per_s",
        "ib_rx_bytes_per_s",
        "ib_utilization",
        "xmit_wait",
        "symbol_error",
        "link_downed",
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
                name: self._counter(name, polled.read_int())
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

    def _counter(self, name: str, value: int | None) -> int | None:
        return None if value == self.saturation.get(name) else value


def _utilization(tx_rate: float | None, rx_rate: float | None, rate_gbps: float | None) -> float | None:
    """The busier direction's share of the link rate, which applies to each direction."""
    if tx_rate is None or rx_rate is None or not rate_gbps:
        return None
    return max(tx_rate, rx_rate) / (rate_gbps * 1e9 / 8)
