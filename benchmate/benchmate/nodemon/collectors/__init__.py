from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..registry import CollectorRegistry, Sink
from .cpu import CpuCollector
from .gpu import GpuCollector
from .ib import IbCollector
from .mem import MemCollector
from .nvlink import NvlinkCollector
from .pcie import PcieCollector

__all__ = ["CpuCollector", "GpuCollector", "IbCollector", "MemCollector", "NvlinkCollector", "PcieCollector", "default_registry"]


def default_registry(
    sink: Sink | None = None,
    options: Mapping[str, Mapping[str, Any]] | None = None,
) -> CollectorRegistry:
    """A registry with every known collector, each registered at its period.

    `options` passes constructor arguments by collector name (e.g. fake sysfs
    roots in tests). Unknown names raise ValueError.
    """
    collectors = {
        "gpu": (GpuCollector, 1.0), # will show thermal throttling as reduced SM clock
        "nvlink": (NvlinkCollector, 10.0),  # costly to read, and cumulative: see NvlinkCollector
        "ib": (IbCollector, 1.0),
        "cpu": (CpuCollector, 2.0), # a 96-core node takes 2 ms per collection
        "mem": (MemCollector, 1.0),
        "pcie": (PcieCollector, 10.0),  # link states and AER counts change rarely
    }
    arguments = dict(options or {})
    unknown = set(arguments) - set(collectors)
    if unknown:
        raise ValueError(f"Unknown collectors {sorted(unknown)}; known: {sorted(collectors)}")
    registry = CollectorRegistry(sink)
    for name, (cls, period_s) in collectors.items():
        registry.register(cls(**arguments.get(name, {})), period_s)
    return registry
