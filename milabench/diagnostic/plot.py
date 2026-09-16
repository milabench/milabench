"""Scatter plots of raw GPU and training metrics from ``*.data`` files."""

from __future__ import annotations

import re
from pathlib import Path

from .analyze import DeviceSeries, PackSeries, _device_sort_key, load_pack_series

_RANK_PACK = re.compile(r"^(?P<base>.+)\.D(?P<rank>\d+)$")


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for diagnostic plots; install with: pip install matplotlib"
        ) from e
    return plt


def _plot_device_metric(ax, devices: dict[str, DeviceSeries], values: str, colors, dot_kw) -> None:
    for i, (dev, ds) in enumerate(sorted(devices.items(), key=lambda item: _device_sort_key(item[0]))):
        y = getattr(ds, values)
        ax.scatter(ds.t, y, c=[colors[i % len(colors)]], label=f"GPU {dev}", **dot_kw)


def plot_pack_series(series: PackSeries, out_path: Path, *, title: str | None = None) -> Path:
    """Write one figure with load, memory, power, and rate on a shared time axis."""
    plt = _require_matplotlib()

    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(12, 9), constrained_layout=True)
    dot_kw = {"s": 6, "marker": ".", "linewidths": 0, "alpha": 0.85}
    colors = plt.cm.tab10.colors
    multi = len(series.devices) > 1

    if multi:
        _plot_device_metric(axes[0], series.devices, "load", colors, dot_kw)
        _plot_device_metric(axes[1], series.devices, "mem_mib", colors, dot_kw)
        _plot_device_metric(axes[2], series.devices, "power_w", colors, dot_kw)
        axes[0].legend(loc="upper right", fontsize=8, ncol=4)
    else:
        axes[0].scatter(series.gpu_t, series.load, c="#1f77b4", **dot_kw)
        axes[1].scatter(series.gpu_t, series.mem_mib, c="#ff7f0e", **dot_kw)
        axes[2].scatter(series.gpu_t, series.power_w, c="#2ca02c", **dot_kw)

    axes[0].set_ylabel("GPU load")
    axes[0].set_ylim(-0.05, 1.05)
    axes[1].set_ylabel("GPU mem (MiB)")
    axes[2].set_ylabel("GPU power (W)")

    if series.rate_t:
        axes[3].scatter(series.rate_t, series.rate, c="#d62728", **dot_kw)
    axes[3].set_ylabel("Rate (items/s)")
    axes[3].set_xlabel("Time (s)")

    fig.suptitle(title or series.name, fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _gpus_file_for_rank_pack(path: Path, folder: Path) -> Path | None:
    match = _RANK_PACK.match(path.stem)
    if not match:
        return None
    gpus = folder / f"{match.group('base')}-gpus.0.data"
    return gpus if gpus.is_file() else None


def _plot_source(path: Path, folder: Path) -> tuple[Path, str]:
    """Resolve rank-local ``*.D#`` packs to their ``*-gpus.0.data`` aggregate."""
    gpus = _gpus_file_for_rank_pack(path, folder)
    if gpus is not None:
        return gpus, _RANK_PACK.match(path.stem).group("base")
    return path, path.stem


def plot_run(
    folder: Path,
    *,
    select: str | None = None,
    plot_dir: Path | None = None,
) -> list[Path]:
    """Generate scatter plots for each matching ``*.data`` pack in *folder*."""
    files = sorted(folder.glob("*.data"))
    if select:
        files = [f for f in files if select in f.name]
    if not files:
        return []

    out_dir = plot_dir or (folder / "diagnostic_plots")
    written: list[Path] = []
    seen_sources: set[Path] = set()
    for path in files:
        source, title = _plot_source(path, folder)
        if source in seen_sources:
            continue
        seen_sources.add(source)
        series = load_pack_series(source)
        out_name = title if source != path or len(series.devices) > 1 else series.name
        out_path = out_dir / f"{out_name}.png"
        written.append(plot_pack_series(series, out_path, title=out_name))
    return written
