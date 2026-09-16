"""Scan milabench ``*.data`` files for metric timeline anomalies."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

SEVERITY_ORDER = {"info": 0, "warn": 1, "error": 2}

BASELINE_MEM_MIB = 2500  # empty GPU-ish NVML reading


@dataclass
class GpuSample:
    t: float
    mem: float
    load: float
    power: float = 0.0


@dataclass
class DeviceSeries:
    t: list[float]
    load: list[float]
    mem_mib: list[float]
    power_w: list[float]


@dataclass
class PackSeries:
    name: str
    gpu_t: list[float]
    load: list[float]
    mem_mib: list[float]
    power_w: list[float]
    rate_t: list[float]
    rate: list[float]
    devices: dict[str, DeviceSeries] = field(default_factory=dict)


@dataclass
class Issue:
    kind: str
    severity: str
    t_start: float
    t_end: float | None
    message: str

    def __post_init__(self):
        if self.severity not in SEVERITY_ORDER:
            raise ValueError(f"unknown severity: {self.severity}")


@dataclass
class PackReport:
    name: str
    duration: float = 0.0
    n_gpudata: int = 0
    n_rates: int = 0
    n_iter: int = 0
    n_format_errors: int = 0
    peak_mem_mib: float = 0.0
    issues: list[Issue] = field(default_factory=list)


@dataclass
class RunReport:
    folder: Path
    packs: list[PackReport] = field(default_factory=list)

    @property
    def issues(self) -> list[tuple[str, Issue]]:
        out: list[tuple[str, Issue]] = []
        for pack in self.packs:
            for issue in pack.issues:
                out.append((pack.name, issue))
        return out


def _parse_data_file(path: Path) -> tuple[list[GpuSample], list[tuple[float, float]], list[tuple[float, float]], int, float | None]:
    """Return gpu samples, rates, iter events, format_error count, t0."""
    gpu_raw: list[tuple[float, float, float, float]] = []
    rates_raw: list[tuple[float, float]] = []
    iters_raw: list[tuple[float, float]] = []
    format_errors = 0

    with open(path, encoding="utf-8", errors="replace") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue

            event = ev.get("event")
            if event == "format_error":
                format_errors += 1
                continue
            if event != "data":
                continue

            data = ev.get("data") or {}
            t_abs = data.get("time")

            if gd := data.get("gpudata"):
                mems = [g["memory"][0] for g in gd.values() if g.get("memory")]
                loads = [g.get("load", 0.0) for g in gd.values()]
                powers = [g.get("power", 0.0) for g in gd.values() if g.get("power") is not None]
                if mems and t_abs is not None:
                    gpu_raw.append(
                        (
                            float(t_abs),
                            sum(mems) / len(mems),
                            max(loads),
                            sum(powers) / len(powers) if powers else 0.0,
                        )
                    )

            if t_abs is not None and (rate := data.get("rate")) is not None:
                rates_raw.append((float(t_abs), float(rate)))

            if t_abs is not None and (it := data.get("__iter__")) is not None:
                iters_raw.append((float(t_abs), float(it)))

    gpu = _relative_timeline(gpu_raw)
    rates = _relative_pairs(rates_raw)
    iters = _relative_pairs(iters_raw)
    t0 = gpu_raw[0][0] if gpu_raw else (rates_raw[0][0] if rates_raw else None)
    return gpu, rates, iters, format_errors, t0


def _relative_timeline(raw: list[tuple[float, ...]]) -> list[GpuSample]:
    if not raw:
        return []

    segments: list[list[tuple[float, ...]]] = [[raw[0]]]
    for i in range(1, len(raw)):
        if raw[i][0] < raw[i - 1][0] - 1.0:
            segments.append([raw[i]])
        else:
            segments[-1].append(raw[i])

    out: list[GpuSample] = []
    offset = 0.0
    for seg in segments:
        base = seg[0][0]
        last = 0.0
        for row in seg:
            t_abs, mem, load = row[0], row[1], row[2]
            power = row[3] if len(row) > 3 else 0.0
            rel = offset + (t_abs - base)
            out.append(GpuSample(rel, mem, load, power))
            last = rel
        if len(seg) > 1:
            offset = last + (seg[-1][0] - seg[-2][0])
        else:
            offset = last + 0.25
    return out


def _relative_pairs(raw: list[tuple[float, float]], *, t0: float | None = None) -> list[tuple[float, float]]:
    if not raw:
        return []
    anchor = raw[0][0] if t0 is None else t0
    return [(t - anchor, v) for t, v in raw]


def _map_abs_to_relative(t_abs: float, gpu_raw: list[tuple[float, ...]]) -> float:
    """Map an absolute timestamp onto the segmented gpu timeline."""
    if not gpu_raw:
        return t_abs

    segments: list[list[tuple[float, ...]]] = [[gpu_raw[0]]]
    for i in range(1, len(gpu_raw)):
        if gpu_raw[i][0] < gpu_raw[i - 1][0] - 1.0:
            segments.append([gpu_raw[i]])
        else:
            segments[-1].append(gpu_raw[i])

    offset = 0.0
    for seg in segments:
        base = seg[0][0]
        end_abs = seg[-1][0]
        if t_abs < base - 1.0:
            continue
        if t_abs <= end_abs + 1.0:
            return offset + (t_abs - base)
        last = offset + (end_abs - base)
        if len(seg) > 1:
            offset = last + (seg[-1][0] - seg[-2][0])
        else:
            offset = last + 0.25

    base = gpu_raw[0][0]
    return t_abs - base


def _device_sort_key(dev: str) -> tuple[int, str]:
    return (int(dev), dev) if dev.isdigit() else (10_000, dev)


def load_pack_series(path: Path) -> PackSeries:
    """Load scatter-plot series for one ``*.data`` pack with a shared time axis."""
    gpu_raw: list[tuple[float, float, float, float]] = []
    device_raw: dict[str, list[tuple[float, float, float, float]]] = {}
    rates_raw: list[tuple[float, float]] = []

    with open(path, encoding="utf-8", errors="replace") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("event") != "data":
                continue

            data = ev.get("data") or {}
            t_abs = data.get("time")

            if gd := data.get("gpudata"):
                mems = [g["memory"][0] for g in gd.values() if g.get("memory")]
                loads = [g.get("load", 0.0) for g in gd.values()]
                powers = [g.get("power", 0.0) for g in gd.values() if g.get("power") is not None]
                if mems and t_abs is not None:
                    t_abs = float(t_abs)
                    gpu_raw.append(
                        (
                            t_abs,
                            sum(mems) / len(mems),
                            max(loads),
                            sum(powers) / len(powers) if powers else 0.0,
                        )
                    )
                    for dev, g in gd.items():
                        if not g.get("memory"):
                            continue
                        device_raw.setdefault(dev, []).append(
                            (
                                t_abs,
                                float(g["memory"][0]),
                                float(g.get("load", 0.0)),
                                float(g.get("power", 0.0) or 0.0),
                            )
                        )

            if t_abs is not None and (rate := data.get("rate")) is not None:
                rates_raw.append((float(t_abs), float(rate)))

    gpu = _relative_timeline(gpu_raw)
    devices: dict[str, DeviceSeries] = {}
    for dev, raw in sorted(device_raw.items(), key=lambda item: _device_sort_key(item[0])):
        samples = _relative_timeline(raw)
        devices[dev] = DeviceSeries(
            t=[s.t for s in samples],
            load=[s.load for s in samples],
            mem_mib=[s.mem for s in samples],
            power_w=[s.power for s in samples],
        )

    if gpu_raw:
        rate_t = [_map_abs_to_relative(t, gpu_raw) for t, _ in rates_raw]
    else:
        rate_t = [t - rates_raw[0][0] for t, _ in rates_raw] if rates_raw else []
    rate = [v for _, v in rates_raw]

    return PackSeries(
        name=path.stem,
        gpu_t=[g.t for g in gpu],
        load=[g.load for g in gpu],
        mem_mib=[g.mem for g in gpu],
        power_w=[g.power for g in gpu],
        rate_t=rate_t,
        rate=rate,
        devices=devices,
    )


def _bucket(gpu: list[GpuSample], width: float = 10.0) -> list[tuple[float, float, float, float, int]]:
    if not gpu:
        return []
    mx = gpu[-1].t + width
    rows = []
    for start in range(0, int(mx) + 1, int(width)):
        seg = [g for g in gpu if start <= g.t < start + width]
        if not seg:
            continue
        rows.append(
            (
                float(start),
                float(start + width),
                sum(g.mem for g in seg) / len(seg),
                sum(g.load for g in seg) / len(seg),
                len(seg),
            )
        )
    return rows


def _detect_issues(
    gpu: list[GpuSample],
    rates: list[tuple[float, float]],
    iters: list[tuple[float, float]],
    format_errors: int,
) -> list[Issue]:
    issues: list[Issue] = []
    if not gpu and not rates:
        issues.append(Issue("NO_METRICS", "error", 0.0, None, "no gpudata or rate samples"))
        return issues

    duration = gpu[-1].t if gpu else (rates[-1][0] if rates else 0.0)

    if format_errors:
        issues.append(
            Issue(
                "FORMAT_ERROR",
                "warn",
                0.0,
                duration,
                f"{format_errors} malformed/concatenated metric line(s) in .data file",
            )
        )

    # --- gpudata gaps & timestamp regressions (on raw samples before rebase) ---
    for i in range(1, len(gpu)):
        dt = gpu[i].t - gpu[i - 1].t
        if dt < -0.5:
            issues.append(
                Issue(
                    "TIMESTAMP_REGRESSION",
                    "error",
                    gpu[i].t,
                    gpu[i].t,
                    f"time went backwards by {abs(dt):.1f}s (mixed clocks or merged writers)",
                )
            )
        elif dt > 3.0:
            at_startup = gpu[i - 1].t < 10 and gpu[i - 1].mem < BASELINE_MEM_MIB
            sev = "error" if dt > 30 else ("info" if at_startup else "warn")
            kind = "JOB_RESTART" if dt > 300 and gpu[i].mem < BASELINE_MEM_MIB else "GPUDATA_GAP"
            msg = (
                f"no gpudata for {dt:.0f}s "
                f"(mem {gpu[i - 1].mem:.0f}->{gpu[i].mem:.0f} MiB, "
                f"load {gpu[i - 1].load:.2f}->{gpu[i].load:.2f})"
            )
            if at_startup:
                msg += " (monitor started before GPU allocations)"
            if kind == "JOB_RESTART":
                msg = f"likely job restart/preemption: {msg}"
            issues.append(Issue(kind, sev, gpu[i - 1].t, gpu[i].t, msg))

    # --- startup warmup ---
    if gpu:
        warmup_end = min(15.0, duration * 0.2)
        warm = [g for g in gpu if g.t <= warmup_end]
        if warm and warm[-1].load < 0.5 and max(g.mem for g in gpu) > warm[0].mem * 2:
            issues.append(
                Issue(
                    "STARTUP_WARMUP",
                    "info",
                    0.0,
                    warmup_end,
                    f"GPU load low while memory ramps to {max(g.mem for g in gpu):.0f} MiB",
                )
            )

    buckets = _bucket(gpu)
    if buckets:
        peak_mem = max(b[2] for b in buckets)

        # --- mid-run load dips with flat memory (epoch/dataloader stall) ---
        for i in range(1, len(buckets) - 1):
            start, end, mem, load, _ = buckets[i]
            if end <= 10 or start > duration - 10:
                continue
            prev_load = buckets[i - 1][3]
            mem_rng = max(b[2] for b in buckets[i - 1 : i + 2]) - min(b[2] for b in buckets[i - 1 : i + 2])
            load_drop = prev_load - load
            if prev_load > 0.65 and load_drop > 0.2 and load < prev_load * 0.85 and mem_rng < 800 and peak_mem > 5000:
                issues.append(
                    Issue(
                        "LOAD_DIP",
                        "warn",
                        start,
                        end,
                        f"avg load {load:.2f} (was {prev_load:.2f}) with flat memory ~{mem:.0f} MiB — likely dataloader/epoch stall",
                    )
                )

        # --- mid-run memory cliffs while still training ---
        for i in range(1, len(buckets) - 1):
            start, end, mem, load, _ = buckets[i]
            if end <= 10 or start > duration - 10:
                continue
            prev_mem = buckets[i - 1][2]
            if prev_mem > 5000 and mem < prev_mem * 0.65 and load > 0.4:
                issues.append(
                    Issue(
                        "MEM_CLIFF_MID",
                        "warn",
                        start,
                        end,
                        f"memory {prev_mem:.0f}->{mem:.0f} MiB while load={load:.2f} — not a simple dataloader stall",
                    )
                )

        # --- teardown at end ---
        last = buckets[-1]
        if duration > 20 and peak_mem > 5000 and last[2] < peak_mem * 0.25:
            issues.append(
                Issue(
                    "TEARDOWN",
                    "info",
                    last[0],
                    duration,
                    f"memory dropped {peak_mem:.0f}->{last[2]:.0f} MiB at benchmark exit",
                )
            )

        # --- IO bound sustained low load ---
        mid = [b for b in buckets if 15 <= b[0] <= duration - 10]
        if mid and peak_mem > 5000:
            low = [b for b in mid if b[3] < 0.55]
            if len(low) >= max(2, len(mid) // 2):
                issues.append(
                    Issue(
                        "IO_BOUND",
                        "warn",
                        mid[0][0],
                        mid[-1][1],
                        f"sustained low GPU load (<{0.55:.2f}) on {len(low)}/{len(mid)} windows — CPU/dataloader bound",
                    )
                )

    # --- dataloader re-init from __iter__ timing ---
    slow_iters = [(t, v) for t, v in iters if v > 5.0]
    for t, v in slow_iters:
        issues.append(
            Issue(
                "EPOCH_ITER",
                "warn" if v > 15 else "info",
                max(0.0, t - v),
                t,
                f"dataloader __iter__ took {v:.1f}s — epoch boundary / worker respawn",
            )
        )

    # --- slow epoch iter inferred from rate gap with flat mem ---
    for i in range(1, len(rates)):
        dt = rates[i][0] - rates[i - 1][0]
        if dt <= 15:
            continue
        t = rates[i - 1][0]
        if t < 15 or t > duration - 10:
            continue
        near = [g for g in gpu if abs(g.t - t) < 5]
        if near:
            mem_rng = max(g.mem for g in near) - min(g.mem for g in near)
            if mem_rng < 800:
                issues.append(
                    Issue(
                        "RATE_GAP",
                        "info",
                        rates[i - 1][0],
                        rates[i][0],
                        f"no rate samples for {dt:.0f}s with flat memory — training loop blocked (often epoch sync)",
                    )
                )

    return _dedupe_issues(issues)


def _dedupe_issues(issues: list[Issue]) -> list[Issue]:
    seen: set[tuple] = set()
    out: list[Issue] = []
    for issue in issues:
        key = (issue.kind, round(issue.t_start, 1), issue.message[:60])
        if key in seen:
            continue
        seen.add(key)
        out.append(issue)
    out.sort(key=lambda i: (i.t_start, SEVERITY_ORDER[i.severity]))
    return out


def analyze_pack(path: Path) -> PackReport:
    gpu, rates, iters, format_errors, _ = _parse_data_file(path)
    name = path.stem
    duration = gpu[-1].t if gpu else (rates[-1][0] if rates else 0.0)
    issues = _detect_issues(gpu, rates, iters, format_errors)
    peak = max((g.mem for g in gpu), default=0.0)
    return PackReport(
        name=name,
        duration=duration,
        n_gpudata=len(gpu),
        n_rates=len(rates),
        n_iter=len(iters),
        n_format_errors=format_errors,
        peak_mem_mib=peak,
        issues=issues,
    )


def analyze_run(folder: Path, *, select: str | None = None) -> RunReport:
    files = sorted(folder.glob("*.data"))
    if select:
        files = [f for f in files if select in f.name]
    packs = [analyze_pack(f) for f in files]
    packs.sort(key=lambda p: p.name)
    return RunReport(folder=folder, packs=packs)


def iter_issues(report: RunReport, min_severity: str = "info") -> Iterator[tuple[str, Issue]]:
    floor = SEVERITY_ORDER[min_severity]
    for pack in report.packs:
        for issue in pack.issues:
            if SEVERITY_ORDER[issue.severity] >= floor:
                yield pack.name, issue
