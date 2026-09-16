"""Human-readable reporting for diagnostic analysis."""

from __future__ import annotations

import json
from collections import Counter

from .analyze import SEVERITY_ORDER, RunReport

KIND_HINTS = {
    "GPUDATA_GAP": "Monitor thread gap — no NVML samples",
    "JOB_RESTART": "Long gap + memory baseline — job preempted/restarted",
    "LOAD_DIP": "GPU idle with flat memory — dataloader/epoch stall",
    "MEM_CLIFF_MID": "Memory freed mid-run while GPU active",
    "TEARDOWN": "Normal process exit at benchmark end",
    "STARTUP_WARMUP": "Model/dataloader warmup before steady state",
    "EPOCH_ITER": "Slow DataLoader __iter__ — epoch boundary",
    "RATE_GAP": "Rate metrics batched — main thread blocked at sync point",
    "IO_BOUND": "GPU under-utilized — likely CPU/IO bound",
    "TIMESTAMP_REGRESSION": "Non-monotonic timestamps — merged/interleaved writers",
    "FORMAT_ERROR": "Unparseable metric lines in .data file",
    "NO_METRICS": "Empty or missing metrics",
}


def print_report(
    report: RunReport,
    *,
    min_severity: str = "info",
    show_clean: bool = False,
    json_out: bool = False,
) -> None:
    if json_out:
        print(json.dumps(_to_json(report, min_severity=min_severity), indent=2))
        return

    print(f"Run folder: {report.folder}")
    print(f"Packs analyzed: {len(report.packs)}")

    counts = Counter()
    flagged = 0
    for pack in report.packs:
        visible = [i for i in pack.issues if SEVERITY_ORDER[i.severity] >= SEVERITY_ORDER[min_severity]]
        if visible:
            flagged += 1
        for issue in visible:
            counts[issue.kind] += 1

    print(f"Packs with issues (>={min_severity}): {flagged}/{len(report.packs)}")
    if counts:
        summary = ", ".join(f"{k}={v}" for k, v in counts.most_common())
        print(f"Issue counts: {summary}")
    print()

    for pack in report.packs:
        visible = [i for i in pack.issues if SEVERITY_ORDER[i.severity] >= SEVERITY_ORDER[min_severity]]
        if not visible and not show_clean:
            continue

        print(f"## {pack.name}  ({pack.duration:.0f}s, gpudata={pack.n_gpudata}, rates={pack.n_rates})")
        if pack.peak_mem_mib:
            print(f"   peak memory: {pack.peak_mem_mib:.0f} MiB")
        if not visible:
            print("   (clean)")
            print()
            continue

        for issue in visible:
            span = _format_span(issue.t_start, issue.t_end)
            hint = KIND_HINTS.get(issue.kind, "")
            print(f"   [{issue.severity:5}] {issue.kind:18} {span}")
            print(f"          {issue.message}")
            if hint:
                print(f"          → {hint}")
        print()


def _format_span(t_start: float, t_end: float | None) -> str:
    if t_end is None or abs(t_end - t_start) < 0.5:
        return f"@ {t_start:6.1f}s"
    return f"@ {t_start:6.1f}s–{t_end:6.1f}s"


def _to_json(report: RunReport, *, min_severity: str) -> dict:
    packs = []
    for pack in report.packs:
        issues = [
            {
                "kind": i.kind,
                "severity": i.severity,
                "t_start": i.t_start,
                "t_end": i.t_end,
                "message": i.message,
            }
            for i in pack.issues
            if SEVERITY_ORDER[i.severity] >= SEVERITY_ORDER[min_severity]
        ]
        packs.append(
            {
                "name": pack.name,
                "duration": pack.duration,
                "n_gpudata": pack.n_gpudata,
                "n_rates": pack.n_rates,
                "n_iter": pack.n_iter,
                "n_format_errors": pack.n_format_errors,
                "peak_mem_mib": pack.peak_mem_mib,
                "issues": issues,
            }
        )
    return {"folder": str(report.folder), "packs": packs}
