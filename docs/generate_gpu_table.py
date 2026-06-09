"""Fetch GPU summary from the milabench dashboard and generate an RST table."""

import json
import os
import pathlib
import urllib.request
import urllib.error

DOCS_DIR = pathlib.Path(__file__).parent
OUTPUT = DOCS_DIR / "_gpu_summary.rst"

DASHBOARD_URL = os.environ.get(
    "MILABENCH_DASHBOARD_URL",
    "https://www.milabench.com",
)
API_ENDPOINT = f"{DASHBOARD_URL}/api/gpu/summary"
TIMEOUT = 10


def _strip_quotes(s):
    if not s:
        return "—"
    return str(s).strip('"')


def _short_tag(s):
    """v1.1.1-39-g6cfebd4 -> 1.1.1"""
    tag = _strip_quotes(s)
    parts = tag.split("-")
    if len(parts) >= 3 and parts[-1].startswith("g"):
        tag = "-".join(parts[:-2])
    return tag.lstrip("v")


def _format_date(iso):
    if not iso:
        return "-"
    return iso[:10]


def _pass_rate_label(passed, total):
    if total == 0:
        return "—"
    pct = passed / total * 100
    return f"{pct:.0f}%"


def _dedup_key(row):
    return (
        _strip_quotes(row.get("gpu", "")),
        row.get("cpu_arch", ""),
    )


def dedup_rows(rows):
    """Keep only the newest milabench version per (GPU, cpu_arch)."""
    seen = {}
    for row in rows:
        key = _dedup_key(row)
        existing = seen.get(key)
        if existing is None or (row.get("latest_date") or "") > (existing.get("latest_date") or ""):
            seen[key] = row
    return list(seen.values())


def fetch_gpu_summary():
    try:
        req = urllib.request.Request(API_ENDPOINT, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return json.loads(resp.read().decode())
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        print(f"[generate_gpu_table] Could not fetch GPU summary: {exc}")
        return None


def build_rst(rows):
    lines = []
    lines.append("Latest GPU Runs")
    lines.append("-" * len(lines[-1]))
    lines.append("")
    lines.append("GPUs milabench has been ran on, with the latest run results.")
    lines.append("")

    deduped = dedup_rows(rows)

    lines.append(".. list-table::")
    lines.append("   :header-rows: 1")
    lines.append("   :widths: 30 15 10 12 12 6 12")
    lines.append("")
    lines.append("   * - GPU")
    lines.append("     - GPUs")
    lines.append("     - CPU")
    lines.append("     - PyTorch")
    lines.append("     - Version")
    lines.append("     - Passed")
    lines.append("     - ")

    for row in deduped:
        gpu = _strip_quotes(row.get("gpu"))
        gpu_count = row.get("gpu_count", "?")
        gpu_mem = row.get("gpu_memory")
        mem_str = f" ({round(gpu_mem / 1024)} GB)" if gpu_mem else ""
        cpu_arch = row.get("cpu_arch", "—")
        pytorch = _strip_quotes(row.get("pytorch"))
        tag = _short_tag(row.get("milabench_tag"))
        passed = row.get("passed", 0)
        total = row.get("total", 0)
        exec_id = row.get("exec_id")
        report_link = f"`Report <{DASHBOARD_URL}/executions/{exec_id}?report=sql>`_" if exec_id else "—"

        lines.append(f"   * - {gpu}")
        lines.append(f"     - {gpu_count}x{mem_str}")
        lines.append(f"     - {cpu_arch}")
        lines.append(f"     - {pytorch}")
        lines.append(f"     - {tag}")
        lines.append(f"     - {_pass_rate_label(passed, total)}")
        lines.append(f"     - {report_link}")

    lines.append("")
    return "\n".join(lines)


def build_fallback():
    lines = []
    lines.append("Latest GPU Runs")
    lines.append("-" * len(lines[-1]))
    lines.append("")
    lines.append(
        "GPU run data is currently unavailable. "
        "Visit the `milabench dashboard <https://www.milabench.com>`_ for live results."
    )
    lines.append("")
    return "\n".join(lines)


def main():
    rows = fetch_gpu_summary()
    if rows:
        rst = build_rst(rows)
        print(f"[generate_gpu_table] Fetched {len(rows)} GPU entries")
    else:
        rst = build_fallback()
        print("[generate_gpu_table] Using fallback (API unavailable)")

    OUTPUT.write_text(rst)
    print(f"[generate_gpu_table] Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
