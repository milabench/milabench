"""Health checks and NCCL tests -> ``node_health`` and ``comm`` rows.

Two sources:
- srygaard/slurm-healthcheck run directories (the window's health check):
  ``report.json``, ``groups/<job>.json`` (nccl-tests; optional per-size
  ``rows``, see FORMAT.md).
- tt_healthcheck.sh directories (``health/<job>/raw.log``, ``excluded.txt``),
  used before the window.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

COLLECTIVE_OF_TEST = {"all_reduce_perf": "all_reduce", "alltoall_perf": "alltoall",
                      "all_gather_perf": "all_gather", "reduce_scatter_perf": "reduce_scatter"}

from .topology import placement

COLLECTIVE = {"intra": "all_reduce", "loopback": "all_reduce", "pair": "all_reduce",
              "group": "all_reduce", "alltoall": "alltoall"}


def _comm_row(cluster, source, job, kind, hosts, leaf_of, busbw, status="ok", size=None, synthetic=False,
              collective=None, condition=None):
    return {
        "cluster": cluster, "source": source, "job_id": str(job), "kind": kind,
        "collective": collective or COLLECTIVE.get(kind, "all_reduce"), "condition": condition,
        "nodes": len(hosts),
        "hosts": ";".join(hosts), **placement(hosts, leaf_of),
        "size_bytes": size, "busbw_gbs": busbw, "status": status, "synthetic": synthetic,
    }


def parse_srygaard(run: Path, cluster: str, leaf_of: dict[str, str], synthetic=False):
    """One srygaard results directory -> (node rows, comm rows)."""
    report = json.loads((run / "report.json").read_text())
    switches = {}
    if (run / "switches.json").exists():
        for leaf, hosts in json.loads((run / "switches.json").read_text()).items():
            switches.update({h: leaf for h in hosts})
    leaf_of = {**leaf_of, **switches}
    job = run.name
    nodes, comm = [], []
    for host, n in report["nodes"].items():
        nodes.append({
            "cluster": cluster, "source": "srygaard", "job_id": job, "host": host,
            "leaf": n.get("switch") or leaf_of.get(host, "?"), "gpu": None,
            "metric": "verdict", "value": None, "status": n["status"],
            "reason": "; ".join(f["detail"] for f in n.get("findings", [])) or None,
            "synthetic": synthetic,
        })
    for path in sorted((run / "groups").glob("*.json")):
        group = json.loads(path.read_text())
        for t in group["tests"]:
            status = t.get("status", "ok")
            if t["kind"] in ("intra", "loopback", "pair"):
                for host in t["nodes"]:
                    nodes.append({
                        "cluster": cluster, "source": "srygaard", "job_id": job, "host": host,
                        "leaf": leaf_of.get(host, "?"), "gpu": None,
                        "metric": f"{t['kind']}_busbw_gbs", "value": t["peak_busbw"],
                        "status": status, "reason": None, "synthetic": synthetic,
                    })
            comm.append(_comm_row(cluster, "srygaard", job, t["kind"], t["nodes"], leaf_of,
                                  t["peak_busbw"], status, synthetic=synthetic))
            for r in t.get("rows", []):
                comm.append(_comm_row(cluster, "srygaard", job, t["kind"], t["nodes"], leaf_of,
                                      r["busbw"], status, size=r["size"], synthetic=synthetic))
    return nodes, comm


def parse_tt_health(job_dir: Path, cluster: str, leaf_of: dict[str, str]):
    """One tt_healthcheck.sh directory -> (node rows, comm rows, excluded {host: reason})."""
    raw = job_dir / "raw.log"
    job = job_dir.name
    nodes, comm, hosts = [], [], set()
    if not raw.exists():
        return nodes, comm, {}
    seen_all = set()
    for line in raw.read_text(errors="replace").splitlines():
        kv = dict(re.findall(r"(\w+)=(\S+)", line))
        if "[tt-health] matmul" in line:
            hosts.add(kv["host"])
            nodes.append({
                "cluster": cluster, "source": "tt_health", "job_id": job, "host": kv["host"],
                "leaf": leaf_of.get(kv["host"], "?"), "gpu": int(kv["gpu"]),
                "metric": "matmul_tflops", "value": float(kv["tflops"]), "status": "ok",
                "reason": None, "synthetic": False,
            })
        elif "[tt-health] allreduce" in line and kv.get("scope", "").startswith("node:"):
            host = kv["scope"][5:]
            nodes.append({
                "cluster": cluster, "source": "tt_health", "job_id": job, "host": host,
                "leaf": leaf_of.get(host, "?"), "gpu": None, "metric": "intra_busbw_gbs",
                "value": float(kv["busbw_gbs"]), "status": "ok", "reason": None, "synthetic": False,
            })
            comm.append(_comm_row(cluster, "tt_health", job, "intra", [host], leaf_of,
                                  float(kv["busbw_gbs"]), size=int(kv["size_mb"]) * 2**20))
        elif "[tt-health] allreduce scope=all" in line and "busbw_gbs" in kv and job not in seen_all:
            seen_all.add(job)  # tee'd twice into raw.log
            all_hosts = sorted(hosts)
            comm.append(_comm_row(cluster, "tt_health", job, "group", all_hosts, leaf_of,
                                  float(kv["busbw_gbs"]), size=int(kv["size_mb"]) * 2**20))
    excluded = {}
    if (job_dir / "excluded.txt").exists():
        for line in (job_dir / "excluded.txt").read_text().splitlines():
            host, _, reason = line.partition("\t")
            if host:
                excluded[host] = reason
    for host, reason in excluded.items():
        nodes.append({
            "cluster": cluster, "source": "tt_health", "job_id": job, "host": host,
            "leaf": leaf_of.get(host, "?"), "gpu": None, "metric": "verdict", "value": None,
            "status": "fail", "reason": reason, "synthetic": False,
        })
    return nodes, comm, excluded


def parse_nccl_tests(path: Path, cluster: str, leaf_of: dict[str, str]):
    """One nccl-tests stdout file -> comm rows (peak, then one row per size; out-of-place busbw).

    Layout: ``nccl_tests/<job>/<condition>-<host>-<host>....txt``; the condition (e.g. "alone",
    "concurrent") is the first dash-separated word of the name when it is not a host.
    """
    text = path.read_text(errors="replace")
    test = re.search(r"Collective test starting: (\w+)", text)
    collective = COLLECTIVE_OF_TEST.get(test.group(1), test.group(1)) if test else "all_reduce"
    hosts = sorted(set(re.findall(r"Rank\s+\d+ Group\s+\d+ Pid\s+\d+ on\s+(\S+)", text)))
    rows = []
    columns = []
    for line in text.splitlines():
        tokens = line.split()
        if tokens[:2] == ["#", "size"]:
            columns = tokens[1:]
        elif columns and tokens and tokens[0].isdigit() and len(tokens) == len(columns):
            rows.append((int(tokens[0]), float(tokens[columns.index("busbw")])))
    if not hosts or not rows:
        return []
    first = path.stem.split("-")[0]
    condition = first if first not in hosts and not first.startswith(tuple(h[:2] for h in hosts)) else "alone"
    kind = "pair" if len(hosts) == 2 else ("intra" if len(hosts) == 1 else "group")
    job = path.parent.name
    out = [_comm_row(cluster, "nccl_tests", job, kind, hosts, leaf_of, max(bw for _, bw in rows),
                     collective=collective, condition=condition)]
    out += [_comm_row(cluster, "nccl_tests", job, kind, hosts, leaf_of, bw, size=size,
                      collective=collective, condition=condition) for size, bw in rows]
    return out
