"""Node and fabric checks (D1-D4) -> ``node_health``, ``pairs``, ``collectives`` rows.

- D1: srygaard/slurm-healthcheck results directory (read with the Rorqual parser), plus the
  Trillium additions in ``nodes/<host>.json``: per-GPU matmul and per-HCA loopback.
- D2: ``nccl/<job>/pairs.jsonl`` (``trillium.pairs.v1``), one line per node pair.
- D3, D4: ``nccl/<job>/collectives.jsonl`` (``trillium.collectives.v1``), one line per
  (collective, nodes, size) or per bisection cut.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .machine import CLUSTER, SUBNET_OF


def _jsonl(path: Path):
    for line in path.read_text().splitlines():
        if line.strip():
            yield json.loads(line)


# The health check on real nodes (jobs in $SCRATCH/slurm-healthcheck/results, DESIGN.md "Health checks"). Which job measured what:
# the nodes before the admins' kernel change, after it, and the tests of the capped nodes. 1238183 was the first try.
# The 1249... and 1250... "after" jobs are the nodes capped or out until Nov 6, once back.
HEALTH_PERIOD = {"1238238": "before", "1238239": "before", "1238240": "before", "1238571": "before",
                 "1245968": "after", "1246233": "after", "1246278": "after",
                 **{str(j): "after" for j in (1249612, 1249619, 1249628, 1249634, 1249662, 1249664, 1250766, 1250767, 1250775, 1250936)},
                 **{str(j): "caps" for j in (1246142, 1246143, 1246144, 1246147, 1246148, 1246149, 1246150, 1246151)},
                 "1238183": "first try"}
_DCGM_FLOPS = re.compile(r"GPU (\d+) calculated at approximately ([\d.]+) gigaflops")


def _groups(run: Path, base: dict) -> list[dict]:
    """NCCL all-reduce per node: within the node (intra), through the NICs (loopback), and with its partner (pair).
    The directory's report.json is not used: one job (1246278) has none, and its verdicts name H100 limits."""
    rows = []
    for path in sorted((run / "groups").glob("*.json")):
        for t in json.loads(path.read_text()).get("tests", []):
            if t["kind"] not in ("intra", "loopback", "pair"):
                continue
            for host in t["nodes"]:
                rows.append({**base, "status": t.get("status", "ok"), "host": host, "gpu": None, "metric": f"{t['kind']}_busbw_gbs",
                             "value": t.get("peak_busbw")})
    return rows


def parse_health(run: Path, synthetic=False) -> list[dict]:
    period = "synthetic" if synthetic else HEALTH_PERIOD.get(run.name, "other")
    base = {"cluster": CLUSTER, "source": "srygaard", "job_id": run.name, "status": "ok", "reason": None, "synthetic": synthetic,
            "period": period}
    nodes = _groups(run, base)
    for path in sorted((run / "nodes").glob("*.json")):
        d = json.loads(path.read_text())
        host = d.get("host", path.stem)
        matmul, hca = d.get("matmul") or {}, d.get("hca") or {}
        for g in (matmul.get("data") or {}).get("gpus", []):
            for dtype in ("bf16", "fp8"):
                if g.get(f"{dtype}_tflops") is not None:
                    nodes.append({**base, "status": matmul.get("status", "ok"), "host": host, "gpu": g["gpu"],
                                  "metric": f"matmul_{dtype}_tflops", "value": g[f"{dtype}_tflops"]})
        for h in (hca.get("data") or {}).get("hcas", []):
            nodes.append({**base, "status": hca.get("status", "ok"), "host": host, "gpu": None, "hca": h["hca"],
                          "rate_gbps": h.get("rate_gbps"), "metric": "hca_loopback_gbs", "value": h["loopback_gbs"]})
        # DCGM's diagnostic: FP16 FLOP/s per GPU; every test passed or not, per GPU.
        tests = ((d.get("dcgm") or {}).get("data") or {}).get("tests") or {}
        for gpu, flops in _DCGM_FLOPS.findall(" ".join((tests.get("diagnostic") or {}).get("messages") or [])):
            nodes.append({**base, "host": host, "gpu": int(gpu), "metric": "dcgm_tflops", "value": float(flops) / 1000})
        for name, test in tests.items():
            for gpu, verdict in (test.get("entities") or {}).items():
                nodes.append({**base, "host": host, "gpu": int(gpu), "status": verdict, "metric": f"dcgm_{name}", "value": float(verdict == "Pass")})
        # CPU clocks under full GPU power, with load on 72 of the 96 cores.
        cpu = (d.get("cpu_load") or {}).get("data") or {}
        if cpu.get("full_power_mhz") is not None:
            nodes += [{**base, "host": host, "gpu": None, "metric": "cpu_mhz_full_power", "value": cpu["full_power_mhz"]},
                      {**base, "host": host, "gpu": None, "metric": "cpu_low_share_full_power", "value": cpu.get("low_share")},
                      {**base, "host": host, "gpu": None, "metric": "gpu_kw_peak", "value": max((s.get("gpu_w") or 0 for s in cpu.get("samples", [])), default=0) / 1000}]
    for n in nodes:
        n["leaf"] = SUBNET_OF.get(n["host"], "?")
    return nodes


def parse_pairs(path: Path, synthetic=False) -> list[dict]:
    rows = []
    for r in _jsonl(path):
        r = {k: v for k, v in r.items() if k != "schema"}
        rows.append({**r, "subnet_a": SUBNET_OF.get(r["a"], "?"), "subnet_b": SUBNET_OF.get(r["b"], "?"),
                     "cluster": CLUSTER, "synthetic": synthetic})
    return rows


def parse_collectives(path: Path, synthetic=False) -> list[dict]:
    rows = []
    for r in _jsonl(path):
        pairs = r.pop("pairs", None) or []
        r.pop("halves", None)
        r.pop("hosts", None)
        r.pop("schema", None)
        r["min_pair_gbs"] = min((p["bw_gbs"] for p in pairs), default=None)
        rows.append({**r, "cluster": CLUSTER, "synthetic": synthetic})
    return rows
