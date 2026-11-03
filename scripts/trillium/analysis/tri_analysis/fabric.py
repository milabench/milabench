"""Node and fabric checks (D1-D4) -> ``node_health``, ``pairs``, ``collectives`` rows.

- D1: srygaard/slurm-healthcheck results directory (read with the Rorqual parser), plus the
  Trillium additions in ``nodes/<host>.json``: per-GPU matmul and per-HCA loopback.
- D2: ``nccl/<job>/pairs.jsonl`` (``trillium.pairs.v1``), one line per node pair.
- D3, D4: ``nccl/<job>/collectives.jsonl`` (``trillium.collectives.v1``), one line per
  (collective, nodes, size) or per bisection cut.
"""

from __future__ import annotations

import json
from pathlib import Path

from tt_analysis.health import parse_srygaard

from .machine import CLUSTER, SUBNET_OF


def _jsonl(path: Path):
    for line in path.read_text().splitlines():
        if line.strip():
            yield json.loads(line)


def parse_health(run: Path, synthetic=False) -> list[dict]:
    nodes, _ = parse_srygaard(run, CLUSTER, SUBNET_OF, synthetic=synthetic)
    # srygaard's loopback is one number per node; per-HCA numbers come from nodes/<host>.json.
    nodes = [n for n in nodes if n["metric"] != "loopback_busbw_gbs"]
    base = {"cluster": CLUSTER, "source": "srygaard", "job_id": run.name, "status": "ok", "reason": None,
            "synthetic": synthetic}
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
