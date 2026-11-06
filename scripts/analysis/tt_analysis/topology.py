"""Leaf switches from ``scontrol show topology`` and placement of a node set."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path


def expand_hostlist(spec: str) -> list[str]:
    """Slurm hostlist -> names: ``rg[21701-21703,21801]`` -> rg21701, rg21702, ..."""
    hosts = []
    for part in re.findall(r"[^,\[]+(?:\[[^\]]*\])?", spec):
        part = part.strip(",")
        m = re.fullmatch(r"(.*)\[(.*)\](.*)", part)
        if not m:
            if part:
                hosts.append(part)
            continue
        prefix, ranges, suffix = m.groups()
        for r in ranges.split(","):
            lo, _, hi = r.partition("-")
            for i in range(int(lo), int(hi or lo) + 1):
                hosts.append(f"{prefix}{i:0{len(lo)}d}{suffix}")
    return hosts


def parse_topology(text: str) -> dict[str, str]:
    """Host -> leaf switch (Level=0 switches)."""
    leaf_of = {}
    for line in text.splitlines():
        fields = dict(f.split("=", 1) for f in line.split() if "=" in f)
        if fields.get("Level") == "0" and "Nodes" in fields:
            for host in expand_hostlist(fields["Nodes"]):
                leaf_of[host] = fields["SwitchName"]
    return leaf_of


def load_topology(path: Path) -> dict[str, str]:
    return parse_topology(path.read_text()) if path.exists() else {}


def compact(nodes_per_leaf: str) -> str:
    """"3+3+3" -> "3×3"; "12+8+7+3+3" -> "12+8+7+2×3" (runs of equal counts as leaves × nodes)."""
    parts, out = nodes_per_leaf.split("+"), []
    i = 0
    while i < len(parts):
        j = i
        while j < len(parts) and parts[j] == parts[i]:
            j += 1
        out.append(f"{j - i}×{parts[i]}" if j - i > 1 else parts[i])
        i = j
    return "+".join(out)


def placement(hosts: list[str], leaf_of: dict[str, str]) -> dict:
    """Leaves used, nodes per leaf (largest first, e.g. "3+3") and switch count."""
    counts = Counter(leaf_of.get(h, "?") for h in hosts)
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    # Leaf size = nodes of the same kind (host-name prefix) under that leaf, e.g. GPU nodes.
    prefix = re.match(r"[A-Za-z]*", hosts[0]).group(0) if hosts else ""
    size = Counter(l for h, l in leaf_of.items() if h.startswith(prefix)) if prefix else Counter()
    return {
        "leaves": ";".join(leaf for leaf, _ in ordered),
        "nodes_per_leaf": "+".join(str(n) for _, n in ordered),
        "leaf_sizes": "+".join(str(size.get(leaf, 0)) for leaf, _ in ordered),
        "n_switches": len(counts),
    }
