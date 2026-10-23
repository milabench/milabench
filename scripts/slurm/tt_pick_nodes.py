#!/usr/bin/env python3
"""Pick node lists for the topology study from the healthy nodes.

    tt_pick_nodes.py --family H --nodes 6 --candidates "$(cat healthy.txt)"
    tt_pick_nodes.py --family S --nodes 6 --leaf ihpc316u25 --candidates ...
    tt_pick_nodes.py --family P --nodes 15 --candidates ... --exclude rg21701

Families (journals/2026-10-23.md, "Rorqual topology"):
    H  3 nodes per leaf switch on N/3 switches (switches with < 3 usable nodes
       are skipped; leaves with exactly 3 GPU nodes are preferred, then big
       leaves contribute 3 nodes each).
    S  all N nodes on one switch (--leaf, or the switch with the most usable
       nodes); N must be a multiple of 3.
    P  fewest switches: fill the switches with the most usable nodes first.

A node is usable if it is in --candidates (healthy nodes, e.g. from
tt_healthcheck.sh, or a reservation's node list) and not in --exclude.
Prints the hostlist (comma-separated) on stdout and the leaf map on stderr.
Exits with status 2 if the request can't be satisfied.
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def hostnames(spec: str) -> list[str]:
    if not spec:
        return []
    out = subprocess.run(
        ["scontrol", "show", "hostnames", spec], capture_output=True, text=True, check=True
    )
    return out.stdout.split()


def leaves() -> dict[str, list[str]]:
    """Leaf switch -> its nodes (Level=0 switches of scontrol show topology)."""
    topo = subprocess.run(
        ["scontrol", "show", "topology"], capture_output=True, text=True, check=True
    ).stdout
    result: dict[str, list[str]] = {}
    for line in topo.splitlines():
        fields = dict(f.split("=", 1) for f in line.split() if "=" in f)
        if fields.get("Level") == "0" and "Nodes" in fields:
            result[fields["SwitchName"]] = hostnames(fields["Nodes"])
    return result


def usable_by_leaf(candidates: set[str], exclude: set[str], prefix: str) -> dict[str, list[str]]:
    by_leaf = {}
    for leaf, nodes in leaves().items():
        ok = [n for n in nodes if n.startswith(prefix) and n in candidates and n not in exclude]
        if ok:
            by_leaf[leaf] = sorted(ok)
    return by_leaf


def leaf_sizes(prefix: str) -> dict[str, int]:
    return {leaf: sum(n.startswith(prefix) for n in nodes) for leaf, nodes in leaves().items()}


def pick(family: str, n: int, by_leaf: dict[str, list[str]], leaf: str | None, prefix: str):
    if family == "H":
        if n % 3:
            raise ValueError("H needs a multiple of 3 nodes")
        sizes = leaf_sizes(prefix)
        ok = [l for l, nodes in by_leaf.items() if len(nodes) >= 3]
        # Small (3-node) leaves first, then big leaves; stable by name.
        ok.sort(key=lambda l: (sizes.get(l, 0) != 3, l))
        if len(ok) < n // 3:
            raise ValueError(f"H{n // 3}: only {len(ok)} switches with 3 usable nodes")
        return {l: by_leaf[l][:3] for l in ok[: n // 3]}
    if family == "S":
        if n % 3:
            raise ValueError("S needs a multiple of 3 nodes")
        target = leaf or max(by_leaf, key=lambda l: (len(by_leaf[l]), l))
        if len(by_leaf.get(target, [])) < n:
            raise ValueError(f"S{n}: {target} has {len(by_leaf.get(target, []))} usable nodes")
        return {target: by_leaf[target][:n]}
    if family == "P":
        chosen, left = {}, n
        for l in sorted(by_leaf, key=lambda l: (-len(by_leaf[l]), l)):
            if left == 0:
                break
            take = by_leaf[l][: min(left, len(by_leaf[l]))]
            chosen[l] = take
            left -= len(take)
        if left:
            raise ValueError(f"P{n}: only {n - left} usable nodes")
        return chosen
    raise ValueError(f"unknown family {family!r}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--family", required=True, choices=["H", "S", "P"])
    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument("--candidates", required=True, help="usable nodes (hostlist or comma/space separated)")
    parser.add_argument("--exclude", default="", help="nodes to leave out (hostlist)")
    parser.add_argument("--leaf", help="switch for family S")
    parser.add_argument("--prefix", default="rg", help="GPU node name prefix")
    args = parser.parse_args(argv)

    candidates = set(hostnames(",".join(args.candidates.split())))
    exclude = set(hostnames(",".join(args.exclude.split())))
    by_leaf = usable_by_leaf(candidates, exclude, args.prefix)
    try:
        chosen = pick(args.family, args.nodes, by_leaf, args.leaf, args.prefix)
    except ValueError as exc:
        print(f"tt_pick_nodes: {exc}", file=sys.stderr)
        return 2
    for leaf, nodes in chosen.items():
        print(f"# {leaf}: {','.join(nodes)}", file=sys.stderr)
    print(",".join(n for nodes in chosen.values() for n in nodes))
    return 0


if __name__ == "__main__":
    sys.exit(main())
