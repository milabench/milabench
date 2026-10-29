"""Topology-aware placement for the megatron pack.

Two knobs decide which GPU talks to which:

* the **node order**: torchrun's node rank is the position in the node list, and Megatron then maps global ranks
  to (tp, cp, ep, dp, pp) groups in a fixed order. Reordering the nodes (``placement.nodes``) moves whole groups
  between switches without touching the model layout.
* the **group order** (``placement.order``): ``tp-cp-ep-dp-pp`` (Megatron default, pipeline stages are far apart)
  or ``tp-cp-ep-pp-dp`` (``--use-tp-pp-dp-mapping``, consecutive ranks hold consecutive pipeline stages).

``describe`` simulates the resulting groups and reports how many nodes / switches each one spans, so a layout can be
checked (and compared) before spending cluster time on it.

Config (``placement:`` in the bench entry)::

    placement:
      nodes: as-is | sorted | topology | [explicit, node, list]
      order: tp-cp-ep-dp-pp | tp-cp-ep-pp-dp
      topology: {node-name: switch-or-rack}   # optional, else node["switch"], else `scontrol show topology`
      gpus_per_node: 4
"""

from __future__ import annotations

import os
import re
import subprocess
from collections import OrderedDict, defaultdict

ORDERS = ("tp-cp-ep-dp-pp", "tp-cp-ep-pp-dp")


# --- switch discovery --------------------------------------------------------------------------------------------

def expand_hostlist(expr: str) -> list[str]:
    """``rc[10101-10103,10110],rg1`` -> hostnames (Slurm hostlist syntax, one bracket group per name)."""
    names = []
    for part in re.findall(r"[^,\[]+(?:\[[^\]]*\])?[^,\[]*", expr):
        m = re.match(r"^(.*?)\[([^\]]*)\](.*)$", part)
        if not m:
            names.append(part)
            continue
        prefix, ranges, suffix = m.groups()
        for r in ranges.split(","):
            lo, _, hi = r.partition("-")
            width = len(lo)
            for i in range(int(lo), int(hi or lo) + 1):
                names.append(f"{prefix}{i:0{width}d}{suffix}")
    return names


def parse_scontrol_topology(text: str) -> dict[str, str]:
    """Leaf switch of every node from ``scontrol show topology`` (lines ``SwitchName=s Level=0 ... Nodes=...``)."""
    leaves = {}
    for line in text.splitlines():
        m = re.search(r"SwitchName=(\S+)\s+Level=(\d+)", line)
        nodes = re.search(r"Nodes=(\S+)", line)
        if m and nodes and m.group(2) == "0":
            for node in expand_hostlist(nodes.group(1)):
                leaves[node] = m.group(1)
    return leaves


def slurm_topology() -> dict[str, str]:
    """From the snapshot the job saved at allocation time (``TT_TOPOLOGY``) if there is one, so the node order at launch
    and the run's record use the same topology; else ``scontrol show topology`` now."""
    snapshot = os.environ.get("TT_TOPOLOGY")
    if snapshot and os.path.isfile(snapshot):
        with open(snapshot) as f:
            return parse_scontrol_topology(f.read())
    try:
        out = subprocess.run(["scontrol", "show", "topology"], capture_output=True, text=True, timeout=30, check=True)
    except (OSError, subprocess.SubprocessError):
        return {}
    return parse_scontrol_topology(out.stdout)


def switch_map(nodes: list[dict], placement: dict) -> dict[str, str]:
    """node name -> switch. Explicit config wins, then the node's own ``switch`` field, then Slurm."""
    given = dict(placement.get("topology") or {})
    result = {}
    slurm = None
    for node in nodes:
        name = node["name"]
        if name in given:
            result[name] = str(given[name])
        elif node.get("switch"):
            result[name] = str(node["switch"])
        else:
            slurm = slurm_topology() if slurm is None else slurm
            result[name] = slurm.get(name, "unknown")
    return result


# --- node order --------------------------------------------------------------------------------------------------
#
# ``placement.nodes``:
#   as-is | sorted | [explicit list]   the nodes as they come / by name / the given order
#   leaf (alias topology)              leaf switch by leaf switch, main node's leaf first, fullest leaves next
#   aligned                            like leaf, on a set with a multiple of ``group_nodes`` nodes per leaf: no group
#                                      of ``group_nodes`` consecutive nodes (an expert group) crosses a switch
#   crossed                            same set, every group takes its nodes from ``group_nodes`` different leaves
#   mixed                              aligned, but ``mixed_swaps`` pairs of one-group leaves swap partners
# ``placement.set``: all (default) | even: keep only a multiple of ``group_nodes`` nodes on every leaf (aligned,
# crossed and mixed always do). ``group_nodes`` = nodes of one expert group = EP size / GPUs per node.

def main_node(nodes: list[dict]) -> dict:
    return next((n for n in nodes if n.get("main")), nodes[0])


def by_leaf(nodes: list[dict], switches: dict[str, str]) -> "OrderedDict[str, list[dict]]":
    leaves = OrderedDict()
    for node in nodes:
        leaves.setdefault(switches[node["name"]], []).append(node)
    return leaves


def even_set(nodes: list[dict], switches: dict[str, str], g: int) -> "OrderedDict[str, list[dict]]":
    """leaf -> the kept nodes: a multiple of ``g`` on each leaf (the main node is kept first on its leaf)."""
    main = main_node(nodes)
    keep = OrderedDict()
    for leaf, members in by_leaf(nodes, switches).items():
        members = sorted(members, key=lambda n: (n["name"] != main["name"], n["name"]))
        kept = members[: len(members) // g * g]
        if kept:
            keep[leaf] = kept
    if not any(n["name"] == main["name"] for ns in keep.values() for n in ns):
        raise ValueError(f"the main node {main['name']} is on a leaf with fewer than {g} nodes: it would be dropped")
    return keep


def leaf_order(keep: "OrderedDict[str, list[dict]]", main: dict, switches: dict[str, str]) -> list[str]:
    """Main node's leaf first, then the fullest leaves."""
    main_leaf = switches[main["name"]]
    return sorted(keep, key=lambda leaf: (leaf != main_leaf, -len(keep[leaf]), leaf))


def order_aligned(keep, main, switches) -> list[dict]:
    return [n for leaf in leaf_order(keep, main, switches) for n in keep[leaf]]


def order_crossed(keep, main, switches, g: int) -> list[dict]:
    """Every group of ``g`` consecutive nodes takes one node from each of ``g`` different leaves."""
    pools = OrderedDict((leaf, list(ns)) for leaf, ns in keep.items())
    main_leaf = switches[main["name"]]
    ordered = []
    while sum(map(len, pools.values())) >= g:
        live = [leaf for leaf in pools if pools[leaf]]
        live.sort(key=lambda leaf: (leaf != main_leaf or bool(ordered), -len(pools[leaf]), leaf))
        if len(live) < g:
            break  # not enough distinct leaves left: the rest cannot cross
        for leaf in live[:g]:
            ordered.append(pools[leaf].pop(0))
    for leaf in pools:  # leftovers, unavoidably aligned
        ordered.extend(pools[leaf])
    return ordered


def order_mixed(keep, main, switches, g: int, swaps: int) -> list[dict]:
    """Aligned, except ``swaps`` pairs of one-group leaves swap partners: (A1 A2)(B1 B2) -> (A1 B1)(A2 B2)."""
    if g != 2:
        raise ValueError("placement.nodes: mixed is defined for groups of 2 nodes (EP8 on 4-GPU nodes)")
    leaves = leaf_order(keep, main, switches)
    slots = {leaf: list(keep[leaf]) for leaf in leaves}
    main_leaf = switches[main["name"]]
    singles = [leaf for leaf in leaves if len(keep[leaf]) == g and leaf != main_leaf]
    if 2 * swaps > len(singles):
        raise ValueError(f"mixed: {swaps} swaps need {2 * swaps} leaves with exactly {g} kept nodes, found {len(singles)}")
    for k in range(swaps):
        a, b = singles[2 * k], singles[2 * k + 1]
        (a1, a2), (b1, b2) = keep[a], keep[b]
        slots[a], slots[b] = [a1, b1], [a2, b2]
    return [n for leaf in leaves for n in slots[leaf]]


def order_nodes(nodes: list[dict], placement: dict | None) -> list[dict]:
    """Return the nodes in rank order (a subset for the ``even`` set and the aligned / crossed / mixed modes).

    The main node always comes first."""
    placement = placement or {}
    mode = placement.get("nodes", "as-is")
    if mode == "as-is" or len(nodes) <= 1:
        return list(nodes)

    main = main_node(nodes)
    rest = [n for n in nodes if n["name"] != main["name"]]
    if isinstance(mode, list):
        position = {name: i for i, name in enumerate(mode)}
        missing = [n["name"] for n in nodes if n["name"] not in position]
        if missing:
            raise ValueError(f"placement.nodes does not list {missing}")
        ordered = sorted(nodes, key=lambda n: position[n["name"]])
        if ordered[0]["name"] != main["name"]:
            raise ValueError(f"placement.nodes must start with the main node {main['name']}")
        return ordered
    if mode == "sorted":
        return [main, *sorted(rest, key=lambda n: n["name"])]

    g = int(placement.get("group_nodes", 2))
    switches = switch_map(nodes, placement)
    if mode == "topology":
        mode = "leaf"
    if mode in ("aligned", "crossed", "mixed") or placement.get("set") == "even":
        # the set keeps a multiple of set_group nodes per leaf (default: the group size; G6 keeps G1's set at EP16)
        keep = even_set(nodes, switches, int(placement.get("set_group", g)))
    else:
        keep = by_leaf(sorted(nodes, key=lambda n: (n["name"] != main["name"], n["name"])), switches)

    if mode == "leaf" or mode == "aligned":
        ordered = order_aligned(keep, main, switches)
    elif mode == "crossed":
        ordered = order_crossed(keep, main, switches, g)
    elif mode == "mixed":
        ordered = order_mixed(keep, main, switches, g, int(placement.get("mixed_swaps", 3)))
    else:
        raise ValueError(f"unknown placement.nodes {mode!r}")

    if mode == "leaf" and placement.get("set") != "even":
        ordered = ordered[: len(ordered) // g * g]  # the last, partial group is dropped
    if ordered[0]["name"] != main["name"]:
        raise ValueError(f"placement.nodes={mode}: the main node {main['name']} is not first, it cannot lead the job")
    return ordered


def crossing(ordered: list[dict], g: int, switches: dict[str, str]) -> tuple[int, int]:
    """(groups of ``g`` consecutive nodes that span more than one leaf, groups)."""
    groups = [ordered[i : i + g] for i in range(0, len(ordered) - g + 1, g)]
    return sum(1 for grp in groups if len({switches[n["name"]] for n in grp}) > 1), len(groups)


def plan(nodes: list[dict], placement: dict) -> dict:
    """The node lists the different modes give, with their crossing counts (login-node check, no GPU)."""
    out = {}
    g = int(placement.get("group_nodes", 2))
    switches = switch_map(nodes, placement)
    for mode in ("aligned", "crossed", "mixed", "leaf"):
        for variant in ("even", "all") if mode == "leaf" else ("even",):
            try:
                ordered = order_nodes(nodes, {**placement, "nodes": mode, "set": variant})
            except ValueError as exc:
                out[f"{mode}/{variant}"] = {"error": str(exc)}
                continue
            bad, total = crossing(ordered, g, switches)
            out[f"{mode}/{variant}"] = {"nodes": [n["name"] for n in ordered], "groups": total, "crossing": bad}
    return out


# --- Megatron rank groups ----------------------------------------------------------------------------------------

def rank_groups(order: str, sizes: dict[str, int], token: str) -> list[list[int]]:
    """Groups of global ranks for ``token`` (same algorithm as megatron.core's ``generate_masked_orthogonal_rank_groups``).

    ``order`` lists the parallel dimensions from the fastest-varying rank bit to the slowest."""
    dims = order.split("-")
    shape = [sizes[d] for d in dims]
    world = 1
    for s in shape:
        world *= s
    strides = {}
    stride = 1
    for d, s in zip(dims, shape):
        strides[d] = stride
        stride *= s

    free = [d for d in dims if d != token]
    groups = []
    # every combination of the other dimensions gives one group along ``token``
    def walk(i, base):
        if i == len(free):
            groups.append([base + k * strides[token] for k in range(sizes[token])])
            return
        for k in range(sizes[free[i]]):
            walk(i + 1, base + k * strides[free[i]])

    walk(0, 0)
    assert sorted(r for g in groups for r in g) == list(range(world))
    return groups


def layout_sizes(world: int, tp: int, pp: int, cp: int, ep: int, etp: int | None = None) -> tuple[dict, dict]:
    """(dense sizes, expert sizes). The expert layers have their own tp/ep and use whatever is left as expert DP."""
    etp = tp if etp is None else etp
    if world % (tp * cp * pp):
        raise ValueError(f"world {world} is not divisible by tp*cp*pp = {tp * cp * pp}")
    dense = {"tp": tp, "cp": cp, "dp": world // (tp * cp * pp), "pp": pp, "ep": 1}
    if world % (etp * ep * pp):
        raise ValueError(f"world {world} is not divisible by etp*ep*pp = {etp * ep * pp}")
    # Megatron's expert generator: cp = 1, tp = etp, "dp" = expert data parallel, ep stands between
    expert = {"tp": etp, "cp": 1, "ep": ep, "dp": world // (etp * ep * pp), "pp": pp}
    return dense, expert


def describe(nodes: list[dict], placement: dict, tp: int, pp: int, cp: int, ep: int, etp: int | None = None) -> dict:
    """Per parallel group: how many groups span more than one node / switch, and the widest one."""
    gpus = int(placement.get("gpus_per_node", 4))
    order = placement.get("order", ORDERS[0])
    if order not in ORDERS:
        raise ValueError(f"placement.order must be one of {ORDERS}")
    world = len(nodes) * gpus
    dense, expert = layout_sizes(world, tp, pp, cp, ep, etp)
    switches = switch_map(nodes, placement)

    def node_of(rank):
        return nodes[rank // gpus]["name"]

    report = OrderedDict()
    for label, sizes, token in (
        ("tp", dense, "tp"), ("cp", dense, "cp"), ("dp", dense, "dp"), ("pp", dense, "pp"),
        ("ep", expert, "ep"), ("expert-dp", expert, "dp"),
    ):
        groups = rank_groups(order, sizes, token)
        node_span = [len({node_of(r) for r in g}) for g in groups]
        switch_span = [len({switches[node_of(r)] for r in g}) for g in groups]
        report[label] = {
            "size": sizes[token],
            "groups": len(groups),
            "max_nodes": max(node_span),
            "max_switches": max(switch_span),
            "groups_across_switches": sum(1 for s in switch_span if s > 1),
        }

    # pipeline p2p: for each pair of consecutive stages of every pipeline, is the hop inside a switch?
    pp_hops = [0, 0]  # [same switch, cross switch]
    for g in rank_groups(order, dense, "pp"):
        for a, b in zip(g, g[1:]):
            pp_hops[switches[node_of(a)] != switches[node_of(b)]] += 1
    report["pp-hops"] = {"same_switch": pp_hops[0], "cross_switch": pp_hops[1]}
    report["switches"] = len(set(switches.values()))
    return report


def format_report(report: dict) -> str:
    lines = [f"switches: {report['switches']}"]
    for label, r in report.items():
        if label in ("switches", "pp-hops"):
            continue
        lines.append(
            f"{label:>10}: size {r['size']:>3}  groups {r['groups']:>4}  max span {r['max_nodes']} nodes / "
            f"{r['max_switches']} switches  ({r['groups_across_switches']} groups cross switches)"
        )
    hops = report["pp-hops"]
    lines.append(f"   pp hops: {hops['same_switch']} inside a switch, {hops['cross_switch']} across switches")
    return "\n".join(lines)


def coords(order: str, sizes: dict[str, int], rank: int) -> dict[str, int]:
    """Index of ``rank`` along every parallel dimension (the inverse of ``rank_groups``)."""
    out, stride = {}, 1
    for d in order.split("-"):
        out[d] = (rank // stride) % sizes[d]
        stride *= sizes[d]
    return out


def format_layout(nodes: list[dict], placement: dict, tp: int, pp: int, cp: int, ep: int, etp: int | None = None) -> str:
    """One line per node, one cell per GPU: global rank and its pipeline stage / data / expert (/ tp / cp) index.

    Example (4 GPUs per node, PP2 x EP4 x DP4)::

        node        switch  | gpu0                | gpu1                | ...
        rc10101     leaf0   | r0   pp0 dp0 ep0    | r1   pp0 dp1 ep1    | ...
    """
    gpus = int(placement.get("gpus_per_node", 4))
    order = placement.get("order", ORDERS[0])
    dense, expert = layout_sizes(len(nodes) * gpus, tp, pp, cp, ep, etp)
    switches = switch_map(nodes, placement)

    def cell(rank):
        c = coords(order, dense, rank)
        e = coords(order, expert, rank)
        parts = [f"pp{c['pp']}", f"dp{c['dp']}", f"ep{e['ep']}"]
        if tp > 1:
            parts.append(f"tp{c['tp']}")
        if cp > 1:
            parts.append(f"cp{c['cp']}")
        return f"r{rank:<3} " + " ".join(parts)

    rows = [[n["name"], switches[n["name"]], *(cell(i * gpus + g) for g in range(gpus))] for i, n in enumerate(nodes)]
    header = ["node", "switch", *(f"gpu{g}" for g in range(gpus))]
    widths = [max(len(r[i]) for r in [header, *rows]) for i in range(len(header))]
    lines = []
    for r in [header, *rows]:
        lines.append(f"{r[0]:<{widths[0]}}  {r[1]:<{widths[1]}} | " + " | ".join(f"{c:<{w}}" for c, w in zip(r[2:], widths[2:])))
    return "\n".join(lines)


def _cli(argv=None):
    import argparse

    p = argparse.ArgumentParser(description="Node lists (aligned / crossed / mixed / leaf) and their crossing counts")
    p.add_argument("--hosts", required=True, help="healthy full-GPU nodes, Slurm hostlist syntax; the first is the main node")
    p.add_argument("--topology-file", help="`scontrol show topology` output (default: run scontrol)")
    p.add_argument("--group-nodes", type=int, default=2, help="nodes per expert group (EP size / GPUs per node)")
    p.add_argument("--mixed-swaps", type=int, default=3)
    args = p.parse_args(argv)

    hosts = expand_hostlist(args.hosts)
    nodes = [{"name": h, "main": i == 0} for i, h in enumerate(hosts)]
    if args.topology_file:
        text = open(args.topology_file).read()
        topo = parse_scontrol_topology(text)
    else:
        topo = slurm_topology()
    placement = {"topology": {h: topo.get(h, "unknown") for h in hosts}, "group_nodes": args.group_nodes,
                 "mixed_swaps": args.mixed_swaps}
    result = plan(nodes, placement)
    for name, r in result.items():
        if "error" in r:
            print(f"{name:>12}: {r['error']}")
        else:
            print(f"{name:>12}: {len(r['nodes'])} nodes, {r['crossing']} / {r['groups']} groups cross a switch")
            print("              " + ",".join(r["nodes"]))


if __name__ == "__main__":
    _cli()
