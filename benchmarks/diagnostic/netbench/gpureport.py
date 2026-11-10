"""GPU matrices and slow-GPU report from netbench rows.

Shared by ``main.py`` (end of a run) and ``report.py`` (a whole milabench run
folder, one netbench entry per test). Rows are the dicts netbench logs:
``test``, ``src``, ``dst``, ``rate`` (GB/s), devices named ``n<node>:gpu<i>``
or ``n<node>:<nic>``.
"""

from __future__ import annotations

import re
from collections import defaultdict
from statistics import median

GPU = re.compile(r"^n(\d+):gpu(\d+)$")
NIC = re.compile(r"^n(\d+):\D*?(\d+)$")  # n0:ionic_3 -> (0, 3): NIC i sits next to GPU i

# column -> (test, side): side is which end of the row the value is charged to
COLUMNS = {
    "h2d": ("h2d_pinned_all", "dst"),
    "d2h": ("d2h_pinned_all", "src"),
    "hbm": ("hbm_triad", "src"),
    "p2p tx": ("p2p", "src"),
    "p2p rx": ("p2p", "dst"),
    "inter tx": ("inter", "src"),
    "inter rx": ("inter", "dst"),
    "xnode tx": ("inter_cross", "src"),
    "xnode rx": ("inter_cross", "dst"),
    "rdma": ("rdma", "both"),
}

# Pair tests: which matrix a column reads from, and which end of the pair it is
PAIR_FAMILIES = {
    "same node": ("p2p",),
    "across nodes": ("inter", "inter_cross"),
}
PAIR_COLUMNS = {
    "p2p tx": ("same node", "src"),
    "p2p rx": ("same node", "dst"),
    "inter tx": ("across nodes", "src"),
    "inter rx": ("across nodes", "dst"),
    "xnode tx": ("across nodes", "src"),
    "xnode rx": ("across nodes", "dst"),
}

# A GPU is blamed for a slow link only when most of its pairs are slow (a slow
# row or column); slow pairs scattered over many GPUs point at the path between
# them (switch, ECMP hashing onto a degraded link), not at a GPU.
GPU_FAULT_FRACTION = 0.75
MIN_PAIRS = 3


def _device(name, test):
    m = GPU.match(name or "")
    if m:
        return int(m.group(1)), int(m.group(2))
    if test == "rdma":
        m = NIC.match(name or "")
        if m:
            return int(m.group(1)), int(m.group(2))
    return None


def _dims(rows):
    nodes, gpus = set(), set()
    for r in rows:
        for key in ("src", "dst"):
            d = _device(r.get(key), r.get("test"))
            if d:
                nodes.add(d[0])
                gpus.add(d[1])
    return sorted(nodes), sorted(gpus)


def print_matrices(rows, line):
    """GB/s matrices: same node (p2p) and node a -> node b (inter + inter_cross)."""
    nodes, gpus = _dims(rows)
    pairs = defaultdict(list)
    for r in rows:
        if r["test"] in ("p2p", "inter", "inter_cross"):
            pairs[(r["src"], r["dst"])].append(r["rate"])
    if not pairs:
        return

    def matrix(title, a, b):
        cells = {
            (i, j): pairs.get((f"n{a}:gpu{i}", f"n{b}:gpu{j}"))
            for i in gpus for j in gpus
        }
        if not any(cells.values()):
            return
        line("")
        line(f"--- {title} (send/recv GB/s, row = src, col = dst) " + "-" * max(0, 44 - len(title)))
        line("          " + "".join(f"{f'n{b}:gpu{j}':>9}" for j in gpus))
        for i in gpus:
            vals = []
            for j in gpus:
                v = cells[(i, j)]
                vals.append(f"{'-' if not v else f'{sum(v) / len(v):.1f}':>9}")
            line(f"  n{a}:gpu{i} " + "".join(vals))

    for a in nodes:
        matrix(f"node {a}", a, a)
    for a in nodes:
        for b in nodes:
            if a != b:
                matrix(f"node {a} -> node {b}", a, b)


def per_gpu(rows):
    """{column: {(node, gpu): mean GB/s}}"""
    acc = defaultdict(lambda: defaultdict(list))
    by_test = defaultdict(list)
    for r in rows:
        by_test[r["test"]].append(r)
    for column, (test, side) in COLUMNS.items():
        for r in by_test.get(test, []):
            keys = ("src", "dst") if side == "both" else (side,)
            for key in keys:
                d = _device(r.get(key), test)
                if d:
                    acc[column][d].append(r["rate"])
    return {
        column: {d: sum(v) / len(v) for d, v in cells.items()}
        for column, cells in acc.items()
    }


def diagnose_pairs(rows, line, threshold):
    """Classify slow GPU pairs as GPU faults or path faults.

    Returns ``{family: {(device, end) or None}}``: the (device, "src"|"dst") ends
    to blame, or None when there are too few pairs per GPU to tell.
    """
    blamed = {}
    for family, tests in PAIR_FAMILIES.items():
        rates = defaultdict(list)
        for r in rows:
            if r["test"] in tests:
                rates[(r["src"], r["dst"])].append(r["rate"])
        if not rates:
            continue
        cells = {pair: sum(v) / len(v) for pair, v in rates.items()}
        med = median(cells.values())
        slow = {pair: v for pair, v in cells.items() if v < med * (1 - threshold)}

        pairs_of = defaultdict(list)
        for pair in cells:
            pairs_of[(pair[0], "src")].append(pair)
            pairs_of[(pair[1], "dst")].append(pair)
        if max(len(v) for v in pairs_of.values()) < MIN_PAIRS:
            blamed[family] = None
        else:
            blamed[family] = {
                end for end, pairs in pairs_of.items()
                if len(pairs) >= MIN_PAIRS
                and sum(p in slow for p in pairs) / len(pairs) >= GPU_FAULT_FRACTION
            }

        line("")
        if not slow:
            line(f"{family}: all {len(cells)} GPU pairs within {100 * threshold:.0f}% of the median ({med:.1f} GB/s)")
            continue
        slow_mean = sum(slow.values()) / len(slow)
        line(f"{family}: {len(slow)}/{len(cells)} GPU pairs slow, {slow_mean:.1f} GB/s vs median {med:.1f}")
        if blamed[family] is None:
            line("  -> too few pairs per GPU to tell a GPU from the path between them"
                 " (run net-inter with --inter-pairs all for the full matrix)")
        else:
            for dev, end in sorted(blamed[family]):
                line(f"  -> GPU fault: {dev} is slow on most pairs it {'sends' if end == 'src' else 'receives'}")
            explained = {
                p for p in slow
                if (p[0], "src") in blamed[family] or (p[1], "dst") in blamed[family]
            }
            path = sorted(set(slow) - explained)
            if path:
                line(f"  -> path fault: {len(path)} slow pair(s) not explained by a slow GPU (each GPU is fast with"
                     " other peers): suspect the network path (switch port, ECMP hashing onto a degraded link)")
                for i in range(0, len(path), 4):
                    line("     " + "   ".join(f"{a}->{b} {cells[(a, b)]:.1f}" for a, b in path[i:i + 4]))
    return blamed


def print_slow_gpus(rows, line, threshold=0.10):
    """Per node GPU x test table; marks cells below (1 - threshold) x column median.

    For pair tests a GPU is only marked '*' (and listed) when the pair diagnosis
    blames it; '~' marks a GPU whose slow pairs are blamed on the path or on its peer.
    """
    table = per_gpu(rows)
    if not table:
        return []

    line("")
    line("--- Slow link diagnosis " + "-" * 76)
    blamed = diagnose_pairs(rows, line, threshold)
    columns = [c for c in COLUMNS if c in table]
    medians = {c: median(table[c].values()) for c in columns}
    nodes, gpus = _dims(rows)

    flagged = []
    line("")
    line(f"--- Slow GPUs: GB/s per GPU, '*' = below {100 * (1 - threshold):.0f}% of the column median, "
         "'~' = blamed on the path or the peer GPU")
    line("          " + "".join(f"{c:>10}" for c in columns))
    for node in nodes:
        for g in gpus:
            cells = []
            for c in columns:
                v = table[c].get((node, g))
                if v is None:
                    cells.append(f"{'-':>10}")
                    continue
                mark = " "
                if v < medians[c] * (1 - threshold):
                    mark = "*"
                    if c in PAIR_COLUMNS:
                        family, end = PAIR_COLUMNS[c]
                        faults = blamed.get(family)
                        # None: cannot tell, keep the GPU flagged
                        if faults is not None and (f"n{node}:gpu{g}", end) not in faults:
                            mark = "~"
                    if mark == "*":
                        flagged.append((node, g, c, v, medians[c]))
                cells.append(f"{v:>9.1f}{mark}")
            line(f"  n{node}:gpu{g} " + "".join(cells))
    line("  median   " + "".join(f"{medians[c]:>9.1f} " for c in columns))

    line("")
    if not flagged:
        line(f"No slow GPU (below {100 * (1 - threshold):.0f}% of the median and not explained by the path)")
    for node, g, c, v, med in sorted(flagged):
        what = f"n{node}:nic{g}" if c == "rdma" else f"n{node}:gpu{g}"
        line(f"  SLOW {what:<10} {c:<9} {v:8.1f} GB/s   {100 * (v / med - 1):+.0f}% vs median {med:.1f}")
    return flagged
