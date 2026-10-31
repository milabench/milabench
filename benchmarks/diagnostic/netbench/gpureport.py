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


def print_slow_gpus(rows, line, threshold=0.10):
    """Per node GPU x test table; marks cells below (1 - threshold) x column median."""
    table = per_gpu(rows)
    if not table:
        return []
    columns = [c for c in COLUMNS if c in table]
    medians = {c: median(table[c].values()) for c in columns}
    nodes, gpus = _dims(rows)

    flagged = []
    line("")
    line(f"--- Slow GPUs: GB/s per GPU, '*' = below {100 * (1 - threshold):.0f}% of the column median " + "-" * 10)
    line("          " + "".join(f"{c:>10}" for c in columns))
    for node in nodes:
        for g in gpus:
            cells = []
            for c in columns:
                v = table[c].get((node, g))
                if v is None:
                    cells.append(f"{'-':>10}")
                    continue
                slow = v < medians[c] * (1 - threshold)
                if slow:
                    flagged.append((node, g, c, v, medians[c]))
                cells.append(f"{v:>9.1f}{'*' if slow else ' '}")
            line(f"  n{node}:gpu{g} " + "".join(cells))
    line("  median   " + "".join(f"{medians[c]:>9.1f} " for c in columns))

    line("")
    if not flagged:
        line(f"No GPU below {100 * (1 - threshold):.0f}% of the median")
    for node, g, c, v, med in sorted(flagged):
        what = f"n{node}:nic{g}" if c == "rdma" else f"n{node}:gpu{g}"
        line(f"  SLOW {what:<10} {c:<9} {v:8.1f} GB/s   {100 * (v / med - 1):+.0f}% vs median {med:.1f}")
    return flagged
