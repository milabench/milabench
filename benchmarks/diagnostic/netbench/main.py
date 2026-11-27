#!/usr/bin/env python
"""Interconnect and GPU memory diagnostic (rccl-tests style, through torch.distributed).

Launched with one process per GPU on every node (torchrun). Measures:

* ``host``  CPU <-> GPU     H2D / D2H copies, pinned and pageable, each GPU alone
                            then all GPUs of a node at once.
* ``p2p``   GPU <-> GPU     CCL send/recv between every ordered GPU pair of a node.
* ``inter`` GPU <-> GPU     CCL send/recv between every GPU of node 0 and every
            across nodes    GPU of node k, both directions, one pair at a time
                            (GPU i <-> GPU i is ``inter``, i != j ``inter_cross``),
                            then GPU i <-> GPU i all at once (one way, both ways).
* ``coll``  collectives     all_reduce / all_gather / reduce_scatter / all_to_all
                            over the node-local group and over the whole world.
* ``net``   node <-> node   ``ib_write_bw`` per RDMA NIC (host memory) and a gloo
                            TCP send/recv between the nodes.
* ``latency`` small messages  all_reduce / all_gather (8 B .. 1 MiB) and a send/recv
                            ping-pong between two GPUs; ``rate`` is the time in us.
* ``hbm``   GPU memory      copy / read / write / triad on every GPU at once.

Every rank records its own timings; rank 0 gathers them, prints a report and
logs one metric line per measurement (``rate`` in GB/s, 1 GB = 1e9 bytes).
Only the headline rows of each test (see ``HEADLINE``) are logged as ``train``
rates, so a milabench entry running one test scores that test alone; the rest
is logged as ``detail``.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import socket
import subprocess
import time
from collections import defaultdict

import torch
import torch.distributed as dist
import torchcompat.core as accelerator
from benchmate.monitor import setupvoir

from gpureport import print_matrices, print_slow_gpus


GB = 1e9
MB = 1024**2

TESTS = ("host", "p2p", "inter", "coll", "net", "latency", "hbm")

# rccl-tests bus bandwidth factors (busbw = algbw * factor(n))
COLLECTIVES = {
    "all_reduce": lambda n: 2 * (n - 1) / n,
    "all_gather": lambda n: (n - 1) / n,
    "reduce_scatter": lambda n: (n - 1) / n,
    "all_to_all": lambda n: (n - 1) / n,
}

# Rows that make up a test's score (milabench averages the ``train`` rates)
HEADLINE = {
    "h2d_pinned_all",   # host:  per GPU, all GPUs of the node copying at once
    "d2h_pinned_all",
    "p2p",              # p2p:   every ordered GPU pair, one at a time
    "inter",            # inter: GPU i <-> GPU i across nodes, one rail at a time
    "rdma",             # net:   ib_write_bw, one NIC at a time
    "hbm_triad",        # hbm:   a + 2 * c -> b (STREAM triad), per GPU
}                       # coll:  busbw of every collective at the largest size
                        # latency: time of every op at the smallest size (us)


class Ctx:
    def __init__(self, args):
        self.args = args
        self.rank = dist.get_rank()
        self.world = dist.get_world_size()
        self.local_rank = int(os.getenv("LOCAL_RANK", 0))
        self.local_world = int(os.getenv("LOCAL_WORLD_SIZE", torch.cuda.device_count()))
        self.node = self.rank // self.local_world
        self.nnodes = self.world // self.local_world
        self.device = accelerator.fetch_device(self.local_rank)
        self.records = []

    def global_rank(self, node, local_rank):
        return node * self.local_world + local_rank

    def record(self, test, src, dst, nbytes, seconds, **extra):
        self.records.append(
            dict(test=test, src=src, dst=dst, bytes=nbytes, seconds=seconds, rank=self.rank, **extra)
        )

    def gpu(self, node=None, local_rank=None):
        node = self.node if node is None else node
        local_rank = self.local_rank if local_rank is None else local_rank
        return f"n{node}:gpu{local_rank}"


def sync():
    accelerator.synchronize()


def timed(fn, repeats, warmup):
    """Average seconds per call of ``fn`` (amortized, like rccl-tests)."""
    for _ in range(warmup):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    sync()
    return (time.perf_counter() - t0) / repeats


def bind_to_gpu_numa(device):
    """Pin this process to the CPUs local to its GPU so pinned buffers land on the right socket."""
    try:
        props = accelerator.get_device_properties(device)
        bdf = f"{props.pci_domain_id:04x}:{props.pci_bus_id:02x}:{props.pci_device_id:02x}.0"
        with open(f"/sys/bus/pci/devices/{bdf}/local_cpulist") as f:
            cpus = set()
            for part in f.read().strip().split(","):
                lo, _, hi = part.partition("-")
                cpus.update(range(int(lo), int(hi or lo) + 1))
        os.sched_setaffinity(0, cpus)
        return bdf
    except Exception as err:
        print(f"[netbench] numa bind failed: {err}", flush=True)
        return None


# --------------------------------------------------------------------------- host
def test_host(ctx):
    args = ctx.args
    nbytes = int(args.size_mb * MB)
    gpu = torch.empty(nbytes, dtype=torch.uint8, device=ctx.device)
    pinned = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
    pageable = torch.empty(nbytes, dtype=torch.uint8)
    pageable.fill_(1)

    def run(suffix):
        for kind, host in (("pinned", pinned), ("pageable", pageable)):
            t = timed(lambda: gpu.copy_(host, non_blocking=True), args.repeats, args.warmup)
            ctx.record(f"h2d_{kind}{suffix}", "cpu", ctx.gpu(), nbytes, t)
            t = timed(lambda: host.copy_(gpu, non_blocking=True), args.repeats, args.warmup)
            ctx.record(f"d2h_{kind}{suffix}", ctx.gpu(), "cpu", nbytes, t)

    # Each GPU alone (nodes run in parallel, they do not share host links)
    for lr in range(ctx.local_world):
        if ctx.local_rank == lr:
            run("")
        dist.barrier()

    # All GPUs of a node at once: host memory / PCIe root complex contention
    run("_all")
    dist.barrier()


# --------------------------------------------------------------------------- p2p
def sendrecv(ctx, test, src, dst, buf, src_name, dst_name):
    """One-way send/recv between two global ranks, timed on the receiver."""
    args = ctx.args
    if ctx.rank == src:
        timed(lambda: dist.send(buf, dst), args.repeats, args.warmup)
    elif ctx.rank == dst:
        t = timed(lambda: dist.recv(buf, src), args.repeats, args.warmup)
        ctx.record(test, src_name, dst_name, buf.numel() * buf.element_size(), t)


def exchange(ctx, test, peer, buf_send, buf_recv, my_name, peer_name):
    """Both directions at once with ``peer``; records what this rank received."""
    args = ctx.args

    def step():
        ops = [
            dist.P2POp(dist.isend, buf_send, peer),
            dist.P2POp(dist.irecv, buf_recv, peer),
        ]
        for req in dist.batch_isend_irecv(ops):
            req.wait()

    t = timed(step, args.repeats, args.warmup)
    ctx.record(test, peer_name, my_name, buf_recv.numel() * buf_recv.element_size(), t)


def test_p2p(ctx):
    nbytes = int(ctx.args.size_mb * MB)
    buf = torch.empty(nbytes, dtype=torch.uint8, device=ctx.device)
    base = ctx.global_rank(ctx.node, 0)

    # Every node walks the same pair list at the same time
    for i in range(ctx.local_world):
        for j in range(ctx.local_world):
            if i == j:
                continue
            sendrecv(ctx, "p2p", base + i, base + j, buf, ctx.gpu(local_rank=i), ctx.gpu(local_rank=j))
            dist.barrier()


# --------------------------------------------------------------------------- inter
def test_inter(ctx):
    if ctx.nnodes < 2:
        return

    nbytes = int(ctx.args.size_mb * MB)
    buf = torch.empty(nbytes, dtype=torch.uint8, device=ctx.device)
    buf2 = torch.empty(nbytes, dtype=torch.uint8, device=ctx.device)
    lr = ctx.local_rank

    for k in range(1, ctx.nnodes):
        # One GPU pair at a time, both directions: i == j isolates each NIC / rail,
        # i != j is the cross-rail traffic all_to_all relies on
        for a, b in ((0, k), (k, 0)):
            for i in range(ctx.local_world):
                for j in range(ctx.local_world):
                    if i != j and ctx.args.inter_pairs == "diag":
                        continue
                    test = "inter" if i == j else "inter_cross"
                    src, dst = ctx.global_rank(a, i), ctx.global_rank(b, j)
                    sendrecv(ctx, test, src, dst, buf, ctx.gpu(a, i), ctx.gpu(b, j))
                    dist.barrier()

        # All GPU pairs at once, node 0 -> node k
        src, dst = ctx.global_rank(0, lr), ctx.global_rank(k, lr)
        sendrecv(ctx, "inter_all", src, dst, buf, ctx.gpu(0, lr), ctx.gpu(k, lr))
        dist.barrier()

        # All GPU pairs at once, both directions
        if ctx.node in (0, k):
            other = k if ctx.node == 0 else 0
            exchange(ctx, "inter_all_bidir", ctx.global_rank(other, lr), buf, buf2, ctx.gpu(), ctx.gpu(other, lr))
        dist.barrier()


# --------------------------------------------------------------------------- coll
def run_collective(ctx, op, group, nranks, nbytes, label, test=None, repeats=None, warmup=None):
    args = ctx.args
    n = max(nbytes // 4 // nranks * nranks, nranks)
    nbytes = n * 4
    x = torch.ones(n, dtype=torch.float32, device=ctx.device)

    if op == "all_reduce":
        fn = lambda: dist.all_reduce(x, group=group)
    elif op == "all_gather":
        # size = gathered output, like rccl-tests
        part = torch.ones(n // nranks, dtype=torch.float32, device=ctx.device)
        fn = lambda: dist.all_gather_into_tensor(x, part, group=group)
    elif op == "reduce_scatter":
        out = torch.empty(n // nranks, dtype=torch.float32, device=ctx.device)
        fn = lambda: dist.reduce_scatter_tensor(out, x, group=group)
    elif op == "all_to_all":
        out = torch.empty_like(x)
        fn = lambda: dist.all_to_all_single(out, x, group=group)

    t = timed(fn, repeats or args.repeats, warmup or args.warmup)
    ctx.record(test or op, label, label, nbytes, t, nranks=nranks)


def coll_groups(ctx):
    wanted = ctx.args.coll_groups.split(",")

    # new_subgroups is collective: every rank creates it even if unused
    intra, _ = dist.new_subgroups(ctx.local_world)
    groups = []
    if "intra" in wanted:
        groups.append((intra, ctx.local_world, "intra"))
    if "world" in wanted and ctx.nnodes > 1:
        groups.append((None, ctx.world, "world"))
    return groups


def test_coll(ctx):
    sizes = [int(float(s) * MB) for s in ctx.args.coll_sizes_mb.split(",")]

    for group, nranks, label in coll_groups(ctx):
        for op in COLLECTIVES:
            for nbytes in sizes:
                run_collective(ctx, op, group, nranks, nbytes, label)
                dist.barrier()


# --------------------------------------------------------------------------- latency
def pingpong(ctx, a, b, nbytes, label):
    """Round trip a <-> b; records the one-way time on ``a``."""
    args = ctx.args
    buf = torch.empty(max(nbytes, 1), dtype=torch.uint8, device=ctx.device)

    def ping():
        dist.send(buf, b)
        dist.recv(buf, b)

    def pong():
        dist.recv(buf, a)
        dist.send(buf, a)

    if ctx.rank == a:
        t = timed(ping, args.lat_repeats, args.lat_warmup)
        ctx.record("lat_sendrecv", label, label, nbytes, t / 2, nranks=2)
    elif ctx.rank == b:
        timed(pong, args.lat_repeats, args.lat_warmup)


def test_latency(ctx):
    args = ctx.args
    sizes = [int(s) for s in args.lat_sizes.split(",")]
    groups = coll_groups(ctx)

    for group, nranks, label in groups:
        for op in ("all_reduce", "all_gather"):
            for nbytes in sizes:
                run_collective(
                    ctx, op, group, nranks, nbytes, label,
                    test=f"lat_{op}", repeats=args.lat_repeats, warmup=args.lat_warmup,
                )
                dist.barrier()

    labels = {label for _, _, label in groups}
    pairs = []
    if "intra" in labels and ctx.local_world > 1:
        # GPU 0 <-> GPU 1 of every node, nodes in parallel
        pairs.append((ctx.global_rank(ctx.node, 0), ctx.global_rank(ctx.node, 1), "intra"))
    if "world" in labels:
        pairs.append((ctx.global_rank(0, 0), ctx.global_rank(1, 0), "inter"))

    for a, b, label in pairs:
        for nbytes in sizes:
            pingpong(ctx, a, b, nbytes, label)
            dist.barrier()


# --------------------------------------------------------------------------- hbm
def test_hbm(ctx):
    """STREAM-like kernels on every GPU at once (GPUs do not share memory)."""
    args = ctx.args
    n = int(args.hbm_size_mb * MB) // 4
    nbytes = n * 4
    a = torch.ones(n, dtype=torch.float32, device=ctx.device)
    b = torch.empty_like(a)
    c = torch.ones_like(a)

    kernels = {
        "hbm_copy": (lambda: b.copy_(a), 2),                           # read a, write b
        "hbm_read": (lambda: a.sum(), 1),
        "hbm_write": (lambda: b.fill_(0.0), 1),
        "hbm_triad": (lambda: torch.add(a, c, alpha=2.0, out=b), 3),   # read a, c, write b
    }
    for name, (fn, factor) in kernels.items():
        t = timed(fn, args.repeats, args.warmup)
        ctx.record(name, ctx.gpu(), ctx.gpu(), nbytes * factor, t)
        dist.barrier()


# --------------------------------------------------------------------------- net
def iface_ip(name):
    import fcntl
    import struct

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        packed = fcntl.ioctl(s.fileno(), 0x8915, struct.pack("256s", name[:15].encode()))  # SIOCGIFADDR
    return socket.inet_ntoa(packed[20:24])


def local_ip():
    ifname = os.getenv("NCCL_SOCKET_IFNAME", "").split(",")[0].lstrip("=^")
    if ifname:
        try:
            return iface_ip(ifname)
        except OSError:
            pass
    return socket.gethostbyname(socket.gethostname())


def rdma_devices():
    hcas = os.getenv("NCCL_IB_HCA", "").lstrip("=^")
    if hcas:
        return [h.split(":")[0] for h in hcas.split(",")]
    if os.path.isdir("/sys/class/infiniband"):
        return sorted(os.listdir("/sys/class/infiniband"))
    return []


IB_BW_LINE = re.compile(r"^\s*(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$")


def ib_write_bw(ctx, server, client, server_ip, hca, port):
    """Runs ``ib_write_bw`` with ``server`` / ``client`` global ranks. Returns GB/s on the client."""
    args = ctx.args
    cmd = [
        "ib_write_bw", "-d", hca, "-p", str(port), "-F", "--report_gbits",
        "-s", str(args.rdma_msg_bytes), "-D", str(args.rdma_seconds), "-q", str(args.rdma_qps),
    ]
    gid = os.getenv("NCCL_IB_GID_INDEX")
    if gid:
        cmd += ["-x", gid]

    proc = None
    if ctx.rank == server:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    dist.barrier()

    if ctx.rank == client:
        time.sleep(1)  # let the server start listening
        try:
            out = subprocess.run(
                cmd + [server_ip], capture_output=True, text=True, timeout=args.rdma_seconds + 30
            ).stdout
        except subprocess.TimeoutExpired as err:
            out = err.stdout or ""
        for line in out.splitlines():
            m = IB_BW_LINE.match(line)
            if m:
                return float(m.group(4)) / 8  # BW average Gb/s -> GB/s
        # Only local rank 0 output is shown; report failures as 0 GB/s so rank 0 sees them
        print(f"[netbench] ib_write_bw {hca} failed:\n{out}", flush=True)
        return 0.0

    if proc is not None:
        try:
            proc.wait(timeout=args.rdma_seconds + 30)
        except subprocess.TimeoutExpired:
            proc.kill()
    return None


def test_net(ctx):
    if ctx.nnodes < 2:
        return

    ips = [None] * ctx.world
    dist.all_gather_object(ips, local_ip())

    # Raw RDMA, host memory, one NIC per local rank (GPU i <-> NIC i on the same switch)
    hcas = rdma_devices()
    if shutil.which("ib_write_bw") and hcas:
        for k in range(1, ctx.nnodes):
            for i in range(min(len(hcas), ctx.local_world)):
                server, client = ctx.global_rank(k, i), ctx.global_rank(0, i)
                bw = ib_write_bw(ctx, server, client, ips[server], hcas[i], 18515 + i)
                if bw is not None:
                    ctx.record("rdma", f"n0:{hcas[i]}", f"n{k}:{hcas[i]}", None, None, gbps=bw)
                dist.barrier()

            lr = ctx.local_rank
            if lr < len(hcas):
                server, client = ctx.global_rank(k, lr), ctx.global_rank(0, lr)
            else:
                server = client = -1
            bw = ib_write_bw(ctx, server, client, ips[server] if server >= 0 else None, hcas[lr % len(hcas)], 18515 + lr)
            if bw is not None:
                ctx.record("rdma_all", f"n0:{hcas[lr]}", f"n{k}:{hcas[lr]}", None, None, gbps=bw)
            dist.barrier()
    elif ctx.rank == 0:
        print("[netbench] ib_write_bw or RDMA devices not found, skipping rdma", flush=True)

    # TCP through gloo between the first rank of each node (default route / GLOO_SOCKET_IFNAME)
    leaders = [ctx.global_rank(k, 0) for k in range(ctx.nnodes)]
    gloo = dist.new_group(leaders, backend="gloo")
    nbytes = int(ctx.args.size_mb * MB)
    buf = torch.empty(nbytes, dtype=torch.uint8)
    for k in range(1, ctx.nnodes):
        src, dst = leaders[0], leaders[k]
        if ctx.rank == src:
            timed(lambda: dist.send(buf, dst, group=gloo), ctx.args.repeats, ctx.args.warmup)
        elif ctx.rank == dst:
            t = timed(lambda: dist.recv(buf, src, group=gloo), ctx.args.repeats, ctx.args.warmup)
            ctx.record("tcp_gloo", f"n0:{ips[src]}", f"n{k}:{ips[dst]}", nbytes, t)
        dist.barrier()


# --------------------------------------------------------------------------- report
def bandwidth(r):
    if r.get("gbps") is not None:
        return r["gbps"]
    return r["bytes"] / r["seconds"] / GB


def summarize(records, ctx):
    """Turn raw per-rank records into report rows: one row per measurement."""
    rows = []
    by_test = defaultdict(list)
    for r in records:
        by_test[r["test"]].append(r)

    for test, rs in by_test.items():
        latency = test.startswith("lat_")
        if test in COLLECTIVES or latency:
            # slowest rank defines the collective time
            agg = defaultdict(list)
            for r in rs:
                agg[(r["src"], r["bytes"], r["nranks"])].append(r["seconds"])
            # headline: largest size for bandwidth, smallest for latency
            pick = min if latency else max
            sizes = defaultdict(list)
            for label, nbytes, _ in agg:
                sizes[label].append(nbytes)
            for (label, nbytes, nranks), times in sorted(agg.items()):
                t = max(times)
                row = dict(
                    test=test, src=label, dst=label, size_bytes=nbytes, nranks=nranks,
                    latency_us=t * 1e6, headline=nbytes == pick(sizes[label]),
                )
                if latency:
                    row.update(rate=t * 1e6, units="us")
                else:
                    algbw = nbytes / t / GB
                    row.update(rate=algbw * COLLECTIVES[test](nranks), algbw=algbw, units="GB/s")
                rows.append(row)
        else:
            for r in rs:
                row = dict(
                    test=test, src=r["src"], dst=r["dst"], size_bytes=r["bytes"], rate=bandwidth(r),
                    units="GB/s", headline=test in HEADLINE,
                )
                if r["seconds"] is not None:
                    row["latency_us"] = r["seconds"] * 1e6
                rows.append(row)
    return rows


def print_report(rows, ctx):
    by_test = defaultdict(list)
    for r in rows:
        by_test[r["test"]].append(r)

    def line(s=""):
        print(f"[netbench] {s}", flush=True)

    def stats(test, title, aggregate=False):
        rs = by_test.get(test)
        if not rs:
            return
        bws = [r["rate"] for r in rs]
        worst = min(rs, key=lambda r: r["rate"])
        msg = (
            f"{title:<34} min {min(bws):8.2f}  mean {sum(bws) / len(bws):8.2f}  max {max(bws):8.2f} GB/s"
            f"   (n={len(bws)}, slowest {worst['src']}->{worst['dst']})"
        )
        if aggregate:
            per_node = defaultdict(float)
            for r in rs:
                per_node[r["dst"].split(":")[0] if r["src"] == "cpu" else r["src"].split(":")[0]] += r["rate"]
            msg += "   sum/node " + " ".join(f"{k}={v:.1f}" for k, v in sorted(per_node.items()))
        line(msg)

    line("=" * 100)
    line(f"world={ctx.world} nodes={ctx.nnodes} gpus/node={ctx.local_world} size={ctx.args.size_mb} MiB  (GB/s = 1e9 B/s)")

    if any(t.startswith(("h2d_", "d2h_")) for t in by_test):
        line("")
        line("--- CPU <-> GPU " + "-" * 60)
        for kind in ("pinned", "pageable"):
            for d in ("h2d", "d2h"):
                stats(f"{d}_{kind}", f"{d} {kind} (one GPU at a time)")
                stats(f"{d}_{kind}_all", f"{d} {kind} (all GPUs at once)", aggregate=True)

    print_matrices(rows, line)
    if "p2p" in by_test:
        line("")
        stats("p2p", "p2p all pairs, same node")
    if "inter" in by_test:
        line("")
        stats("inter", "across nodes, GPU i <-> GPU i")
        stats("inter_cross", "across nodes, GPU i <-> GPU j != i")
        stats("inter_all", "GPU i <-> GPU i all at once (one way)", aggregate=True)
        stats("inter_all_bidir", "GPU i <-> GPU i all at once (both ways)", aggregate=True)

    colls = [r for r in rows if r["test"] in COLLECTIVES]
    if colls:
        line("")
        line("--- Collectives (busbw like rccl-tests) " + "-" * 36)
        line(f"  {'op':<16}{'group':<8}{'ranks':>6}{'size':>12}{'time us':>12}{'algbw':>10}{'busbw':>10}")
        for r in colls:
            line(
                f"  {r['test']:<16}{r['src']:<8}{r['nranks']:>6}{r['size_bytes'] / MB:>10.1f}Mi"
                f"{r['latency_us']:>12.1f}{r['algbw']:>10.2f}{r['rate']:>10.2f}"
            )

    lats = [r for r in rows if r["test"].startswith("lat_")]
    if lats:
        line("")
        line("--- Small-message latency (one way for sendrecv) " + "-" * 26)
        line(f"  {'op':<16}{'group':<8}{'ranks':>6}{'bytes':>10}{'time us':>12}")
        for r in lats:
            line(f"  {r['test'][4:]:<16}{r['src']:<8}{r['nranks']:>6}{r['size_bytes']:>10}{r['latency_us']:>12.1f}")

    if any(t.startswith("hbm_") for t in by_test):
        line("")
        line(f"--- GPU memory ({ctx.args.hbm_size_mb:.0f} MiB buffers, all GPUs at once) " + "-" * 24)
        for name in ("hbm_copy", "hbm_read", "hbm_write", "hbm_triad"):
            stats(name, name.replace("_", " "))

    if "rdma" in by_test or "tcp_gloo" in by_test:
        line("")
        line("--- Node <-> Node " + "-" * 58)
        for r in by_test.get("rdma", []):
            line(f"  ib_write_bw {r['src']} -> {r['dst']}   {r['rate']:8.2f} GB/s   ({r['rate'] * 8:.0f} Gb/s)")
        stats("rdma", "rdma one NIC at a time")
        stats("rdma_all", "rdma all NICs at once", aggregate=True)
        stats("tcp_gloo", "tcp (gloo, single stream)")

    print_slow_gpus(rows, line, ctx.args.slow_threshold)
    line("=" * 100)


def main():
    parser = argparse.ArgumentParser(description="Interconnect bandwidth diagnostic")
    parser.add_argument("--tests", type=str, default=",".join(TESTS), help=f"Comma separated subset of {TESTS}")
    parser.add_argument("--size-mb", type=float, default=256, help="Message size (MiB) for copies and send/recv")
    parser.add_argument("--coll-sizes-mb", type=str, default="1,16,256,1024", help="Collective sizes (MiB)")
    parser.add_argument("--coll-groups", type=str, default="intra,world", help="Collective groups: intra (node-local), world")
    parser.add_argument("--repeats", type=int, default=20, help="Timed iterations per measurement")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations per measurement")
    parser.add_argument("--lat-sizes", type=str, default="8,64,512,4096,32768,262144,1048576", help="Latency sizes (bytes)")
    parser.add_argument("--lat-repeats", type=int, default=200, help="Timed iterations per latency measurement")
    parser.add_argument("--lat-warmup", type=int, default=20, help="Warmup iterations per latency measurement")
    parser.add_argument("--hbm-size-mb", type=float, default=4096, help="Buffer size (MiB) for GPU memory kernels")
    parser.add_argument("--inter-pairs", choices=("all", "diag"), default="all", help="Across nodes: every GPU pair, or GPU i <-> GPU i only")
    parser.add_argument("--slow-threshold", type=float, default=0.10, help="Flag GPUs this fraction below the median")
    parser.add_argument("--rdma-seconds", type=int, default=3, help="ib_write_bw duration per run")
    parser.add_argument("--rdma-msg-bytes", type=int, default=1 << 20, help="ib_write_bw message size")
    parser.add_argument("--rdma-qps", type=int, default=4, help="ib_write_bw queue pairs")
    parser.add_argument(
        "--set-env", action="append", default=[], metavar="NAME=VALUE",
        help="Set before the process group starts, on every rank (workers only get system.env otherwise)",
    )
    parser.add_argument("--no-numa-bind", action="store_true", help="Do not pin ranks to their GPU's NUMA node")
    args = parser.parse_args()

    tests = [t.strip() for t in args.tests.split(",") if t.strip()]
    unknown = set(tests) - set(TESTS)
    if unknown:
        parser.error(f"unknown tests {sorted(unknown)}")

    for assignment in args.set_env:
        name, _, value = assignment.partition("=")
        os.environ[name] = value

    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = accelerator.fetch_device(local_rank)
    accelerator.set_device(device)  # object collectives allocate on the current device
    bdf = None if args.no_numa_bind else bind_to_gpu_numa(device)

    # gloo picks the interface from the hostname by default, which may not route between nodes
    if os.getenv("NCCL_SOCKET_IFNAME") and not os.getenv("GLOO_SOCKET_IFNAME"):
        os.environ["GLOO_SOCKET_IFNAME"] = os.environ["NCCL_SOCKET_IFNAME"].lstrip("=^")

    accelerator.init_process_group(backend=accelerator.ccl)
    ctx = Ctx(args)

    print(
        f"[netbench] host={socket.gethostname()} rank={ctx.rank}/{ctx.world} node={ctx.node} "
        f"local_rank={ctx.local_rank} device={device} pci={bdf}",
        flush=True,
    )

    for name in TESTS:
        if name not in tests:
            continue
        t0 = time.perf_counter()
        globals()[f"test_{name}"](ctx)
        dist.barrier()
        if ctx.rank == 0:
            print(f"[netbench] {name} done in {time.perf_counter() - t0:.1f}s", flush=True)

    gathered = [None] * ctx.world if ctx.rank == 0 else None
    dist.gather_object(ctx.records, gathered, dst=0)

    if ctx.rank == 0:
        rows = summarize([r for rs in gathered for r in rs], ctx)
        print_report(rows, ctx)

        log, monitor = setupvoir(monogpu=False, interval=0.5)
        for row in rows:
            task = "train" if row.pop("headline") else "detail"
            log({"task": task, "time": time.time(), **row})
        monitor.stop()

    dist.barrier()
    accelerator.destroy_process_group()


if __name__ == "__main__":
    main()
