"""Shared pieces of the D2-D4 network sweeps: process group from Slurm, node-pair exchange, JSONL output.

One process per GPU, started by ``srun --ntasks-per-node=<gpus per node>`` (``net.sbatch``). Every
process sees all the GPUs of its node and drives GPU ``SLURM_LOCALID``, so NCCL sees the whole node
topology (NVLink, the HCA nearest to each GPU). ``MASTER_ADDR`` / ``MASTER_PORT`` come from the launcher.
"""

from __future__ import annotations

import json
import os
import socket
import statistics
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

GiB = 2**30


class Net:
    """The job's processes: rank, node list (in rank order), GPU index within the node."""

    def __init__(self):
        self.rank = int(os.environ["SLURM_PROCID"])
        self.world = int(os.environ["SLURM_NTASKS"])
        self.device = torch.device("cuda", int(os.environ["SLURM_LOCALID"]))
        torch.cuda.set_device(self.device)
        dist.init_process_group("nccl", rank=self.rank, world_size=self.world, device_id=self.device,
                                timeout=timedelta(minutes=10))
        self.host = socket.gethostname().split(".")[0]
        hosts = self.all_gather(self.host)
        self.nodes = list(dict.fromkeys(hosts))
        self.ranks_of = {h: [r for r, x in enumerate(hosts) if x == h] for h in self.nodes}
        self.gpus = len(self.ranks_of[self.host])
        assert all(len(r) == self.gpus for r in self.ranks_of.values()), "uneven GPUs per node"
        self.local = self.ranks_of[self.host].index(self.rank)
        self.job_id = os.environ.get("SLURM_JOB_ID", "local")
        self.nccl_version = ".".join(map(str, torch.cuda.nccl.version()))
        self._buffers = {}

    def all_gather(self, obj):
        out = [None] * self.world
        dist.all_gather_object(out, obj)
        return out

    def buffer(self, name, nbytes, dtype=torch.uint8):
        """One reusable buffer per name (re-allocated when a larger one is needed)."""
        numel = nbytes // torch.tensor([], dtype=dtype).element_size()
        buf = self._buffers.get(name)
        if buf is None or buf.dtype != dtype or buf.numel() < numel:
            self._buffers.pop(name, None)
            buf = self._buffers[name] = torch.ones(numel, dtype=dtype, device=self.device)
        return buf[:numel]

    def log(self, msg):
        if self.rank == 0:
            print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    def close(self):
        dist.destroy_process_group()


def timed(fn, iters, warmup, barrier=True):
    """Seconds per call of ``fn`` on this GPU, after ``warmup`` calls; a world barrier aligns the start."""
    for _ in range(warmup):
        fn()
    if barrier:
        dist.barrier()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / 1e3 / iters


def exchange(net: Net, pairs, size, iters=10, warmup=3, lat_iters=200, lat_size=8):
    """All ``pairs`` of nodes at once: GPU i of a <-> GPU i of b, both directions together.

    Bandwidth: every GPU sends and receives ``size`` bytes per iteration; the pair's figure is the
    bytes one node sends per second (one direction, all its GPUs and HCAs): gpus x size / slowest GPU.
    Latency: ping-pong of ``lat_size`` bytes on every GPU pair, one-way time, median over GPU pairs.
    Nodes outside ``pairs`` only join the barriers. Returns ``{(a, b): (bw_gbs, lat_us)}`` on every rank.
    """
    peer = first = None
    for a, b in pairs:
        if net.host in (a, b):
            peer = net.ranks_of[b if net.host == a else a][net.local]
            first = net.host == a

    if peer is not None:
        send, recv = net.buffer("send", size), net.buffer("recv", size)

        def both_ways():
            ops = [dist.P2POp(dist.isend, send, peer), dist.P2POp(dist.irecv, recv, peer)]
            for w in dist.batch_isend_irecv(ops):
                w.wait()

        small_send, small_recv = net.buffer("lsend", lat_size), net.buffer("lrecv", lat_size)

        def ping_pong():
            if first:
                dist.send(small_send, peer)
                dist.recv(small_recv, peer)
            else:
                dist.recv(small_recv, peer)
                dist.send(small_send, peer)
    else:
        both_ways = ping_pong = lambda: None

    bw_s = timed(both_ways, iters, warmup)
    lat_s = timed(ping_pong, lat_iters, 20) / 2
    mine = None if peer is None else (net.host, bw_s, lat_s)
    results = {}
    gathered = [r for r in net.all_gather(mine) if r]
    for a, b in pairs:
        rows = [r for r in gathered if r[0] in (a, b)]
        bw = net.gpus * size / max(r[1] for r in rows) / 1e9
        lat = statistics.median(r[2] for r in rows if r[0] == a) * 1e6
        results[(a, b)] = (round(bw, 3), round(lat, 2))
    return results


class JsonlWriter:
    """Rank 0 appends one JSON object per line and flushes it (partial sweeps stay readable)."""

    def __init__(self, net: Net, path: Path):
        self.f = None
        if net.rank == 0:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.f = open(path, "a")

    def write(self, row):
        if self.f:
            self.f.write(json.dumps(row) + "\n")
            self.f.flush()


def out_dir(arg):
    """``--out`` or ``$NET_OUT`` (the launcher sets ``$SCRATCH/trillium/nccl/<job>``)."""
    return Path(arg or os.environ.get("NET_OUT") or f"nccl/{os.environ.get('SLURM_JOB_ID', 'local')}")
