"""D4: bisection -> ``<out>/collectives.jsonl`` (``trillium.collectives.v1``, ``experiment: D4``).

Each cut splits the nodes into two random halves and pairs them off at random; all pairs exchange at
once (GPU i <-> GPU i, both directions, ``netlib.exchange``). With an odd node count one node sits out.
``busbw_gbs`` is the median over pairs of the bytes one node sends per second. ``--gpus``: only these GPUs
of each node exchange, as in ``pairs.py`` (on B200, 0,2,4,6: one per NIC).

    srun --ntasks-per-node=<gpus> python bisection.py [--cuts 3] [--seed 0] [--gpus 0,2,4,6]
"""

from __future__ import annotations

import argparse
import random
import statistics
import time

from netlib import GiB, JsonlWriter, Net, exchange, out_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", help="directory of collectives.jsonl (default $NET_OUT)")
    parser.add_argument("--size", type=int, default=GiB, help="bytes per GPU per direction")
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--cuts", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpus", help="comma-separated GPU indices of a node that exchange (default all)")
    args = parser.parse_args()

    net = Net()
    out = JsonlWriter(net, out_dir(args.out) / "collectives.jsonl")
    gpus = [int(g) for g in args.gpus.split(",")] if args.gpus else list(range(net.gpus))
    half = len(net.nodes) // 2
    net.log(f"D4: {2 * half} nodes x {net.gpus} GPUs, GPUs {args.gpus or 'all'} exchange, {args.cuts} cuts")
    for cut in range(1, args.cuts + 1):
        order = list(net.nodes)
        random.Random(args.seed * 1000 + cut).shuffle(order)
        pairs = list(zip(order[:half], order[half:2 * half]))
        res = exchange(net, pairs, args.size, args.iters, gpus=gpus)
        halves = [sorted(order[:half]), sorted(order[half:2 * half])]
        out.write({"schema": "trillium.collectives.v1", "experiment": "D4", "job_id": net.job_id,
                   "collective": "bisection", "cut": cut, "nodes": 2 * half, "gpus": 2 * half * len(gpus),
                   "hosts": halves[0] + halves[1], "halves": halves,
                   "pairs": [{"a": a, "b": b, "bw_gbs": bw, "lat_us": lat} for (a, b), (bw, lat) in res.items()],
                   "size_bytes": args.size, "busbw_gbs": round(statistics.median(bw for bw, _ in res.values()), 3),
                   "algbw_gbs": None, "time_us": None, "repetition": 1, "time": time.time(),
                   "nccl_version": net.nccl_version, "status": "ok"})
        net.log(f"cut {cut}: median {statistics.median(bw for bw, _ in res.values()):.1f} GB/s per node, "
                f"min {min(bw for bw, _ in res.values()):.1f}")
    net.close()


if __name__ == "__main__":
    main()
