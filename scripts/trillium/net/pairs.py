"""D2: bandwidth and latency of every node pair -> ``<out>/pairs.jsonl`` (``trillium.pairs.v1``).

Round-robin schedule (circle method): N - 1 rounds of N / 2 disjoint pairs (52 nodes: 51 rounds of
26 pairs, all 1,326 pairs). The pairs of a round run at once (``--sequential``: one pair at a time).
Each pair: GPU i of a <-> GPU i of b on all GPUs, both directions together (``netlib.exchange``).

    srun --ntasks-per-node=<gpus> python pairs.py [--size 1073741824] [--sequential]
"""

from __future__ import annotations

import argparse
import time

from netlib import GiB, JsonlWriter, Net, exchange, out_dir


def round_robin(nodes):
    """All pairs in rounds of disjoint pairs; with an odd count, one node sits out each round."""
    n = list(nodes) + ([None] if len(nodes) % 2 else [])
    rounds = []
    for _ in range(len(n) - 1):
        rounds.append([(n[i], n[-1 - i]) for i in range(len(n) // 2) if None not in (n[i], n[-1 - i])])
        n = [n[0], n[-1], *n[1:-1]]
    return rounds


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", help="directory of pairs.jsonl (default $NET_OUT)")
    parser.add_argument("--size", type=int, default=GiB, help="bytes per GPU per direction")
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--lat-iters", type=int, default=200)
    parser.add_argument("--sequential", action="store_true", help="one pair at a time, the rest idle")
    parser.add_argument("--rounds", type=int, help="stop after this many rounds (tests)")
    args = parser.parse_args()

    net = Net()
    out = JsonlWriter(net, out_dir(args.out) / "pairs.jsonl")
    tool = (f"torch.distributed NCCL {net.nccl_version} send/recv, GPU i <-> GPU i on all {net.gpus} GPUs, "
            f"both directions at once, {'one pair at a time' if args.sequential else 'whole round at once'}")
    rounds = round_robin(net.nodes)[:args.rounds]
    net.log(f"D2: {len(net.nodes)} nodes x {net.gpus} GPUs, {len(rounds)} rounds, NCCL {net.nccl_version}")
    for r, pairs in enumerate(rounds, 1):
        t0 = time.time()
        batches = [[p] for p in pairs] if args.sequential else [pairs]
        for batch in batches:
            for (a, b), (bw, lat) in exchange(net, batch, args.size, args.iters, lat_iters=args.lat_iters).items():
                out.write({"schema": "trillium.pairs.v1", "job_id": net.job_id, "round": r, "a": a, "b": b,
                           "tool": tool, "size_bytes": args.size, "bw_gbs": bw, "lat_us": lat, "lat_size_bytes": 8,
                           "time": time.time(), "status": "ok"})
        net.log(f"round {r}/{len(rounds)}: {len(pairs)} pairs in {time.time() - t0:.1f} s")
    net.close()


if __name__ == "__main__":
    main()
