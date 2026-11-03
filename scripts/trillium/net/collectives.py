"""D3: NCCL collectives vs nodes -> ``<out>/collectives.jsonl`` (``trillium.collectives.v1``, ``experiment: D3``).

all_reduce, all_gather, reduce_scatter, alltoall (fp32) on the first k nodes of the job, k in
``--nodes`` (default 1, 2, 4, 8, 13, 26, 52, cut to the job's size, plus the whole job), 1 GiB, with
``--repetitions``; on the ``--sweep-nodes`` (default 2 and the whole job) the first repetition sweeps
message sizes. The other nodes idle meanwhile. Sizes, time and bandwidths follow nccl-tests: size = the
largest buffer of a rank, time = mean over ranks, algbw = size / time, busbw = algbw x (n - 1) / n
(x 2 for all_reduce).

    srun --ntasks-per-node=<gpus> python collectives.py [--nodes 1,2,4] [--sweep-nodes 2]
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch
import torch.distributed as dist

from netlib import GiB, JsonlWriter, Net, out_dir, timed

BUS_FACTOR = {"all_reduce": lambda n: 2 * (n - 1) / n, "all_gather": lambda n: (n - 1) / n,
              "reduce_scatter": lambda n: (n - 1) / n, "alltoall": lambda n: (n - 1) / n}
SWEEP = [2**k for k in range(10, 33, 2)]  # 1 KiB - 4 GiB


def op(net: Net, coll, size, group, n):
    """The collective on ``size`` bytes (rounded down to a multiple of n fp32 values); returns (fn, bytes)."""
    count = max(1, size // (4 * n))
    full, part = net.buffer("full", 4 * n * count, torch.float32), net.buffer("part", 4 * count, torch.float32)
    if coll == "all_reduce":
        return lambda: dist.all_reduce(full, group=group), full.nbytes
    if coll == "all_gather":
        return lambda: dist.all_gather_into_tensor(full, part, group=group), full.nbytes
    if coll == "reduce_scatter":
        return lambda: dist.reduce_scatter_tensor(part, full, group=group), full.nbytes
    out = net.buffer("full2", 4 * n * count, torch.float32)
    return lambda: dist.all_to_all_single(out, full, group=group), full.nbytes


def iterations(size):
    return (50, 10) if size <= 2**20 else (20, 5) if size <= 2**26 else (10, 3)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", help="directory of collectives.jsonl (default $NET_OUT)")
    parser.add_argument("--nodes", default="1,2,4,8,13,26,52")
    parser.add_argument("--sweep-nodes", default="2,all")
    parser.add_argument("--collectives", default=",".join(BUS_FACTOR))
    parser.add_argument("--size", type=int, default=GiB)
    parser.add_argument("--repetitions", type=int, default=2)
    args = parser.parse_args()

    net = Net()
    out = JsonlWriter(net, out_dir(args.out) / "collectives.jsonl")
    total = len(net.nodes)
    counts = sorted({k for k in map(int, args.nodes.split(",")) if k <= total} | {total})
    sweep = {total if k == "all" else int(k) for k in args.sweep_nodes.split(",")}
    # new_group is collective: every rank creates every group, in the same order.
    groups = {k: None if k == total else
              dist.new_group([r for h in net.nodes[:k] for r in net.ranks_of[h]], device_id=net.device)
              for k in counts}
    net.log(f"D3: {total} nodes x {net.gpus} GPUs, node counts {counts}, size sweep at {sorted(sweep)}")
    for k in counts:
        hosts, n, member = net.nodes[:k], k * net.gpus, net.host in net.nodes[:k]
        for rep in range(1, args.repetitions + 1):
            sizes = SWEEP if k in sweep and rep == 1 else [args.size]
            for coll in args.collectives.split(","):
                t0 = time.time()
                for size in sizes:
                    fn, nbytes = op(net, coll, size, groups[k], n) if member else (lambda: None, 0)
                    iters, warmup = iterations(size)
                    sec = timed(fn, iters, warmup)
                    times = [s for s in net.all_gather(sec if member else None) if s is not None]
                    mean = statistics.mean(times)
                    alg = nbytes / mean / 1e9 if member else 0
                    out.write({"schema": "trillium.collectives.v1", "experiment": "D3", "job_id": net.job_id,
                               "collective": coll, "nodes": k, "gpus": n, "hosts": hosts, "size_bytes": size,
                               "bytes": nbytes, "time_us": round(mean * 1e6, 2), "algbw_gbs": round(alg, 3),
                               "busbw_gbs": round(alg * BUS_FACTOR[coll](n), 3), "time_max_us": round(max(times) * 1e6, 2),
                               "cut": None, "repetition": rep, "time": time.time(),
                               "nccl_version": net.nccl_version, "status": "ok"})
                net.log(f"{k} nodes rep {rep} {coll}: {len(sizes)} sizes in {time.time() - t0:.1f} s")
    net.close()


if __name__ == "__main__":
    main()
