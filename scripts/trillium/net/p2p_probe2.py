"""One-off: rounds of pairs.py with only a subset of each node's GPUs exchanging, twice over."""
import sys, os
sys.path.insert(0, os.environ["NET_DIR"])
import torch.distributed as dist
from netlib import GiB, Net, timed
from pairs import round_robin

net = Net()
size = GiB
send, recv = net.buffer("send", size), net.buffer("recv", size)
for rep in (1, 2):
    for pairs in round_robin(net.nodes):
        for active in ([0, 2, 4, 6], [1, 3, 5, 7], list(range(net.gpus))):
            peer = None
            for a, b in pairs:
                if net.host in (a, b):
                    peer = net.ranks_of[b if net.host == a else a][net.local]
            def both():
                ops = [dist.P2POp(dist.isend, send, peer), dist.P2POp(dist.irecv, recv, peer)]
                for w in dist.batch_isend_irecv(ops):
                    w.wait()
            fn = both if peer is not None and net.local in active else (lambda: None)
            s = timed(fn, 10, 3)
            rows = [r for r in net.all_gather((net.host, net.local, s) if fn is both else None) if r]
            out = []
            for a, b in pairs:
                t = [r[2] for r in rows if r[0] in (a, b)]
                out.append(f"{a}-{b} {len(active)*size/max(t)/1e9:6.1f} ({size/max(t)/1e9:.1f}-{size/min(t)/1e9:.1f})")
            net.log(f"rep {rep} active {active}: " + "  ".join(out))
net.close()
