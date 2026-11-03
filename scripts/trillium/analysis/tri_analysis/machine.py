"""Trillium B200 partition: constants shared by the parsers, the synthetic data and the figures."""

from __future__ import annotations

CLUSTER = "trillium"
GPUS_PER_NODE = 8
ALL_NODES = [f"trig{101 + i:04d}" for i in range(52)]  # trig0101 - trig0152
# Nodes we expect to use. trig0148-trig0152 may stay with SciNet's inference service: set 47 then.
N_NODES = 52
NODES = ALL_NODES[:N_NODES]
# /etc/hosts puts the nodes in 8 subnets (racks?), in host order. The switch layout is unknown.
SUBNET_SIZES = [7, 7, 7, 7, 4, 7, 7, 6]
SUBNET_OF = {}
_i = 0
for _g, _n in enumerate(SUBNET_SIZES, 1):
    for _h in ALL_NODES[_i:_i + _n]:
        SUBNET_OF[_h] = f"s{_g}"
    _i += _n

# MFU denominator: B200 dense BF16 *spec* peak, for every precision recipe (MFU stays in BF16
# units so that recipes compare on one scale). SciNet measured 1,668 TFLOP/s BF16 and 3,041
# TFLOP/s FP8 on 8192^2 matmuls.
PEAK_BF16 = 2.25e15
MATMUL_BF16 = 1668e12
MATMUL_FP8 = 3041e12
NODE_NET_GBS = 100.0  # 800 Gb/s per node
