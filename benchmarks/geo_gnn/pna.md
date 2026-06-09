## Overview

## Config

- **Model**: PNA (Principal Neighbourhood Aggregation)
- **Batch size**: 4096 graphs per mini-batch
- **Architecture**: 64 hidden channels, 64 layers, 4 aggregators x 3 scalers
- **Task**: L1 loss on normalized HOMO-LUMO gap prediction (PCQM4Mv2Subset)

## 2D message-passing

PNA operates on 2D molecular graphs (no 3D coordinates). Forward pass returns
per-node embeddings, pooled via `global_max_pool` to graph-level predictions.

## Degree histogram

`train_degree()` computes the in-degree histogram over the full training set
(two full passes on CPU) before training starts. Required by PNA's degree-based
scalers. Can add significant startup time on large subsets.

## Computational profile

- **Bottleneck**: scatter/gather operations in message passing
- **VRAM**: ~40 GB on H100 at batch_size=4096
- **Throughput**: ~8k samples/s typical

## Quirks

- 64 layers is unusually deep for a GNN; intentional stress test.
- Batch size counted as number of graphs (from PyG batch vector), not number of nodes.
- No validation or evaluation -- purely measuring training throughput.
- Y targets are normalized using dataset mean/std computed at startup (another full pass).
