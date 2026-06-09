# Geometric GNN

Graph neural network benchmark for molecular property prediction using PyTorch Geometric.

## What it measures

GPU throughput on graph-structured workloads (message-passing and 3D geometric convolutions) with irregular, variable-size batches. Stresses scatter/gather kernels and sparse operations rather than dense matmuls.

## Variants

| Config name | Model | Input type | Default batch size |
|-------------|-------|------------|--------------------|
| `pna`       | PNA (Principal Neighbourhood Aggregation) | 2D molecular graphs | 4096 |
| `dimenet`   | DimeNet (Directional Message Passing) | 3D molecular geometry (atomic coords) | 16 |

PNA uses multi-aggregator message passing (mean/min/max/std) with degree-based scaling.
DimeNet uses spherical Bessel functions and angles between edges -- much heavier per-sample.

## Dataset

PCQM4Mv2Subset -- a configurable-size subset of OGB's PCQM4Mv2, predicting HOMO-LUMO gap from molecular graphs. 3D coordinates are extracted from SDF conformer data via RDKit. Default: 100k samples.

## Framework / dependencies

- `torch-geometric`, `torch-cluster`, `torch-sparse`, `torch-scatter`
- `rdkit` (SDF parsing for 3D coordinates)
- Requires CUDA compilation of PyG C++ extensions (`FORCE_CUDA=1` set in benchfile)

## Execution

Single-GPU only (`per_gpu`). No distributed training. No eval pass -- train-only loop.
