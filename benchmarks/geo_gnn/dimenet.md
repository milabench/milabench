## Overview

## Config

- **Model**: DimeNet (Directional Message Passing)
- **Batch size**: 16
- **Architecture**: 64 hidden channels, 6 blocks, 7 spherical / 6 radial basis functions, cutoff 10 A
- **Task**: L1 loss on normalized HOMO-LUMO gap prediction (PCQM4Mv2Subset)
- **Flag**: `--use3d` enables 3D coordinate usage

## 3D spherical Bessel

DimeNet uses atomic numbers (`z`) and 3D coordinates (`pos`) from SDF conformer data.
The spherical Bessel basis computation is O(edges^2) per graph, which is why batch
size is much smaller than PNA (16 vs 4096).

## RDKit dependency

The data pipeline (`pcqm4m_subset.py`) uses `rdkit.Chem.SDMolSupplier` to parse 3D
positions from ~3 GB of SDF data. Download/processing failures are usually RDKit or
network issues.

## Computational profile

- **Bottleneck**: spherical Bessel + bilinear layers
- **VRAM**: ~5 GB on H100 at batch_size=16
- **Throughput**: ~350 samples/s typical

## Quirks

- `FORCE_CUDA=1` in `benchfile.py` means install will fail on CPU-only machines.
- No validation or evaluation -- purely measuring training throughput.
- Forward pass outputs graph-level predictions directly (no pooling needed).
