# FNO Benchmark

Fourier Neural Operator (FNO) training benchmark for scientific computing
workloads. Wraps the [operator_learning](https://github.com/Delaunay/operator_learning)
library (milabench branch) to train neural operators on physics simulation data.

## What it measures

Single-GPU training throughput for neural operator architectures solving PDEs.
Reports samples/sec through the training loop.

## Variants (`benchmarks/fno_benchmark/dev.yaml`)

| Name | Problem | Data file |
|------|---------|-----------|
| `pic1d` | 1D electrostatic plasma (PIC) | PIC1D_electrostatic.h5 |
| `pic2d` | 2D electrostatic plasma (PIC) | PIC2D_electrostatic.h5 |
| `rbc2d` | 2D Rayleigh-Benard convection (Dedalus) | RBC2D_256x64_Ra1e7_dt1e-3_update.h5 |
| `rbc3d` | 3D Rayleigh-Benard convection (pySDC) | RBC3D_64x64x32_Ra1e5_dt0_5_solution.h5 |

## Key dependencies

torch, h5py, scipy, numpy, finufft, configmypy, mpi4py, cantilever, accelerate,
calflops, opt_einsum

## Notes

- Uses AMP (`--use_amp 1`) but not complex AMP.
- Dataset from HuggingFace: `chelseajohn/FNOBenchmark` (requires git-lfs).
- Source code is cloned from an external repo at install time, not vendored.
- Single-GPU only (`plan: njobs, n: 1`, tag: monogpu).
- `torch.compile` is disabled by default (`--compile_train 0`).
