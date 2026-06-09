## Overview

## Config

- **Model**: `torchvision.models.regnet_y_128gf()`, random weights
- **Batch size**: 64
- **Precision**: tf32-fp16 (TF32 matmul + autocast fp16, GradScaler enabled)
- **Loader**: `pytorch` (FakeImageNet)

## Notes

Largest model in the torchvision group. The smaller batch size (64 vs 256 for
resnet50) reflects the higher per-sample memory cost of the 128GF RegNet variant.

Uses the same training loop and data pipeline as the other torchvision benchmarks.
No special optimizations (no `channel_last`, no `inductor`).

## Quirks

- fp16 autocast silently promotes to bf16 on non-CUDA devices.
- `accelerator.mark_step()` calls after backward/optimizer step are no-ops on CUDA
  but required for XLA/HPU correctness.
