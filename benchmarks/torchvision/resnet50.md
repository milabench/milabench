## Overview

## Config

- **Model**: `torchvision.models.resnet50()`, random weights
- **Batch size**: 256
- **Precision**: tf32-fp16 (TF32 matmul + autocast fp16, GradScaler enabled)
- **Optimization**: `channel_last` memory format applied to both model and input tensors

## resnet50 (standard)

Uses `pytorch` loader reading from `FakeImageNet/train/` with standard augmentations.
60 measurement iterations after warmup skip.

## resnet50-noio

Uses `synthetic_fixed` loader: one random batch generated in RAM, repeated every iteration.
500 measurement iterations (vs 60 for standard). This makes it the purest compute-bound
measurement in the suite -- isolates GPU throughput from I/O and data augmentation overhead.

## Quirks

- `channel_last` is set on the model via `model.to(memory_format=channels_last)` AND on
  input tensors via the `transform` dict in `train_epoch`. Both are required for the
  optimization to take effect.
- fp16 autocast silently promotes to bf16 on non-CUDA devices (ROCm/XPU).
