## Overview

## Config

- **Model**: `torchvision.models.convnext_large()`, random weights
- **Batch size**: 128
- **Loader**: `pytorch` (FakeImageNet)
- **Iterations**: 30 measurement samples after warmup skip

## Precision variants

Four configs exist for precision comparison:

| Config suffix | Precision | AMP | GradScaler |
|---------------|-----------|-----|------------|
| `-fp32`       | fp32      | No  | No         |
| `-fp16`       | fp16      | Yes | Yes        |
| `-tf32`       | tf32      | No  | No (TF32 matmul only) |
| `-tf32-fp16`  | tf32-fp16 | Yes | Yes        |

The 30-iteration stop keeps runs short -- the goal is comparing precision overhead,
not convergence.

## Quirks

- No `channel_last` optimization (unlike resnet50).
- fp16 falls back to bf16 on non-CUDA backends.
- tf32 is enabled via `accelerator.set_enable_tf32()`, which sets
  `torch.backends.cuda.matmul.allow_tf32 = True`.
