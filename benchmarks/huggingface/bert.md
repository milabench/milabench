## Overview

## Config

- **Model**: `BertLargeForMaskedLM` via `AutoModelForMaskedLM.from_config()`, random weights
- **Batch size**: 32
- **Sequence length**: train_length / eval_length set in model registry
- **Iterations**: 30 measurement samples after warmup skip

## Precision variants

Four configs for precision comparison, same pattern as torchvision/convnext:

| Config suffix | Precision | AMP | GradScaler |
|---------------|-----------|-----|------------|
| `-fp32`       | fp32      | No  | No         |
| `-fp16`       | fp16      | Yes | Yes        |
| `-tf32`       | tf32      | No  | No (TF32 matmul only) |
| `-tf32-fp16`  | tf32-fp16 | Yes | Yes        |

## Data

Synthetic: random `input_ids` + `labels` of shape `(train_length,)` with values
in `[0, vocab_size)`. Pre-generated in RAM, served round-robin.

## Quirks

- Loss comes from `outputs.loss` (HF model computes loss internally when `labels` passed).
- fp16 falls back to bf16 on non-CUDA backends.
- `num_workers=8` in dataloader config but irrelevant -- all data is pre-generated in RAM.
