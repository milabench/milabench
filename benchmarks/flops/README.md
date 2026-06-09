# Synthetic FLOPS

Synthetic matrix-multiplication benchmark measuring raw compute throughput
in TFLOPS for various floating-point precisions.

## What it measures

Runs repeated `torch.mm` (or `torch._scaled_mm` for fp8) on square matrices
and reports sustained TFLOPS. No model weights, no data loading - pure ALU
stress test.

## Variants

| Name | dtype | MM kernel | Notes |
|------|-------|-----------|-------|
| fp8  | float8_e4m3fn | `_scaled_mm` (tensor-wise) | Requires SM90+ (Hopper) |
| fp16 | float16 | `torch.mm` | |
| bf16 | bfloat16 | `torch.mm` | |
| tf32 | float32 + TF32 | `torch.mm` | TF32 tensor cores enabled |
| fp32 | float32 | `torch.mm` | Full IEEE754 single precision |

## Key parameters

- `--m` / `--n`: matrix dimensions (default 8192x8192 for all variants)
- `--number`: inner-loop iterations per timing sample
- `--repeat`: number of timing samples

## Metric

`rate` in TFLOPS, computed as `N * (2*m*n*n + 2*m*n*n) / elapsed / 1e12`.
Each inner iteration performs two matmuls (dual-out ping-pong to avoid allocation).
