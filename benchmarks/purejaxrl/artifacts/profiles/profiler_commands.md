# Profiler Commands

## JAX Profiler Trace

```bash
cd /tmp/milabench/benchmarks/purejaxrl
PYTHON=/tmp/output/venv/torch/bin/python
$PYTHON profile_dqn.py
```

Produces: `/tmp/artifacts/profiles/jax_trace/plugins/profile/<timestamp>/perfetto_trace.json.gz`

View with: https://ui.perfetto.dev/  (upload the .json.gz file)

## Key findings (200k steps, 2026-03-24)

Total execution time: 1827ms

| Operation             | Calls | Total(ms) | Avg(ms) | % of total |
|-----------------------|-------|-----------|---------|------------|
| cond.108 (learn cond) | 1562  | 1359.9    | 0.871   | 74.5%      |
| volta_sgemm_128x64_nn | 844   | 440.1     | 0.521   | 24.1%      |
| loop_slice_fusion     | 210   | 301.2     | 1.434   | 16.5%      |
| volta_sgemm_128x64_nt | 211   | 149.3     | 0.708   | 8.2%       |
| cond.107 (target upd) | 1562  | 137.5     | 0.088   | 7.5%       |
| All sgemm kernels     | ~2321 | ~728      | -       | ~39.9%     |

## Bottleneck analysis

- **110 gradient updates** (learning_starts + buffer fill) × ~11ms each = ~1210ms
- Per learning call breakdown:
  - GEMM kernels: ~6.6ms (58%)
  - Buffer sampling (loop_slice_fusion): ~2.74ms (23%)
  - Other (cond overhead, optimizer): ~2.3ms (19%)

## Hypothesis

Using FP16 for GEMM operations in the learning phase should give ~2.3x GEMM speedup
(confirmed by micro-benchmark: FP32=0.69ms, FP16=0.30ms for (65536,400)x(400,120) GEMM).

Expected improvement: GEMM time 728ms → ~316ms = saving ~412ms = ~22% total speedup.
