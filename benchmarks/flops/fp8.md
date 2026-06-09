## Overview

**Benchmark:** fp8
**dtype:** `torch.float8_e4m3fn`
**Matrix size:** 8192 x 8192
**Kernel:** `torch._scaled_mm` with tensor-wise scaling (scale=1.0)

### Config (config/synthetic.yaml)

```yaml
fp8:
  inherits: _flops
  argv:
    --number: 30
    --repeat: 90
    --m: 8192
    --n: 8192
    --dtype: fp8
```

### Notes

- Uses `torch._scaled_mm` instead of `torch.mm`; transposes the second operand.
- Requires Hopper (SM90+) or later for hardware fp8 tensor cores.
- Scale factors are fixed at 1.0 (no dynamic quantization overhead).
- Output stays in fp8 (no cast to higher-precision output).
- The `out` parameter is used to avoid allocation in the inner loop.

### Run standalone

```bash
python main.py --number 30 --repeat 90 --m 8192 --n 8192 --dtype fp8
```
