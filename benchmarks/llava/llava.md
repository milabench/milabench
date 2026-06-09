## Overview

**Benchmark:** llava (llava-single / llava-gpus)
**Model:** llava-hf/llava-1.5-7b-hf
**Task:** multimodal fine-tuning (VQA)
**Metric:** samples/s

### Config (config/training.yaml)

```yaml
_llava:
  inherits: _defaults
  definition: ../benchmarks/llava
  plan:
    method: per_gpu
  argv:
    --batch_size: "auto_batch(1)"
    --num_workers: "auto({n_worker}, 4)"
    --gradient_accumulation_steps: 1

llava-single:
  inherits: _llava
  plan:
    method: per_gpu

llava-gpus:
  inherits: _llava
  plan:
    method: njobs
    n: 1
```

### Architecture

1. `prepare.py` downloads model, processor, and dataset.
2. `main.py` loads LLaVA with `device_map` and wraps with Accelerate.
3. Iterates batches from The Cauldron (aokvqa), applies a chat template
   to format conversations as `<image>\nHuman: ...\nAssistant: ...`.
4. Processes images + text through `AutoProcessor`, trains with cross-entropy.
5. `BenchObserver` records throughput and early-stops at 70 steps.

### Multi-GPU

`llava-gpus` uses `AccelerateAllNodes` - launches a single Accelerate job
spanning all GPUs on the node. Model uses `device_map` for sharding.

### Run standalone

```bash
accelerate launch main.py --batch_size 1 --num_workers 4
```
