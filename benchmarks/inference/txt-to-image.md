## Overview

Text-to-image generation using the FLUX.1-dev diffusion model. Measures
denoising step throughput, not full image generation rate.

## Model and data

- Model: `FluxPipeline.from_pretrained` (FLUX.1-dev), bfloat16, `device_map="cuda"`
- Dataset: webdataset from `jackyhate/text-to-image-2M` (10 tar shards, non-streaming)
- Prompts truncated to 70 chars
- Default generation: 256x256, 50 inference steps, guidance_scale=3.5
- batch_size: 16

## Code path

`main.py:main()` parses `Arguments` via argklass, dispatches to
`FluxBenchmark` (mode=flux). Inherits from `InferenceBenchmark`.

Key methods:
- `load_model()`: loads FluxPipeline
- `load_dataset()`: downloads and loads webdataset tar shards
- `run()`: runs diffusion pipeline with `callback_on_step_end` for per-step measurement

## Instrumentation

No voirfile. `BenchObserver` with `custom_step=True` -- does NOT auto-step
on iteration. Instead `on_step()` is called per denoising step via
`callback_on_step_end`. `earlystop=65` batches.

## Gotchas

- Measures denoising steps, not full images; comparing to other image benchmarks is misleading
- VRAM-intensive; cpu_offload code is commented out
- `torch.compile` is commented out
- `--prepare` flag short-circuits: loads data + model then exits
