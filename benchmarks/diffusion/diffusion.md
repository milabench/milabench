## Overview

## Entry points

- **benchfile.py** (`Diffusion`): Runs `main.py` via `AccelerateAllNodes`. Handles `HF_TOKEN` / `MILABENCH_HF_TOKEN` env propagation. No voirfile -- `VoirCommand` is explicitly disabled (`if False:`).
- **main.py**: Self-contained training script with model loading, dataset prep, and training loop.
- **prepare.py**: Downloads model weights (CLIP, VAE, UNet, tokenizer, scheduler) and the naruto-blip-captions dataset from HuggingFace Hub.

## Code flow

1. `main()` bumps `RLIMIT_NOFILE` to the hard limit (many open files from data workers + HF cache).
2. `prepare_voir()` creates `BenchObserver(earlystop=60, raise_stop_program=True)`.
3. `train()` initializes `Accelerator(mixed_precision="bf16")`.
4. Dataset is loaded, images are transformed (resize to 512, crop, normalize to [-1,1]), captions are tokenized via CLIP tokenizer.
5. VAE and CLIP text encoder are frozen and cast to bf16. Only UNet is trained.
6. `accelerator.prepare()` wraps UNet, optimizer, dataloader, and LR scheduler.
7. Training loop per step:
   - VAE encodes pixel values to latents, scaled by `vae.config.scaling_factor`.
   - Random noise and random timesteps are sampled.
   - Noisy latents are created via `noise_scheduler.add_noise()`.
   - CLIP encodes captions to `encoder_hidden_states`.
   - UNet predicts noise from noisy latents + timesteps + text conditioning.
   - MSE loss between predicted noise and actual noise.
   - `accelerator.backward()`, optimizer step, LR step, zero grad.

## Instrumentation approach

- No voirfile is used. `BenchObserver` is created directly in `main.py`.
- `observer.iterate(loader)` wraps the dataloader to measure throughput and trigger early stop.
- `batch_size_fn=lambda x: x["pixel_values"].shape[0]` -- batch size from the pixel tensor.
- `stdout=True` outputs metrics to stdout; `benchfile.py` uses `.use_stdout()` to collect them.
- `StopProgram` exception is caught in `main()` to exit cleanly.

## Dataset sampling trick

The naruto dataset is small (~1k images). To get enough training steps:
```python
total_samples = args.batch_size * 70 * WORLD_SIZE
sampler = RandomSampler(train_dataset, replacement=True, num_samples=total_samples)
```
This oversamples with replacement so each epoch has a fixed, predictable number of steps regardless of dataset size. `shuffle=False` is required when using a custom sampler.

## Accelerate integration

- `AccelerateAllNodes` in benchfile handles multi-GPU/multi-node launch.
- `gradient_accumulation_steps=1` by default (configurable).
- LR scheduler: `constant` with 500 warmup steps (warmup is irrelevant given early stop at 60 iterations).
- `scale_lr=True` in Arguments dataclass but not actually used in `train()`.

## Gotchas

- `variant` is declared twice in the `Arguments` dataclass (lines 29 and 42). The second silently overwrites the first.
- The voirfile is intentionally skipped (`if False:` in benchfile). All observation happens in-process.
- `prepare.py` downloads real model weights (~5GB). Needs network access and potentially `HF_TOKEN` for gated models.
- `persistent_workers=True` in the dataloader keeps workers alive across epochs -- important for throughput.
- `lr_warmup_steps=500` and `lr_scheduler="constant"` mean the LR is effectively constant from step 0 (constant scheduler ignores warmup).

## Dependencies

`diffusers`, `accelerate`, `transformers`, `datasets`, `torch`, `torchvision`, `argklass`, `voir`, `benchmate`
