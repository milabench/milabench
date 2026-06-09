## Overview

## Architecture

- Encoder: ViT-Huge/16 (`vit_huge` in config), patch size 16, crop 224x224
- Predictor: 12 layers, embed dim 384, uses mask tokens
- Target encoder: EMA copy of main encoder (momentum schedule 0.998 -> 1.0)
- Input: 16 frames, tubelet size 2, sampling rate 4
- Masking: multiblock3d (8 small + 2 large spatiotemporal blocks)

## Computational profile

- Three-network forward: target_encoder(clips) + encoder(clips, masks) + predictor(z, h, masks)
- Target encoder is frozen (no grad), updated via EMA after each step
- Mixed precision: bfloat16 with GradScaler
- Optimizer: AdamW with cosine LR schedule, weight decay schedule, gradient clipping (10.0)
- DDP with `static_graph=True` for all three networks when RANK is set

## Loss function

- L1 loss (`loss_exp=1.0`): `mean(|predicted - target|)` averaged over mask pairs
- Regularization: variance collapse prevention via `relu(1 - std(z))`, coeff=0.0 in default config
- Total loss = JEPA loss + reg_coeff * reg_loss

## Code flow

1. `main()` in `main.py`: parses args, loads `config/vith16.yaml`, overrides data/batch params
2. Calls `acc.init_process_group()` if distributed, then `_main(params)`
3. `_main()`: full training loop inline (not imported from JEPA repo)
   - Inits encoder/predictor/target_encoder via JEPA's `init_video_model`
   - Creates mask collator (multiblock3d or random tube)
   - Creates video dataloader via JEPA's `init_data`
   - Wraps loader with `BenchObserver` (earlystop=65, stdout=True)
   - Training loop: load_clips -> train_step -> EMA update -> log

## voirfile.py

Minimal: just calls `voirfile_monitor(ov, options)` for GPU polling.
All instrumentation is handled directly in `main.py` via `BenchObserver`.

## Data pipeline

- `prepare.py`: generates 1000 random MP4 videos using OpenCV
  - 640x480, 300 frames, 30fps, random pixel noise
  - Parallelized with `multiprocessing.Pool` (up to 16 workers)
  - Writes CSV manifest (space-delimited: `path label`)
  - `MILABENCH_TESTING_PREPARE` env var can reduce count for CI
- `main.py` rewrites manifest paths to absolute at runtime (`generate_absolute_metainfo`)
- JEPA's `VideoDataset` + `MaskCollator` handle decoding and masking

## Gotchas

- `main.py` inlines the entire JEPA training loop rather than calling upstream's `main()`
  - This gives full control over the observer integration but means upstream changes aren't picked up
- The manifest CSV uses space delimiter (not comma) -- this is JEPA's expected format
- `generate_absolute_metainfo` rewrites paths because JEPA's manifest uses absolute paths
  but milabench relocates data directories
- `PT_HPU_LAZY_MODE=0` is set in env (Gaudi/HPU support, forces eager mode)
- The observer is placed after dataloader init but wraps the loader iterator directly
- `acc.mark_step()` calls are present for XLA/HPU compatibility (no-op on CUDA)

## Known bottlenecks

- Video decoding: 16 frames per clip from MP4 is CPU-intensive
- Memory: three ViT-Huge models in memory (encoder + predictor + target_encoder)
- Mask application involves gather/scatter operations on variable-length token subsets
- DDP all-reduce on three separate networks per step
- `AllReduce.apply` for input variance logging adds an extra sync per iteration
