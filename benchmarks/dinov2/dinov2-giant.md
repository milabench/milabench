## Overview

## Architecture

- Model: ViT-Giant/14 (vitg14 config, ~1.1B params)
- SSL method: self-distillation with global/local crops (DINO + iBOT style)
- Student network + EMA teacher (no gradient on teacher)
- Patch size 14, input resolution 384x384 (3 global crops + local crops)

## Computational profile

- Dual forward pass per step: student through encoder, teacher (EMA) through frozen copy
- FSDP sharding across GPUs; the benchmark patches out `reshard_fsdp_model` and fixes `fsdp_synchronize_streams` with a `torch.cuda.synchronize()` fallback
- Mixed precision not explicitly set in config (framework default)
- Optimizer: AdamW (from dinov2 internals)

## Code flow

1. `main.py`: appends `src/` to path, calls `dinov2.train.train.main(args)`
2. `benchfile.py`: clones dinov2 into `src/`, sets `TorchrunAllNodes` launcher
3. `voirfile.py`: the real orchestration entry point during `voir` runs

## voirfile instrumentation

- Patches `SSLMetaArch.fsdp_synchronize_streams` to use `torch.cuda.synchronize()` instead of stream sync (fixes FSDP hang)
- Patches out `reshard_fsdp_model` (set to no-op lambda)
- Overrides `_is_slurm_job_process()` to return False (runs without SLURM)
- Overrides `_parse_dataset_str()` to redirect to `FakeImageNet/train` ImageFolder
- Probes: loader (via `make_data_loader`), loss (`losses_reduced`), optimizer (`build_optimizer`)
- Batch size extracted from `collated_global_crops.shape[0]`
- Early stop after `skip + stop` iterations (default: 65 batches)

## Data pipeline

- `prepare.py`: generates synthetic ImageNet via `benchmate.datagen.generate_fakeimagenet`
  - 384x384 images, batch_count=60 batches worth
  - Creates hard links to fill DINOv2's expected metadata (split lengths)
  - Calls `dataset.dump_extra()` for DINOv2's custom metadata format
- At runtime, `ImageFolder` is used (override in voirfile redirects to `FakeImageNet/train`)

## Gotchas

- The dinov2 source is a pinned fork commit (`451bc15`), not the upstream repo
- SLURM detection is patched out; multi-node uses torchrun env vars instead
- `working_directory` is set to `src/` so relative imports in dinov2 resolve correctly
- The `_parse_dataset_str` override is critical: without it, dinov2 tries its custom ImageNet class
- If `SLURM_JOB_ID` env var is set at prepare time, it's explicitly deleted

## Loss

- DINOv2 internal multi-component loss (cross-entropy distillation + iBOT masked patch prediction)
- Reported via `losses_reduced` probe

## Known bottlenecks

- Memory: dual encoder forward + FSDP overhead; ViT-g is ~1.1B params x2 (student + teacher)
- Data loading: multi-crop augmentation is CPU-heavy (multiple random crops per image)
- The `torch.cuda.synchronize()` patch adds a sync point per step (correctness over speed)
