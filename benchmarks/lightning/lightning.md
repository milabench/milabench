## Overview

## Entry points

- **benchfile.py** (`LightningBenchmark`): Runs `main.py` via `TorchrunAllNodes`. Always uses torchrun even for single-GPU (commented-out `TorchrunAllGPU` path).
- **main.py**: Self-contained training script. Defines model, dataloader, observer, and Lightning Trainer.
- **voirfile.py**: Minimal -- just yields `init` and `run_script` phases. All instrumentation lives in `main.py` instead.
- **prepare.py**: Generates `FakeImageNet` via `benchmate.datagen.generate_fakeimagenet()`.

## Code flow

1. `main()` parses args, creates a `torchvision_models.<model>()` instance.
2. Model is wrapped in `TorchvisionLightning(L.LightningModule)` which defines `training_step` and `configure_optimizers`.
3. `prepare_voir()` creates a `BenchObserver` (earlystop=100, stdout=True, raise_stop_program=False).
4. The dataloader is wrapped: `observer.loader(imagenet_dataloader(...))`.
5. `L.Trainer` is created with `precision="bf16-mixed"`, `max_steps=120`, checkpointing off, progress bar off.
6. `trainer.fit()` runs inside a `bench_monitor()` context for GPU stats.

## Instrumentation approach

Unlike timm, lightning does NOT use voir probes to intercept internals. Instead:
- `BenchObserver` wraps the dataloader directly in `main.py`.
- `raise_stop_program=False` means training runs until `max_steps` (120) rather than early-stopping on rate count.
- `stdout=True` prints metrics to stdout for collection.
- The voirfile is nearly empty -- it exists only to satisfy the voir framework's expectations.

## Lightning Trainer config

```python
L.Trainer(
    accelerator="auto",       # GPU auto-detection
    devices=n,                 # local_world_size GPUs
    num_nodes=nnodes,          # derived from WORLD_SIZE / LOCAL_WORLD_SIZE
    strategy="auto",           # Lightning picks DDP/FSDP/etc
    max_epochs=args.epochs,    # from config (default 10)
    precision="bf16-mixed",
    max_steps=120,             # hard cap, overrides epochs in practice
    reload_dataloaders_every_n_epochs=1,
)
```

## HPU quirk

Line 8: `os.environ["PT_HPU_LAZY_MODE"]` is set based on `WORLD_SIZE`. When `WORLD_SIZE <= 0` (no distributed), lazy mode is enabled (1); otherwise disabled (0). This is HPU-specific and harmless on CUDA.

## Gotchas

- `max_steps=120` is hard-coded in `main.py`, not configurable via YAML. This caps the actual training regardless of epoch count.
- `reload_dataloaders_every_n_epochs=1` forces Lightning to re-create the dataloader each epoch, which re-triggers the observer wrapper.
- `accelerator.set_enable_tf32(True)` is called but precision is bf16-mixed -- TF32 applies to the non-mixed ops.
- `benchfile.py` always uses `TorchrunAllNodes` with `.use_stdout()` even for single-GPU, relying on torchrun to handle the trivial 1-node case.

## Dependencies

`torch`, `torchvision`, `lightning`, `torchcompat`, `voir`, `benchmate`
