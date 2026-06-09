## Overview

## Entry points

- **benchfile.py** (`TimmBenchmarkPack`): Clones `huggingface/pytorch-image-models` at a pinned commit, then runs timm's `train.py` via `TorchRunCommand > VoirCommand > PackCommand`.
- **voirfile.py** (`instrument_main`): The core instrumentation. Overrides four objects inside timm's training loop using `voir` probes.
- **prepare.py**: Calls `benchmate.datagen.generate_fakeimagenet()` to create synthetic data.

## Code flow

1. `benchfile.py` clones timm into `{code}/pytorch-image-models/` if missing.
2. `build_run_plan` constructs: `torchrun ... voir ... train.py --amp --data-dir ... --dataset FakeImageNet ...`
3. `voirfile.py` waits for `load_script` phase, then patches timm internals:
   - **Data loader**: `create_loader()` return value is wrapped by `BenchObserver.loader` to measure throughput.
   - **Loss function**: `train_one_epoch > loss_fn` is wrapped by `BenchObserver.criterion`.
   - **Optimizer**: `train_one_epoch > optimizer` is wrapped by `BenchObserver.optimizer`.
   - **Saver**: `main > saver` is replaced with a no-op lambda to skip checkpoint writes.
4. After `skip` warmup iterations, rate logging begins. After `stop` measurements, `early_stop` fires `StopProgram`.

## Instrumentation details

- `BenchObserver` uses `accelerator.Event` for GPU-side timing and calls `accelerator.mark_step` after backward and step (needed for lazy-mode backends like HPU).
- `batch_size_fn=lambda x: len(x[0])` -- batch size is inferred from the first element of the data tuple.
- GPU monitoring (`monitor_monogpu` or `multigpu_monitor`) is selected based on `RANK` env var.
- Early stop signal is only emitted from rank 0.

## Probe paths (voir)

These must match timm's internal symbol names. If timm refactors its code, these break silently:

```
/timm.data.loader/create_loader() as loader
//train_one_epoch > loss_fn
//train_one_epoch > optimizer
//main > saver
```

## Config knobs (from training.yaml)

- `--amp-dtype bfloat16`: Mixed precision type.
- `--batch-size auto_batch(128)`: Auto-scaled batch size (128 is the base).
- `--workers auto({n_worker}, 8)`: Data loader workers, auto-detected.
- `--val-split ''`: Validation is disabled (empty string).
- `--dataset FakeImageNet`: Synthetic data, no real ImageNet needed.

## Gotchas

- The pinned commit (`BRANCH = "cb0e4391..."`) determines which timm version runs. Updating it may break the voir probe paths.
- `--checkpoint-hist 1` is passed from argv but the saver is also no-op'd by the voirfile -- belt and suspenders.
- `working_directory` is set to the cloned repo, not the benchmark dir. Relative paths in timm resolve from there.
- No `main.py` exists in benchmarks/timm/; the main script is `pytorch-image-models/train.py`.

## Dependencies

`torch`, `torchvision`, `pyyaml`, `huggingface_hub`, `safetensors`, `voir`, `benchmate`, `torchcompat`
