## Overview

## Entry point

`main.py:main()` -> `mp.spawn(worker_main, nprocs=world_size)`.
Each spawned process runs `worker_main(rank, world_size, args)`.

## Process setup (`ddp_setup`)

Sets `MASTER_ADDR=localhost`, `MASTER_PORT=12355`, calls
`accelerator.init_process_group(backend=accelerator.ccl)`.
The backend token `ccl` is resolved by torchcompat to `nccl` on CUDA, `ccl` on XPU, etc.

## Training loop (`Trainer`)

1. Model instantiated via `torchvision_models.<name>()` (random weights).
2. Wrapped in `DDP(model, device_ids=[device])`. FSDP path is commented out.
3. Loss: `F.cross_entropy`. Optimizer: `SGD(lr=1e-3)`.
4. Precision: hardcoded `autocast(dtype=torch.bfloat16)` in `_run_batch`. No GradScaler.
5. Data loaded via `benchmate.dataloader.imagenet_dataloader()` with rank/world_size
   for `DistributedSampler`. Sampler epoch set each epoch for proper shuffling.

## Observer / instrumentation

Unlike `torchvision`, the `BenchObserver` is created directly in the `Trainer.__init__`,
not injected by voirfile probes:
- `earlystop=60`, `raise_stop_program=True`, `stdout=True`.
- Wraps the dataloader: `self.observer.loader(train_data)`.
- Loss logged via `self.observer.record_loss(loss.detach())`.

The voirfile is a stub -- it only yields `run_script` and catches `StopProgram`.
GPU monitoring is handled by `multigpu_monitor(poll_interval=3)` wrapping the
`mp.spawn` call in `main()`.

## benchfile.py

`TorchvisionBenchmarkDDP` overrides `build_run_plan` to pipe stdout through
`cmd.ActivatorCommand` + `.use_stdout()`. This is needed because `mp.spawn` workers
print metrics to stdout rather than through voir probes.

## Data pipeline

Same as `torchvision`: `imagenet_dataloader(args, model, rank, world_size)`.
Current config uses `--loader: torch` which maps to the `pytorch` path
(reads JPEG from `FakeImageNet/train/`), with a `DistributedSampler`.

## Gotchas

- Precision is NOT configurable at runtime despite `--precision` being parsed. The
  `_run_batch` method always uses `torch.bfloat16`. The arg is ignored.
- `mp.spawn` forks processes inside a single job. This means voir cannot probe into
  the workers -- all instrumentation must be self-contained in worker code.
- Worker exceptions are caught and printed but not re-raised (except `StopProgram`),
  so a crash in one rank can silently degrade results.
- `MASTER_PORT` is hardcoded to `12355`. Running multiple instances on the same node
  will collide.
- `destroy_process_group()` is called per-worker after training, but `StopProgram`
  skips it, which may leak resources on early stop.

## File layout

```
benchmarks/torchvision_ddp/
  main.py         # DDP training script (spawn-based)
  voirfile.py     # stub, no real instrumentation
  prepare.py      # generates FakeImageNet
  benchfile.py    # Package class with custom run plan for stdout capture
  requirements.in # torch, torchvision, torchcompat, tqdm, voir
```
