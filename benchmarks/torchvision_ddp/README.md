# Torchvision DDP

Multi-GPU image classification training using PyTorch DistributedDataParallel (DDP).
Measures multi-GPU scaling throughput for convolutional networks on synthetic ImageNet
data. Complements the single-GPU `torchvision` benchmark.

## Concrete benchmarks

| Name | Model | Batch size | Precision | GPUs |
|------|-------|------------|-----------|------|
| `resnet152-ddp-gpus` | ResNet-152 | 256 | bf16 (hardcoded) | all available |

## Data

Same `FakeImageNet` dataset as `torchvision`. Prepared by `prepare.py` using
`benchmate.datagen.generate_fakeimagenet()`.

## Scheduling

Runs with `plan.method: njobs, n: 1` -- a single process that internally calls
`mp.spawn()` to fork one worker per GPU. Requires `gpu['count'] > 1`.

## Key differences from `torchvision`

- Uses `mp.spawn` + `DDP` instead of per-GPU process launch.
- Hardcodes `bf16` autocast (not configurable via `--precision` at runtime).
- Observer is created inside each worker (rank-aware), not injected by voirfile probes.
- voirfile is essentially a no-op passthrough; all instrumentation lives in `main.py`.
- Uses `multigpu_monitor` context manager for GPU telemetry.

## Key dependencies

torch, torchvision, torchcompat, benchmate, voir.
