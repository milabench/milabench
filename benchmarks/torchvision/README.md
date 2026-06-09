# Torchvision

Single-GPU image classification training using `torchvision.models`. Measures raw
training throughput (samples/sec) for convolutional networks on synthetic ImageNet-shaped
data (224x224x3, 1000 classes).

## Concrete benchmarks

| Name | Model | Batch size | Precision | Notes |
|------|-------|------------|-----------|-------|
| `resnet50` | ResNet-50 | 256 | tf32-fp16 | channels-last, real FakeImageNet I/O |
| `resnet50-noio` | ResNet-50 | 256 | tf32-fp16 | synthetic_fixed loader, pure compute |
| `convnext_large-{fp32,fp16,tf32,tf32-fp16}` | ConvNeXt-Large | 128 | varies | precision comparison suite |
| `regnet_y_128gf` | RegNetY-128GF | 64 | tf32-fp16 | largest single-GPU vision model tested |

## Data

`prepare.py` calls `benchmate.datagen.generate_fakeimagenet()` which writes ~1000 JPEG
images into `$MILABENCH_DATA/FakeImageNet/train/`. The `resnet50-noio` variant bypasses
disk entirely using a fixed synthetic tensor batch.

## Scheduling

All variants run with `plan.method: per_gpu` (one process per GPU, independent).

## Key dependencies

torch, torchvision, torchcompat (device abstraction), benchmate (data + observer),
voir (instrumentation), ptera (probe overrides).
