# DINOv2

Self-supervised vision transformer (ViT-Giant/14) pre-training using Meta's DINOv2 framework.
Measures throughput of a self-distillation SSL pipeline with multi-crop augmentation on ImageNet-scale data.

## What it measures

- GPU compute throughput for large vision transformer training (ViT-g/14, ~1.1B params)
- Multi-GPU scaling via FSDP (Fully Sharded Data Parallel) + torchrun
- Memory pressure from dual forward passes (student + teacher EMA network)

## Framework

Wraps [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) (forked at Delaunay/dinov2 for a synchronization fix).

## Config variants

| Name | Scale | Notes |
|------|-------|-------|
| `dinov2-giant-single` | per-GPU | batch 32 |
| `dinov2-giant-gpus` | all GPUs, 1 node | default batch |
| `dinov2-giant-nodes` | multi-node | batch 12, 2 machines |

## Data

Synthetic FakeImageNet (384x384, 3-channel) generated at prepare time via `benchmate.datagen`.
DINOv2 expects an ImageNet-like folder structure; metadata files are hardlinked to fill the expected sample count.

## Key dependencies

- dinov2 source (cloned into `src/`)
- PyTorch, torchvision, FSDP
- benchmate (observer, datagen, monitoring)
- voir (instrumentation probes)
