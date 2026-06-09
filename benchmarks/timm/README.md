# timm

Image classification training benchmark wrapping [pytorch-image-models](https://github.com/huggingface/pytorch-image-models) (timm).

## What it measures

Single-GPU throughput (samples/sec) for training vision models on synthetic ImageNet-shaped data. The workload is a full training step: forward pass, loss, backward, optimizer step.

## Framework

Uses timm's own `train.py` script, cloned from a pinned commit. Milabench does **not** ship a custom training loop; it instruments timm's loop via `voir` probes.

## Config entries

| Entry | Model | Scaling |
|-------|-------|---------|
| `focalnet` | `focalnet_base_lrf` | 1 process per GPU |

All entries inherit `_timm`, which sets AMP (bfloat16), FakeImageNet data, and `auto_batch(128)`.

## Dataset

`FakeImageNet` -- synthetic images generated at prepare time by `benchmate.datagen.generate_fakeimagenet`. No download required.

## Key details

- **Precision**: AMP with bfloat16
- **Optimizer**: Determined by timm defaults (typically SGD with momentum)
- **Checkpointing**: Disabled via voirfile override (`saver` probe returns `None`)
- **Early stop**: After 20 rate measurements (configurable via `voir.options.stop`)
- **Dependencies**: `timm`, `torch`, `torchvision`, `voir`, `safetensors`
