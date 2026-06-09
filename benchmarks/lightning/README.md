# Lightning

Image classification training benchmark using [PyTorch Lightning](https://lightning.ai/) as the training framework.

## What it measures

Training throughput (samples/sec) for a torchvision model wrapped in a Lightning `Trainer`. Tests Lightning's overhead and distributed coordination versus raw PyTorch.

## Config entries

| Entry | Scaling | Notes |
|-------|---------|-------|
| `lightning` | 1 process per GPU | Single-node, mono-GPU |
| `lightning-gpus` | 1 job, all GPUs | Single-node, multi-GPU via Lightning DDP |

All inherit `_lightning`. Default model is `resnet152`.

## Dataset

`FakeImageNet` -- synthetic images generated at prepare time. Loaded via `benchmate.dataloader.imagenet_dataloader`.

## Key details

- **Model**: Any `torchvision.models` model (default: `resnet152`)
- **Loss**: `F.cross_entropy`
- **Optimizer**: Adam, lr=1e-3
- **Precision**: bf16-mixed
- **Max steps**: 120 (hard-coded in Trainer)
- **Distributed**: Lightning `strategy="auto"`, `accelerator="auto"`
- **Checkpointing**: Disabled (`enable_checkpointing=False`)
- **Dependencies**: `torch`, `torchvision`, `lightning`, `torchcompat`, `voir`
