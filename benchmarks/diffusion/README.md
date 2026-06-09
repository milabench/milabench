# Diffusion

Stable Diffusion v2 fine-tuning benchmark using HuggingFace `diffusers` and `accelerate`.

## What it measures

Training throughput for diffusion model fine-tuning: UNet noise-prediction on a text-to-image pipeline. Exercises VAE encoding, CLIP text encoding, noise scheduling, and UNet training in bf16 mixed precision.

## Config entries

| Entry | Scaling | Notes |
|-------|---------|-------|
| `diffusion-single` | 1 process per GPU | Mono-GPU |
| `diffusion-gpus` | 1 job, all GPUs | Multi-GPU via Accelerate |
| `diffusion-nodes` | 1 job, 2 nodes | Multi-node via Accelerate |

All inherit `_diffusion`.

## Model components

- **UNet2DConditionModel** (trainable): Predicts noise residual
- **AutoencoderKL** (frozen, bf16): Encodes images to latent space
- **CLIPTextModel** (frozen, bf16): Encodes text captions
- **DDPMScheduler**: Adds noise at random timesteps

Pre-trained weights from `Milabench/stable-diffusion-2` (HuggingFace Hub).

## Dataset

`lambdalabs/naruto-blip-captions` -- small real dataset (~1k images). A `RandomSampler` with replacement upsamples to `batch_size * 70 * WORLD_SIZE` samples per epoch to ensure enough steps.

## Key details

- **Loss**: MSE between predicted noise and actual noise
- **Optimizer**: AdamW (lr=1e-4, scaled by batch/accumulation/num_processes)
- **Precision**: bf16 mixed via `accelerate.Accelerator`
- **Resolution**: 512x512
- **Early stop**: After 60 observed iterations
- **Dependencies**: `diffusers`, `accelerate`, `transformers`, `datasets`, `torch`, `argklass`
