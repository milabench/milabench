# LLaVA

Vision-language model fine-tuning benchmark using LLaVA 1.5 7B.

## What it measures

Training throughput (samples/s) fine-tuning `llava-hf/llava-1.5-7b-hf` on
the A-OKVQA visual question answering subset of The Cauldron dataset.

## Model

LLaVA 1.5 7B (`LlavaForConditionalGeneration`) loaded in bfloat16 with a
pinned revision. Combines a vision encoder with a LLaMA-based language model
for multimodal conditional generation.

## Variants

| Name | Scope | Plan |
|------|-------|------|
| llava-single | 1 GPU | `per_gpu` |
| llava-gpus | all GPUs | `njobs: 1` (Accelerate multi-GPU) |

## Key details

- Uses HuggingFace `Accelerate` for distributed training.
- Dataset: `HuggingFaceM4/the_cauldron` (aokvqa split).
- Optimizer: AdamW, lr=5e-5, gradient clipping at 1.0.
- Batch size and workers auto-tuned via `auto_batch` / `auto`.
- Early-stops after 70 steps via `BenchObserver`.
- Sets `PT_HPU_LAZY_MODE=0` for Gaudi compatibility.

## Metric

`rate` in samples/s from `BenchObserver`.
