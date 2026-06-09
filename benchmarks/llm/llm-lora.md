## Overview

## Config

- **Model**: Llama 3.1 8B (32 layers, 4096 hidden dim, GQA with 8 KV heads, RoPE)
- **Method**: LoRA finetuning (rank=8, alpha=16, targets q_proj/v_proj/output_proj + MLP)
- **Batch size**: 8
- **Gradient accumulation**: 8 steps
- **Precision**: bf16
- **Optimizer**: AdamW (fused=True for single-device, lr=3e-4)
- **Loss**: `CEWithChunkedOutputLoss` (chunked cross-entropy to reduce memory)

## Recipe scripts

- `bench/lora_finetune_single_device.py` -> `LoRAFinetuneRecipeSingleDevice`
- `bench/lora_finetune_distributed.py` -> `LoRAFinetuneRecipeDistributed` (FSDP2)

Launched via torchtune CLI: `torchtune._cli.tune run <recipe> --config <yaml>`.
`benchfile.py` wraps this in a custom `Torchtune` executor that forces torchrun
even for single-device (torchtune expects distributed env vars).

## Instrumentation

`bench/utils.py:prepare_voir(recipe)` wraps the dataloader with `BenchObserver`.
Batch size = `batch["tokens"].shape[0] * shape[1]` (total tokens per batch).
`recipe.log_loss` is monkey-patched for loss recording. Early stop after 30 steps.

## Quirks

- Only adapter params are trainable; base model frozen via `set_trainable_params`.
- HF token (`MILABENCH_HF_TOKEN`) required; benchmark tagged `gated`.
- Single-device variant is compute-bound on forward/backward of full 8B model.
- Gradient accumulation (8 steps) adds latency before each optimizer step.
- Data: Alpaca dataset downloaded and tokenized via torchtune at prepare time.
