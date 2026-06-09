## Overview

## Config

- **Model**: Llama 3.1 70B (80 layers, 8192 hidden dim, GQA with 8 KV heads, RoPE)
- **Method**: Full finetuning (all params trainable)
- **Batch size**: 2
- **Precision**: bf16
- **Optimizer**: AdamW (lr=2e-5)
- **Loss**: `CEWithChunkedOutputLoss` (chunked cross-entropy to reduce memory)
- **LoRA config (if applied)**: rank=16, alpha=32, same targets as 8B

## Recipe script

`bench/full_finetune_distributed.py` -> `FullFinetuneRecipeDistributed` (FSDP2 + optional TP)

FSDP required -- 70B does not fit on a single device. Model sharded per-layer
(TransformerSelfAttentionLayer boundary). `tok_embeddings` and `output` are sharded
separately due to large vocab embedding size.

## Distributed details

- Gradients normalized by total tokens across all ranks (all_reduce before optimizer step).
- All params trainable, FSDP shards everything.
- `TorchtuneAllNodes` adds a `WorkingDir` wrapper to set cwd correctly on remote nodes.

## Quirks

- Memory-bound: FSDP resharding + activation checkpointing overhead dominates.
- Weight download in prepare phase can be slow (~140GB for 70B).
- HF token (`MILABENCH_HF_TOKEN`) required; benchmark tagged `gated`.
- `torchcompat.core.acc.init_process_group()` used instead of raw `init_process_group`.
- llama3 source is cloned for the `Transformer` class (used only in `prepare.py` for
  weight generation); training uses torchtune's model definitions.
- tiktoken requirement is stripped from llama3 to avoid version conflicts.
