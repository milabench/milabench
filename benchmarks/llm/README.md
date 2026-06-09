# LLM (torchtune)

LLM finetuning benchmark using Meta's torchtune framework on Llama 3.1 models.
Tests LoRA and full-parameter finetuning across single-GPU, multi-GPU (FSDP), and multi-node configurations.

## What it measures

- Tokens/second throughput for LLM finetuning workloads
- Scaling efficiency across GPUs (FSDP2) and nodes
- Memory pressure differences between LoRA (8B) and full finetuning (70B)

## Framework

Built on [torchtune](https://github.com/pytorch/torchtune) with custom recipe scripts forked from upstream.
Uses torchtune's config system (OmegaConf YAML) for model/optimizer/data specification.

## Config variants

| Name | Model | Method | Scale |
|------|-------|--------|-------|
| `llm-lora-single` | Llama-3.1-8B | LoRA | per-GPU |
| `llm-lora-ddp-gpus` | Llama-3.1-8B | LoRA + FSDP | all GPUs |
| `llm-lora-ddp-nodes` | Llama-3.1-8B | LoRA + FSDP | multi-node |
| `llm-lora-mp-gpus` | Llama-3.1-70B | LoRA + FSDP | all GPUs |
| `llm-full-mp-gpus` | Llama-3.1-70B | Full finetune | all GPUs |
| `llm-full-mp-nodes` | Llama-3.1-70B | Full finetune | multi-node |

## Data

- Dataset: `torchtune.datasets.alpaca_cleaned_dataset` (8B) or `alpaca_dataset` (70B)
- Tokenizer: Llama3 SentencePiece (`tokenizer.model`) downloaded from HuggingFace

## Key dependencies

- torchtune (recipes, model definitions, checkpointers)
- PyTorch FSDP2, DTensor
- llama3 model code (cloned from meta-llama/llama3 for weight generation)
- torchcompat (device-agnostic process group init)
- benchmate (observer, monitoring)
- HuggingFace token required for weight download (gated model)
