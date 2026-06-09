# Llama (inference)

Single-GPU text generation inference benchmark using LLaMA 2 7B.

## What it measures

Token throughput (Tok/s) running `transformers.pipeline("text-generation")`
on WikiText-103. Measures end-to-end inference including tokenization,
autoregressive decoding, and sampling.

## Model

LLaMA 2 7B Chat (pretrained weights from `meta-llama/Llama-2-7b-chat-hf`).
Uses a public tokenizer (`hf-internal-testing/llama-tokenizer`) with
`model_max_length=256`. Input and output limited to 256 tokens.

## Key details

- Runs in `torch.no_grad()` with `bfloat16` precision.
- Sampling: `do_sample=True, top_k=10`.
- Early-stops after ~40 observations via `benchmate` monitor.
- Config lives in `config/inference.yaml` under `llama:`.
- Tagged `gated` - requires HuggingFace access token for pretrained weights.

## Metric

`rate` in Tok/s = (input_tokens + output_tokens) / elapsed_time, logged
every time the accumulated token count exceeds 30.
