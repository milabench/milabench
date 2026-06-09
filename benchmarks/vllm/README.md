# vLLM

Client-server LLM inference benchmark using the vLLM serving engine. Launches a
`vllm serve` process, then hammers it with concurrent requests using vLLM's
built-in benchmarking client (`vllm bench serve`).

## Variants (`config/inference.yaml`)

| Name | Model | Dataset | Architecture |
|------|-------|---------|-------------|
| `vllm-dense-physics-gpus` | Mistral-Small-3.1-24B-Instruct | GPQA Diamond (physics/reasoning) | Dense transformer |
| `vllm-moe-code-gpus` | Llama-4-Scout-17B-16E | edit_10k_char (code editing) | Mixture-of-Experts |

Both use all available GPUs via `--tensor-parallel-size "{gpu_count}"`.

## What it measures

Multi-GPU inference serving throughput: request throughput, time-to-first-token
(TTFT), inter-token latency (ITL), end-to-end latency, and tokens/sec per
request. Reports both aggregate and per-request metrics.

## Key dependencies

vllm (0.18.1), flashinfer-python, flashinfer-cubin, flashinfer-jit-cache, torch,
transformers, datasets

## Notes

- Server and client args are split by `--` delimiter in argv.
- Metric collection monkey-patches `vllm.benchmarks.serve.calculate_metrics`.
- Server is monitored in a background thread; benchmark aborts if server crashes.
- `--request-rate inf` means max-throughput mode (no pacing between requests).
- Dense variant uses Mistral tokenizer mode (`--tokenizer-mode mistral`).
- `config/vllm.yaml` adds concurrency sweep variants for both dense and MoE.
