## Overview

## Config

- **Model**: Llama-4-Scout-17B-16E (Mixture of Experts)
- **Dataset**: `edit_10k_char` (code editing tasks)
- **max_duration**: 3600 seconds -- these runs are slow
- **Parallelism**: Tensor-parallel across available GPUs

## MoE architecture

17B total parameters with 16 experts. Only a subset of experts are activated per
token, but all expert weights must reside in memory.

## Server setup

Same `subprocess.Popen(["vllm", "serve", ...])` pattern as the dense variant.
Startup can take minutes; client uses `--ready-check-timeout-sec 1200`.

## Metrics

Same as dense variant: ttfts, e2els, itl, input_tok, output_tok, request_rate
pushed via `observer.record_metric`.

## Quirks

- Can OOM on smaller GPUs -- Maverick variant was considered but was too large.
- `max_duration: 3600` reflects the slow throughput of MoE inference.
- `--wth-config` in server argv is remapped to `--config` (collision with voir's `--config`).
- Environment vars for flashinfer have trailing spaces in keys (bug in benchfile.py).
- No voirfile; all instrumentation is in `main.py`.
