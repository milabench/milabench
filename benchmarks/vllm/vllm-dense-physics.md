## Overview

## Config

- **Model**: Mistral-Small-3.1-24B (dense transformer)
- **Dataset**: GPQA Diamond (`hendrydong/gpqa_diamond`) -- graduate-level physics QA
- **Tokenizer/config/load format**: `mistral` (requires `--config_format mistral --load_format mistral`)
- **Parallelism**: Tensor-parallel across available GPUs

## Custom dataset class

`GPQADiamond` subclasses `vllm.benchmarks.datasets.HuggingFaceDataset`. Samples
problem/solution pairs, tokenizes to get lengths, wraps in `SampleRequest`. Supports
oversampling if dataset is smaller than `--num-prompts`.

Monkey-patching in `main.py` routes `hendrydong/gpqa_diamond` to this custom class.

## Server setup

Started via `subprocess.Popen(["vllm", "serve", ...])`. Startup can take minutes
for 24B; client uses `--ready-check-timeout-sec 1200`.

## Metrics

Pushed via `observer.record_metric`: ttfts, e2els, itl, input_tok, output_tok,
request_rate (tok/s). 30 observations sampled for timeline reporting.

## Quirks

- `--wth-config` in server argv is remapped to `--config` (collision with voir's `--config`).
- `--config_format mistral --load_format mistral` is mandatory or model fails to load.
- Environment vars for flashinfer have trailing spaces in keys (bug in benchfile.py).
- No voirfile; all instrumentation is in `main.py`.
