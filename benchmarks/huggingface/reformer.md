## Overview

## Config

- **Model**: Reformer via `AutoModelForMaskedLM.from_config()`, random weights
- **Batch size**: 32
- **Precision**: tf32-fp16

## num_buckets quirk

Reformer requires `num_buckets` to be set explicitly in the HF config (defaults to
128 if falsy). Without this, the LSH (Locality-Sensitive Hashing) attention mechanism
breaks. This is handled in `bench/models.py` during model registration.

## Data

Synthetic: random `input_ids` + `labels`, same generator path as BERT
(`AutoModelForMaskedLM` category).

## Quirks

- `accelerator.optimize()` is called WITHOUT explicit dtype globally for all
  huggingface models. This was done specifically because passing dtype causes
  Reformer to fail (commented out in the code).
- fp16 autocast falls back to bf16 on non-CUDA backends.
- The epoch loop is removed; training runs as a single infinite epoch with
  early stopping via `BenchObserver`.
