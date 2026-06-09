## Overview

LLM chat inference using Llama-3.1-8B-Instruct. Measures token-based
throughput on reasoning problems.

## Model and data

- Model: `transformers.pipeline("text-generation")`, `device_map="auto"`
- Dataset: `hendrydong/gpqa_diamond` test split (math/science reasoning)
- batch_size: 1 (variable-length generation)

## Code path

`main.py:main()` parses `Arguments` via argklass, dispatches to
`ChatBenchmark` (mode=chat). Inherits from `InferenceBenchmark`.

Key methods:
- `load_model()`: loads text-generation pipeline with Llama-3.1-8B-Instruct
- `load_dataset()`: loads GPQA Diamond test split
- `run()`: generates completions, counts tokens via `TokenizerWrapper`

## Token counting

- `TokenizerWrapper` counts input tokens
- Output tokens counted from `generated_token_ids`
- `custom_step=True` with manual `dataset.step(tok_tot)` reporting total tokens
- `get_batch_size` returns `len(batch) * 100` (inflated to make rate units ~tok/s)

## Instrumentation

No voirfile. `BenchObserver` with `custom_step=True`. `earlystop=65` batches.

## Gotchas

- batch_size is typically 1 due to variable-length generation
- Async streaming generation code exists but is unused
- `--prepare` flag short-circuits: loads data + model then exits
