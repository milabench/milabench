## Overview

**Benchmark:** llama
**Model:** meta-llama/Llama-2-7b-chat-hf
**Task:** text-generation inference
**Metric:** Tok/s

### Config (config/inference.yaml)

```yaml
llama:
  inherits: _defaults
  definition: ../benchmarks/llama
  group: llm
  plan:
    method: per_gpu
  argv:
    --pretrained: true
```

### Architecture

1. Loads WikiText-103 dataset and public LLaMA tokenizer.
2. Instantiates `LlamaForCausalLM` (pretrained, bf16, device_map=cuda).
3. Wraps in `transformers.pipeline("text-generation")`.
4. Iterates dataset entries, generates up to 256 new tokens per sample.
5. Reports cumulative token rate every ~30 tokens.

### Important flags

- `--pretrained`: uses real weights (default in config). Without it, uses random init which never emits EOS - much slower.
- `--model`: selects config variant (`llama2-7b`, `llama2-13b`, `llama2-70b`).
- `--cache`: required, sets `XDG_CACHE_HOME` for HuggingFace downloads.

### Prepare step

Runs `main.py --prepare` which downloads the tokenizer and dataset (and model weights if `--pretrained`).

### Run standalone

```bash
python main.py --cache /tmp/cache --pretrained --model llama2-7b
```
