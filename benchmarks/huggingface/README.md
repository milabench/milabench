# HuggingFace

Single-GPU training throughput for HuggingFace Transformers models. All data is
synthetic (random tensors matching each model's expected input shape). Measures
pure compute throughput with no I/O bottleneck.

## Concrete benchmarks

| Name | Model class | HF category | Batch size | Precision |
|------|-------------|-------------|------------|-----------|
| `bert-{fp32,fp16,tf32,tf32-fp16}` | `Bert` (base, 110M) | AutoModelForMaskedLM | 32 | varies |
| `t5` | `T5` (t5-small, 60M) | AutoModelForSeq2SeqLM | 16 | tf32-fp16 |
| `reformer` | `Reformer` (default config) | AutoModelForMaskedLM | 32 | tf32-fp16 |
| `whisper` | `Whisper` (whisper-tiny) | AutoModelForAudioClassification | 64 | tf32-fp16 |

Note: `focalnet` inherits from `_timm`, not `_hf`. It is a different benchmark group.

## Data

No preparation step. `SyntheticData` generates random `input_ids`/`labels` tensors
(or `input_features` for Whisper) matching each model's vocabulary and sequence length.
Data is created once in RAM and repeated via `__getitem__(i % n)`.

## Scheduling

All variants run with `plan.method: per_gpu`. Tagged `noio` and `monogpu`.

## Key dependencies

torch, transformers, torchcompat, benchmate, voir, torchaudio (for Whisper feature extraction).
