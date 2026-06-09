# Inference (HF)

Multi-mode HuggingFace inference benchmark covering three distinct modalities:

- **whisper** -- Speech-to-text using `openai/whisper-large-v3` on LibriSpeech
- **flux** -- Text-to-image generation using `black-forest-labs/FLUX.1-dev` (diffusers)
- **chat** -- LLM text generation using `meta-llama/Llama-3.1-8B-Instruct` on GPQA Diamond

## What it measures

Single-GPU inference throughput under realistic workloads. Each mode exercises a
different HuggingFace pipeline and reports samples/sec (whisper), denoising
steps/sec (flux), or tokens/sec (chat).

## Config entries (`config/inference.yaml`)

| Name | Mode | Scheduling |
|------|------|-----------|
| `whisper-transcribe-single` | whisper | per_gpu |
| `txt-to-image-single` | flux | per_gpu |
| `llm-chat-completion` | chat | per_gpu (n=1) |

## Key dependencies

torch, transformers, diffusers, datasets, accelerate, torchcodec, argklass, cantilever

## Notes

- No voirfile -- instrumentation is done inline via `benchmate.observer.BenchObserver`.
- Precision is bfloat16 across all modes.
- Whisper uses a custom pipeline (not HF `pipeline()`) for tighter control.
- Flux measures per-denoising-step time via `callback_on_step_end`, not per-image time.
- Chat mode counts total tokens (input + output) per second using a `TokenizerWrapper`.
