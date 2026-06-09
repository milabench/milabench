## Overview

Speech-to-text inference using OpenAI's Whisper model. Measures transcription
throughput on audio samples.

## Model and data

- Model: `AutoModelForSpeechSeq2Seq` (openai/whisper-large-v3), bfloat16
- Dataset: `openslr/librispeech_asr` clean split, audio decoded via torchcodec
- batch_size: 64 (via `auto_batch`)

## Code path

`main.py:main()` parses `Arguments` via argklass, dispatches to
`WhisperBenchmark` (mode=whisper). Inherits from `InferenceBenchmark`.

Key methods:
- `load_model()`: loads pretrained whisper model + processor
- `load_dataset()`: streams librispeech_asr clean split
- `transform()`: averages stereo to mono, preserves sample rate metadata
- `run()`: processor -> `model.generate()` under `torch.inference_mode()` + autocast

## Instrumentation

No voirfile. `BenchObserver` instantiated in `InferenceBenchmark.prepare_voir()`,
wraps dataloader and reports batch rates. `earlystop=65` batches.

## Gotchas

- HF pipeline path exists in code but is disabled (`if False`)
- `batch_size_fn` returns `len(batch)` (dead code for seconds-based metric exists)
- `torch.compile` is commented out
- `--prepare` flag short-circuits: loads data + model then exits
