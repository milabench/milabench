## Overview

## Config

- **Model**: Whisper via `AutoModelForAudioClassification.from_config()`, random weights
- **Batch size**: 64
- **Precision**: tf32-fp16

## Why AudioClassification, not speech-to-text

This is intentional. The benchmark targets the Whisper encoder compute path, not
transcription quality. `AutoModelForAudioClassification` exercises the same encoder
forward pass without the decoder overhead.

## Data

Synthetic: `WhisperFeatureExtractor` processes random waveforms. Labels are scalar
class indices (not sequences). Uses a dedicated generator path in `synth.py` for
the `AutoModelForAudioClassification` category.

## Quirks

- The model registry in `bench/models.py` sets `sampling_rate` and `extractor_class`
  on the returned namespace -- these are Whisper-specific fields not used by other models.
- `prepare.py` instantiates the model at prepare time to trigger config validation,
  but does NOT download weights or real audio data.
- fp16 autocast falls back to bf16 on non-CUDA backends.
