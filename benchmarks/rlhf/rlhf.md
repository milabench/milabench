## Overview

## Entry points

- `benchfile.py`: `Rlhf` package. Overrides `build_run_plan()` to use `AccelerateAllNodes` launch (not standard `voir` wrapping). Calls `.use_stdout()` so output goes to stdout.
- `prepare.py`: Downloads dataset (`trl-internal-testing/descriptiveness-sentiment-trl-style`) and model (`EleutherAI/pythia-1b-deduped`) via `benchmate.hugginface` helpers.
- `main.py`: Training entry point. Subclasses TRL's experimental `PPOTrainer` for instrumentation.
- `voirfile.py`: Standard `voirfile_monitor` but effectively unused -- `bench_monitor()` context manager in `main.py` handles instrumentation instead.

## Code flow

1. `main()` parses `ScriptArguments`, `PPOConfig`, and `ModelConfig` via HuggingFace `HfArgumentParser`.
2. Loads 4 models from `EleutherAI/pythia-1b-deduped`: policy, ref_policy, reward_model, value_model. All share the same base weights initially.
3. Dataset is pre-tokenized with left-padding. Train/eval split is hard-coded (last 100 samples = eval).
4. `PPOv2TrainerIntrumented` wraps TRL's `PPOTrainer`:
   - Monkey-patches `accelerate.Accelerator` with `torchcompat`'s version for hardware abstraction.
   - Wraps `self.dataloader` with `BenchObserver.iterate()` for throughput measurement.
   - Disables `generate_completions()`, `_save_checkpoint()`, and `save_model()` (no-ops).
   - Disables reporting (`report_to = []`).
5. `trainer.train()` runs the PPO loop. Each step: generate completions from policy, score with reward model, estimate values, compute PPO loss, update policy.
6. `bench_monitor()` context manager handles voir integration. `StopProgram` exception is caught at the top level.

## Batch size measurement

Measured as `input_ids.shape[0] * input_ids.shape[1]` -- total tokens in the batch (batch_size x sequence_length), not just number of sequences.

## Instrumentation path

Unlike most benchmarks, this does NOT use `voirfile.py` for instrumentation. Instead:
- `main.py` wraps execution in `bench_monitor()` context manager
- `BenchObserver` is created inside `PPOv2TrainerIntrumented.__init__()` with `earlystop=70`, `raise_stop_program=True`
- The observer wraps the dataloader iterator, not the training step

## Known gotchas

- 4 full model copies in VRAM (~4x 1B params). Memory-heavy; `rlhf-single` needs ~20+ GB even at small batch sizes.
- `accelerate.Accelerator` is monkey-patched globally at import time inside `__init__`. This could break if other code also patches it.
- `output_dir` is deleted with `shutil.rmtree` at startup -- destructive if pointed at wrong path.
- `generate_completions()` is a no-op, so the benchmark skips the actual text generation evaluation step.
- `_save_checkpoint` and `save_model` are no-ops -- no disk I/O from checkpointing.
- Uses `trl.experimental.ppo` -- this is TRL's experimental PPO, not the stable API. Breaking changes expected.
- `SIMPLE_CHAT_TEMPLATE` is applied as fallback if the tokenizer lacks one.
- `dtype: bfloat16` is passed via config but only affects model loading, not training precision (controlled by accelerate).

## Computational profile

| Variant | Bottleneck | VRAM (H100) | Typical throughput |
|---------|-----------|-------------|-------------------|
| rlhf-single | Autoregressive generation in PPO rollout | ~20 GB @ bs=64 | ~2k-3k tokens/s |
| rlhf-gpus | Cross-GPU communication + generation | ~20 GB/GPU @ bs=64 | ~10k-20k tokens/s |

## Key dependencies

`trl` (experimental PPO), `transformers`, `accelerate`, `datasets`, `torchcompat`
