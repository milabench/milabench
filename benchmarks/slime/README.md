# slime

Megatron-LM (NVIDIA) training driven by [THUDM/slime](https://github.com/THUDM/slime) on Ray.

| flavour | slime mode | what is measured |
|---|---|---|
| `*-sft`  | `--debug-train-only` + `sft_loss` | Megatron training throughput (`perf/actor_train_tok_per_s`) |
| `*-grpo` | full GRPO loop | the same train metric, with SGLang rollout + weight sync in the loop |

SFT-mode datasets (`dataset:` in the config): `openhermes` (default, chat) and `fineweb` ([HuggingFaceFW/fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb), `sample-10BT`, first 4096 docs). FineWeb is raw text, so each document becomes `[user: "Write a web document.", assistant: <text>]` (truncated to 16k chars) to fit slime's chat-based SFT rollout; the loss covers the whole document.

Entries: `slime-qwen2.5-0.5b-{sft,grpo}-smoke` (1 GPU bring-up) and `slime-glm5-{sft,grpo}` (256 GPU real run).

## How it works
- **Config owns the command line.** `model.argv` (Megatron architecture) and `argv` (everything else) in the yaml are assembled by `benchfile.py` and passed straight to slime's `train.py`, like the vllm pack's `server`/`client` sections. Change parallelism, batch size, steps, datasets... in the config. `{gpu_count}` / `{num_machines}` / `{milabench_data}` placeholders work. `--spec` (two values) is written as one string.
- **Ray through milabench.** Multi-node (`num_machines > 1`, or `ray.cluster: true`) wraps the driver in `RayCluster` (`milabench.commands.ray`): one symmetric srun brings Ray up on every node and `main.py` runs on the main node with `RAY_ADDRESS=auto`. Single node: `main.py` calls `ray.init()` and starts a local cluster. Ray tuning goes in a `ray:` config section (`port`, `start_args`, ...), see `RayCluster`.
- `install` (benchfile) clones slime and Megatron-LM at pinned commits into the code dir (slime's `train.py` and `tools/` are not pip-installed) and applies slime's Megatron patches. `prepare.py` downloads `model.hf`, converts it to `--ref-load` (`model.convert: false` skips it, or set `SLIME_CONVERT=1`), and builds the `dataset:` selected in the config.
- Training steps happen in Ray actors, so `slime_hook.py` (via `--custom-megatron-init-path`) taps `slime.observability.logging_utils.log` in the workers and pushes `rate`/`tflops`/`loss` events to a small collector actor (`slime_metrics.py`). `main.py` drains it and sends the events through `benchmate.monitor.setupvoir()` (stdout smuggling, so the plan uses `use_stdout()` and no `VoirCommand`/voirfile: `voir --config` would need the voir config on every node). The GPU monitor only covers the main node. No files, no shared FS needed.

## Status / TODO
- Nothing has been run yet: install, prepare and both smoke entries still need bring-up.
- `rate` is slime's `actor_train_tok_per_s`, computed from the logging rank's local sequences. Check whether it should be scaled by the DP size.
- GLM-5 needs a pre-converted `torch_dist` checkpoint at `--ref-load` (or `SLIME_CONVERT=1`). The GLM-5 parallelism comes from slime's 744B recipe and is unvalidated.
- Pins (slime `a4202e0`, Megatron `4b4acac`, TE 2.18.0) mirror slime's Dockerfile. TE must build against the installed torch/CUDA.
