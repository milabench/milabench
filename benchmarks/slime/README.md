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

## Entries

| entry | GPUs | what it exercises |
|---|---|---|
| `slime-qwen2.5-0.5b-{sft,fineweb,grpo}-smoke` | 1 | bring-up; GRPO adds the SGLang rollout + weight sync |
| `slime-qwen2.5-0.5b-sft-8gpu` | 8 | data parallel (DP=8), NCCL, `rate` vs DP size |
| `slime-qwen2.5-0.5b-grpo-colocate-8gpu` / `-disagg-4p4` | 8 / 4+4 | colocated trainer+engines vs separate trainer and rollout pools (`--rollout-num-gpus`) |
| `slime-qwen2.5-0.5b-grpo-schedule-smoke` | 1 | response-length cap changing inside one run (`schedule.response_len`) |
| `slime-qwen3-30b-a3b-sft` / `-sft-routing` | 8 | MoE proxy (TP4 x CP2 x EP8); `-routing` forces balanced routing for the first rollouts, then real routing |
| `slime-glm5.3-flash-grpo` | 64 | 8 nodes (PP4 x CP8 x EP16), config only |
| `slime-glm5-{sft,fineweb,grpo}` | 256 | GLM-5.3 744B (TP4 x PP8 x CP8 x EP32), config only |

Every entry's command line is checked against slime's own `parse_args` (pinned versions, faked GPU, dummy checkpoint
dirs): it catches mismatches with the HF config (rotary base, ...) and batch-size arithmetic without any GPU.

## In-run schedules and extra metrics
- `schedule:` in an entry (read by `main.py`, forwarded to the workers as `SLIME_SCHEDULE`):
  `response_len: {rollout_id: cap}` (`slime_rollout.py`, use it with `--rollout-function-path slime_rollout.generate_rollout`)
  and `router: {force_balance_until: N}` (`slime_hook.py`, toggles Megatron's forced load balancing).
- Events sent besides `rate` / `tflops` / `loss`: `step_time_rank` (every rank, with `rank` and `node`: stragglers),
  `expert_imbalance` (busiest / mean expert, per step) with `router_forced`, `time_to_first_step`, slime's other
  `perf/*` metrics, and `response_cap` when the cap changes.

## Install notes (learned the hard way)
- The pack does not use the cluster's CUDA module: nvcc, the `crt`/`cccl` headers and `nvvm` come from the `nvidia-*`
  wheels (same 13.0 release, pinned together), `CUDA_HOME` / `PATH` / `CPATH` / `LD_LIBRARY_PATH` point at them, and
  unversioned `libfoo.so` symlinks are created for the linker. TE's runtime kernels need `NVTE_CUDA_INCLUDE_DIR` to
  match NVRTC.
- sglang is installed `--no-deps` (its pins pull heavy native packages), plus flashinfer and the light imports; slime's
  sglang patches (`docker/patch/<version>/sglang*.patch`) are applied to the installed package: without them
  SGLang rejects the weight updates.
- `--attention-backend auto` (`flash` leaves no backend without flash-attn); fineweb needs selective recompute.

## Status / TODO
- Validated on 1 node of 8 x A100 (all passing): the three smoke entries, `sft-8gpu`, the colocated and the
  disaggregated 8-GPU GRPO entries, the response-cap schedule, and the Qwen3-30B-A3B MoE proxy with and without the
  routing switch (forced balance: expert imbalance ~1.4, real routing: ~10-14). The GLM-5.3 and GLM-5.3-Flash entries
  are config only (they pass slime's argument validation).
- `rate` is slime's `actor_train_tok_per_s`: the tokens of the **whole** rollout batch (all data-parallel ranks,
  `Timer().seq_lens` in `slime/utils/data.py`) over the training time, so it is already the global throughput:
  do not scale it by the DP size. It is logged by DP rank 0 only. The first training step is several times slower
  (warm-up; ~180 s vs ~30 s on the 30B MoE), so drop the first step(s) before comparing runs.
- Multi-node (`RayCluster`) is untested: the venv and the data have to be on a shared filesystem (the slurm scripts
  install under the per-job `/tmp` of one node), and the GPU monitor only covers the main node.
- GLM-5.x: the checkpoint has to be converted on a multi-node job (`SLIME_CONVERT=1`) or provided at `--ref-load`;
  208-296 GPUs need their own parallel layouts (the 256-GPU one is TP4 x PP8 x CP8, DP=1). A truncated variant for
  1-4 nodes needs a truncated HF checkpoint.
- Pins (slime `a4202e0`, Megatron `4b4acac`, TE 2.18.0, sglang 0.5.21, torch 2.13.0) mirror slime's Dockerfile.
- 4-GPU nodes (Rorqual): `slime-qwen2.5-0.5b-{sft,grpo-colocate}-2n4g`, `slime-qwen3-30b-a3b-sft{,-routing}-2n4g` (2 nodes) and
  `slime-glm5.3-flash-sft-frozen` (8 nodes, 32 GPUs: PP4 x DP8, EP8, `--freeze-params-name-list experts`; frozen weights keep
  no gradient or optimizer state). Config only, not validated. Submit with `slurm/schedule_slime_multinode.sh`.
