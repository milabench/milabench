# slime GRPO on Trillium (B200)

GRPO post-training with slime (`benchmarks/slime`, Megatron-LM trainer, SGLang rollouts, Ray) for the GLM-RL
workload (`ops/GLM_RL_BRIEF.md`). Bring-up on a Qwen3-30B-A3B stand-in first, then the GLM-5.3 entries.

Everything lives in `B = $SCRATCH/trillium/new/glm-rl`: `src` (this repository, branch `trillium-new`, with the
milabench CLI venv in `src/.venv`), `venv/torch` (the pack's venv), `data`, `runs`, `slurm` (job logs), `uv-cache`,
`wheelhouse`. Not the campaign's `$SCRATCH/trillium/glm`: the slime entries install into `venv/torch` too.

| step | command (login node) | what |
|------|------------------|------|
| install | `src/scripts/trillium/slime/install.sh` | downloads (below) |
| 1 | `src/scripts/trillium/slime/submit.sh q30-1n 00:55:00 1 slime-qwen3-30b-a3b-grpo -- --nodelist=trig0146` | 1 node: GPU build of TE and torch_memory_saver (first job only), the HF -> torch_dist conversion (once), 6 GRPO steps |

`submit.sh NAME TIME NODES ENTRY... [-- sbatch args]` submits one job named `nw-glm-rl-NAME` that runs `job.sbatch`:
power check, `milabench install` (offline), `milabench prepare` on the first node, then one `milabench run` per entry.
The job runs `$B/src` as it is: `git -C $B/src pull --ff-only` first.

## Install

Compute nodes have no internet. `install.sh` (login node) creates the CLI venv, runs the pack's install with
`SLIME_DOWNLOAD_ONLY=1` (everything but TransformerEngine and torch_memory_saver, which need a GPU node), puts TE
2.18.0's three files from PyPI in `wheelhouse` and installs TE's own dependencies, fetches torch_memory_saver's pinned
git commit into the uv cache (a dry-run install; offline, uv takes a pinned commit from its cache) and downloads the
entry's HF checkpoint and dataset (`milabench prepare` with `SLIME_DOWNLOAD_ONLY=1`). The first job then builds TE
(`NVTE_PYTORCH_FORCE_BUILD`, `NVTE_CUDA_ARCHS=100`) and torch_memory_saver offline (`UV_OFFLINE`,
`UV_FIND_LINKS=wheelhouse`), as the campaign's GLM install did (`scripts/trillium/glm/README.md`, Install).

- Own uv cache, `UV_LINK_MODE=copy`: the pack's `patch_sglang` rewrites sglang's installed files, and with uv's default
  hardlinks it rewrites the cache's files too. The campaign's `venv/torch` shares its inodes with `$SCRATCH/uv-cache`
  (link count 2, 2026-11-07), so the sglang 0.5.21 in that cache is already patched: a venv installed from it
  would refuse `sglang.patch` (it does not apply twice).
- torch_memory_saver is required here (`Slime.memory_saver_required`): `--colocate` turns `--offload-train` on, which
  pauses the trainer's GPU memory with it while SGLang generates. The campaign's GLM venv went without it. Its link
  needs `-lcuda`, which cvmfs's linker does not find in `/usr/lib64`: `job.sbatch` puts `/usr/lib64/nvidia` (the
  driver's libraries only) on `LIBRARY_PATH` (job 1261365).
- `$HOME` is read-only from the compute nodes: besides `XDG_CACHE_HOME`, sglang needs `SGLANG_CACHE_DIR` (it points
  torch inductor's cache there, job 1261398) and `SGLANG_JIT_CACHE_DIR` (its RoPE kernel build, job 1261434),
  both under `~/.cache/sglang` by default. `job.sbatch` puts them in `/dev/shm`.
- With the 1.56 TB Kimi K3 download running beside it (2026-11-07), uv's 30 s HTTP timeout failed on torch's
  `nvidia-cublas`: `install.sh` sets 300 s. The login-node install then took about 40 min, Qwen3-30B-A3B (61 GB) 14 min.

## Entries

`slime-qwen3-30b-a3b-grpo` (`benchmarks/slime/dev.yaml`): slime's `scripts/run-qwen3-30B-A3B.sh` on one node,
colocated: the trainer (TP4 x EP8, DP 2, CPU-offloaded Adam) and one SGLang engine of 8 GPUs take turns on the
GPUs. DAPO-Math-17k, 32 prompts x 8 samples = 256 samples per step, one training step per rollout, responses capped
at 4,096 tokens (the recipe: 131,072), 6 steps, no eval. GRPO flags of the GLM-5 entry (score centering, REINFORCE
loss, TIS) except rollout routing replay. Differences from the recipe, each from a failed job:

| what | why |
|---|---|
| `--rm-type dapo --reward-key score` (+1 / -1 on the prompt's `Answer:` line) | the recipe's `deepscaler` scores 0 any response without `</think>`: every response of Instruct-2507 (job 1261612, 512 samples) |
| `--sglang-moe-runner-backend triton` | on B200 SGLang picks `flashinfer_trtllm`, which downloads its cubins at the first forward (job 1261477) |
| no `--use-rollout-routing-replay` | PyPI's `sglang-router` 0.3.2 returns no routed experts (job 1261523). slime's Dockerfile installs a fork for it (`zhuzilin/sgl-router` v0.3.2-9daabcd, `0.3.2+slime`); not in this install |
| `--sglang-attention-backend triton`, `--sglang-sampling-backend pytorch` | as the validated GRPO entries (no flashinfer JIT) |

## Measured

Step 1, `slime-qwen3-30b-a3b-grpo` on 1 node (`trig0146`, job 1261710, 33 min, run 1,952 s, 6 steps): slime's
`perf/*` and `rollout/*` lines in the job log. Steps 1-5 (step 0 is warm-up), mean (range), seconds:

| | per step | share |
|---|---|---|
| step (`perf/step_time` = trainer wait + train) | 279.7 (269.1-295.9) | |
| rollout generation (`perf/rollout_time`) | 155.5 (143.1-166.8) | 56 % |
| training (`perf/train_time`) | 80.7 (71.6-89.1): actor 55.6, reference log-probs 24.8 | 29 % |
| weight sync to SGLang (`perf/update_weights_time`) | 13.3 (11.3-20.5) | 5 % |
| trainer offload + onload (`perf/sleep_time` + `wake_up_time`) | 18.3 + 5.7 | 9 % |
| rest of the trainer's wait | 6.2 | 2 % |

- Rollout: 604 tokens/s per GPU (`perf/tokens_per_gpu_per_sec`, 4.8k/s on the node; 256 x 2,913 tokens / 155.6 s
  in step 1), 42-54 % of the responses at the 4,096-token cap: the rollout lasts as long as the longest sample.
  Mean response 2,708-3,111 tokens. Training: 14,288 tokens/s (`perf/actor_train_tok_per_s`, already global; prompt +
  response tokens over the actor time: 785k / 58.8 s in step 1).
- Reward (`rollout/raw_reward`, +1 / -1): -0.27, -0.38, -0.28, -0.26, -0.35 in steps 1-5 (-0.14 in step 0; 32 new
  prompts per step, lr 1e-6): computed and non-zero, no trend in 6 steps, none expected. About half the responses stop
  at the 4,096-token cap without an `Answer:` line and score -1, so the reward mostly tracks truncation.
- Startup (`time_to_first_step`, launch to the first training step): 470.7 s: 3.4 min to the engines up and the first
  weight sync, then the first rollout (141 s) and training step (98 s).
- Memory: GPU at most 147 of 192 GB used (nvml's 140,613 of 183,359 MiB, during rollouts: SGLang's static 0.7); the
  trainer 42.7 GB used, 25.1 GB allocated at the step's end (slime's snapshots: not the in-step peak). Host (CPU
  Adam): 850-875 GB used in steps 1-5 (slime's snapshots; 568 GB before the first training step), at most 968 GB
  (benchmate's `cpudata`), of 2,164 GB.
- Conversion HF -> torch_dist on 8 GPUs: 3.5 min (job 1261398). Disk: 57 GB HF + 57 GB torch_dist, venv 8.8 GB, uv
  cache 9.4 GB.

Against the estimate before the run: training (30-90 s) and the sync (a few to 20 s) as expected; the rollout took
2.5 times the 60 s estimated: decode runs at about 27 tokens/s per sequence with ~125 in flight (one TP8 engine,
SGLang's Triton attention and MoE).

Bring-up jobs on `trig0146`: 1261365 (install, torch_memory_saver link), 1261382 (install, 90 s), 1261398
(conversion; `~/.cache/sglang`), 1261434 (JIT cache), 1261477 (flashinfer cubins), 1261523 (routed experts), 1261612
(reward 0, cancelled after 2 steps), 1261710 (passed). `NVRM: knvlinkSendInbandData_IMPL: Failed to send inband data`
lines in the kernel messages of every job that ran on the GPUs (6 to 332, more in longer jobs); the K3 training jobs on `trig0145` / `trig0146` have
them too, the campaign's GLM logs that list kernel messages do not. No effect seen.

## Untested

- More than one node (Ray across nodes through `RayCluster`), the GLM-5.3 entries.
- Rollout routing replay (the router fork), SGLang's flashinfer MoE kernels on B200 (`flashinfer-cubin`, offline).
