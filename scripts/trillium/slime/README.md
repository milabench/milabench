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
| 2 | `src/scripts/trillium/slime/submit.sh q30-2n 00:45:00 2 slime-qwen3-30b-a3b-grpo-2n -- --nodelist=trig0146,trig0147` | 2 nodes, Ray across them; `-2n-r3` with rollout routing replay, `-2n-engine16` with one SGLang engine over both nodes |

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
- Rollout routing replay needs slime's `sglang-router` fork (PyPI's 0.3.2 returns no routed experts, job 1261523), as
  slime's Dockerfile installs it: `0.3.2+slime` from Zilin Zhu's `github.com/zhuzilin/sgl-router`, release
  `v0.3.2-9daabcd`, installed `--no-deps` in this workload's venv only (the user's approval, 2026-11-07).
  `install.sh` checks its sha256 before installing it:
  `https://github.com/zhuzilin/sgl-router/releases/download/v0.3.2-9daabcd/sglang_router-0.3.2-cp38-abi3-manylinux_2_28_x86_64.whl`,
  sha256 `c26d31b9decd4bef04c5c2bf72470e7660c2802b9599b885eb7c1dfbfbd9a4b5` (the digest GitHub lists for that asset).
  Since 2026-11-07 21:10 every entry in this venv runs with it (jobs 1262032, 1262053 and before: PyPI's).

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
| no `--use-rollout-routing-replay` (the `-2n-r3` entry has it) | simpler; it needs the router fork (Install) |
| `--sglang-attention-backend triton`, `--sglang-sampling-backend pytorch` | as the validated GRPO entries (no flashinfer JIT) |

2 nodes (`num_machines: 2`, Ray through milabench's `RayCluster`: one `rayrun` per node, the driver on the first):

| entry | trainer | rollout |
|---|---|---|
| `slime-qwen3-30b-a3b-grpo-2n` | 16 GPUs, TP4 x EP8, DP 4 | 2 SGLang engines of 8 GPUs, one per node, behind slime's router on the first node |
| `slime-qwen3-30b-a3b-grpo-2n-r3` | same, with `--use-rollout-routing-replay` | same; SGLang returns each token's experts, the trainer replays them |
| `slime-qwen3-30b-a3b-grpo-2n-engine16` | same | 1 engine of 16 GPUs over both nodes, the GLM-5 entry's engine layout: EP 16, DP attention 16, DP LM head |

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

### Step 2: 2 nodes

All passed, 6 steps each, 29.9-32.8 min per job. Steps 1-5, mean, seconds (1 node from above):

| | 1 node (1261710) | 2n (1262032) | 2n again (1262235) | 2n-r3 (1262234) | 2n-engine16 (1262053) |
|---|---|---|---|---|---|
| nodes | trig0146 | trig0146-0147 | trig0149-0150 | trig0146-0147 | trig0149-0150 |
| sglang-router | PyPI | PyPI | fork | fork | PyPI |
| step | 279.7 | 208.1 | 211.2 | 217.2 | 213.4 |
| rollout | 155.5 | 111.1 | 113.4 | 116.0 | 120.9 |
| training (actor + reference log-probs) | 80.7 (55.6 + 24.8) | 44.2 (30.1 + 13.7) | 47.2 (30.4 + 16.4) | 52.9 (35.5 + 17.0) | 52.6 (31.8 + 20.4) |
| weight sync | 13.3 | 15.2 | 17.9 | 20.9 | 12.3 |
| offload + onload | 18.3 + 5.7 | 20.4 + 7.7 | 15.2 + 9.5 | 14.2 + 5.8 | 14.3 + 6.7 |
| rest of the trainer's wait | 6.2 | 9.6 | 8.0 | 7.5 | 6.6 |
| rollout tokens/s per GPU | 604 | 422 | 414 | 404 | 389 |
| training tokens/s (global) | 14,288 | 26,274 | 26,050 | 22,627 | 25,233 |
| reward | -0.31 | -0.32 | -0.33 | -0.34 | -0.32 |
| startup (`time_to_first_step`) | 471 | 790 | 631 | 572 | 641 |

- 2 nodes against 1: the step 1.32-1.34 times shorter. Training 1.71-1.83 times shorter (actor tokens/s 1.82-1.84
  times higher with 2 times the GPUs), the rollout 1.37-1.40 times: twice the engines, but it lasts as long as the
  longest response (40-57 % at the 4,096 cap, as on 1 node), and the per-GPU rollout rate falls from 604 to about 420
  tokens/s. The sync goes
  to 2 engines: 15-18 s against 13.
- Rollout routing replay (against the repeat, same router): the actor 5.0 s longer (35.5 against 30.4; per-step
  medians 35.3 against 29.7; the two runs without it differ by 0.4 s), training tokens/s 13 % lower; the rollout
  +2.7 s and the sync +3.0 s, within the 2.3-2.7 s by which the two runs without it differ. Step +6.1 s (+2.9 %).
  Host memory on the first node 667-688 GB against 600-615 GB (2n, same nodes).
- The router fork alone (2n against 2n again, other nodes): no difference beyond that spread.
- One 16-GPU engine over both nodes (TP 16, EP 16, DP attention 16, `nnodes 2`): works. Rollout 121 s against 111-113
  with 2 engines of 8; the sync 12.3 s against 15-18 (not analysed; with EP 16 each GPU holds 8 of the 128 experts).
- What 1 node did not exercise, now run: Megatron's NCCL across nodes inside the Ray actors (the expert gradients
  reduce between the nodes at TP4 x EP8; training scales 1.75 times, so NCCL is on InfiniBand, inferred, NCCL's own
  log is off), the router reaching the engine on the other node, the weight sync to 2 engines and to an engine over 2
  nodes.
- Addresses: Ray registers the nodes on ib0 (172.30.x, `rayrun --head-ip` from milabench's system file), slime gives
  the engines Ray's node IP (`host` and `dist_init_addr` on 172.30.x), the router listens on ens27f0 (172.21.x, the
  hostname's address) and reaches the engines on 172.30.x. No problem seen.
- Memory: GPU at most 147 GB of 192 (nvml, first node; 145 GB with the 16-GPU engine), the trainer 41-42 GB used;
  host 580-620 GB per node after step 0 (CPU Adam over 16 GPUs instead of 8).
- `NVRM: knvlinkSendInbandData_IMPL` lines, counted on every node (`job.sbatch`; each job starts at 0): 332 on each
  node of the three runs with 2 engines of 8 (trig0146, 0147, 0149, 0150), 324 on each with the 16-GPU engine, 332 on
  trig0146 alone at 1 node. The same count on every node of a configuration: they follow the workload, not a node.

## Step 3: GLM-5.3 (prepared, not run)

`zai-org/GLM-5.3` is FP8 (e4m3, 128 x 128 blocks with `weight_scale_inv`, `model_type: glm_moe_dsa`): 755.6 GB in
141 shards (755,617,140,416 bytes), 743,377,019,904 parameters without the MTP layer (which our Megatron arguments do
not build; 9.95 B more). slime's converter reads it as it is: its `SafetensorReader` multiplies each 128 x 128 block
by its scale on load (bf16; slime's `fp8_cast_bf16.py` does it in fp32, a last-bit difference). So no BF16 copy:
on disk 755.6 GB FP8 + 1.487 TB torch_dist (bf16, one `.distcp` file per converting rank).

The converter's own layout (pipeline = world size, even split) can never start every stage on a DSA computing layer
(layer number 1, 2, 3 or 7, 11, ..., 75; `get_glm5_spec` refuses the split), so the entries give one in
`model.convert_argv`, and `model.convert_nodes` runs one torchrun per node (`prepare.py`):

| checkpoint | conversion | per GPU |
|---|---|---|
| GLM-5.3, `slime-glm5-*` | 2 nodes, TP8 x PP2 (last stage 40 layers, from layer 39) x EP8 | about 99 GB of bf16 weights |
| 10 layers, `slime-glm5-l10-grpo-2n` | the run's first node, PP2 (6 + 4, stage 2 from layer 7) x EP4 | about 20 GB |

Order (each download after the Kimi K3 one has ended, `ps -u $USER -o pid,args | grep "hf download"`):

```bash
cd $SCRATCH/trillium/new/glm-rl; H=data/slime/models/glm5/hf
# 1. the configuration and index, then the 20 shards that hold layers 0-9, the embeddings, the final norm and the
#    LM head: 106.6 GB
nice venv/torch/bin/hf download zai-org/GLM-5.3 --local-dir $H --include "*.json" "*.jinja"
nice venv/torch/bin/hf download zai-org/GLM-5.3 --local-dir $H --include $(python3 -c '
import json, re; wm = json.load(open("'$H'/model.safetensors.index.json"))["weight_map"]
print(" ".join(sorted({f for k, f in wm.items() if not (m := re.match(r"model\.layers\.(\d+)\.", k)) or int(m[1]) < 10})))')
# 2. the truncated checkpoint: 74.1 GB FP8, 72.19 B parameters (torch_dist 144.4 GB, converted by the run's prepare)
nice venv/torch/bin/python src/scripts/trillium/slime/truncate_hf.py $H data/slime/models/glm5-l10/hf 10
src/scripts/trillium/slime/submit.sh glm5-l10 00:50:00 2 slime-glm5-l10-grpo-2n -- --nodelist=...
# 3. the rest of the checkpoint (resumes: 649 GB more), then its conversion, a 2-node job that stops after prepare
nice venv/torch/bin/hf download zai-org/GLM-5.3 --local-dir $H --include "*.json" "*.jinja" "*.safetensors"
src/scripts/trillium/slime/submit.sh glm5-conv 01:00:00 2 SLIME_CONVERT=1 slime-glm5-grpo -- --nodelist=...
```

`slime-glm5-grpo` follows slime's `run-glm5.3-744B-A40B.sh` GRPO flags (no KL loss, so no reference
forward pass; per-token loss) and takes the SGLang MoE and sampling backends that ran offline with the Qwen stand-in.

## Untested

- More than 2 nodes, the GLM-5.3 entries, disaggregated rollout across nodes.
- SGLang's flashinfer MoE kernels on B200 (cubins offline: `scripts/trillium/kimi/kernels.sh` stages FlashInfer's).
