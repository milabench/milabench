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
The job runs `$B/src` as it is: `git -C $B/src pull --ff-only` first. At its end, and 5 min before its time limit
(SIGUSR1), it stops Ray on every node and waits until no process holds a GPU before Slurm cleans up (`GPUs free
after N s` per node): `trig0146-0147` drained with `health.chk.nvidia-smi.unresponsive` 1 s after job 1262234 ended,
3 s after its `ray stop` (1 drain in 4 two-node teardowns; cleared at once). Tested on 2 nodes: a normal end (job
1262531: `GPUs free after 0 s` on both, 6 s from the end of Ray to the end of the job) and SIGUSR1 mid-run (job
1262532, `--time 00:15:00`: stopped at 10:00, `ray stop --force` took 10 s, GPUs free after 0 s, ended 12 s after the
signal, exit 124); no drain after either (`sinfo -n`). 3 drain-free teardowns are not proof against a 1-in-4 event.

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
- Rollout routing replay, 3 runs with it (jobs 1262234, 1262690, 1262691) against 3 without (1262032, 1262235,
  1262531), on the same 3 node pairs (trig0146-0147, trig0149-0150, trig0145-0146). Means of steps 1-5, range over
  the runs (mean):

  | | without | with | with - without |
  |---|---|---|---|
  | step | 207.3-211.2 (208.9) | 210.4-217.2 (212.9) | +4.0 s, +1.9 % |
  | rollout | 111.1-113.4 (111.9) | 112.3-116.0 (113.7) | +1.8 s |
  | training (actor) | 44.2-51.6 (47.6) (actor 30.1-35.5) | 51.3-52.9 (52.1) (actor 32.8-35.5) | +4.5 s (actor +2.0) |
  | sync | 15.2-17.9 (16.9) | 17.0-20.9 (19.3) | +2.3 s |
  | offload + onload | 20.5-28.1 (24.4) | 18.6-21.5 (20.0) | -4.4 s |
  | training tokens/s | 22,870-26,274 (25,065) | 22,627-24,437 (23,568) | -6.0 % |
  | host memory, first node | 615-626 GB | 688-696 GB | +72 GB |

  Measured (the ranges do not overlap): host memory only, +72 GB. All the times overlap: routing replay's time cost,
  if any, is within the run-to-run spread at 3 runs each (the first run with it, 217.2 s per step, was the slow one;
  an earlier version of this section, from fewer runs, gave the actor +5 s and training tokens/s -13 %).
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

Downloaded on `tri-dm1`, 2026-11-07 23:33-23:36: the 20 shards in 33 s, all 141 in 168 s (755,663,733,656 bytes with
the configuration; about 4.5 GB/s), each shard's size and header checked against the Hub. Truncated checkpoint
(`truncate_hf.py`, on `tri-dm1`): 74,146,421,681 bytes, 3 min 52 s; indexer weights on layers 1, 2, 3, 7.

Order (transfers above 10 GB on the data-transfer nodes `tri-dm1`-`4`, one at a time):

```bash
cd $SCRATCH/trillium/new/glm-rl; H=data/slime/models/glm5/hf
# 1. download_glm5.sh on a data-transfer node: the 20 shards that hold layers 0-9, the embeddings, the final norm and
#    the LM head (106.6 GB) first, then the rest (649 GB more); one `hf download` (one pattern per --include)
ssh tri-dm1 'nohup $SCRATCH/trillium/new/glm-rl/src/scripts/trillium/slime/download_glm5.sh > $SCRATCH/trillium/new/glm-rl/download-glm5.log 2>&1 &'
# 2. once the 20 shards are there: the truncated checkpoint (74.1 GB FP8, 72.19 B parameters, torch_dist 144.4 GB
#    converted by the run's prepare), then its test
nice venv/torch/bin/python src/scripts/trillium/slime/truncate_hf.py $H data/slime/models/glm5-l10/hf 10
src/scripts/trillium/slime/submit.sh glm5-l10 00:50:00 2 slime-glm5-l10-grpo-2n -- --nodelist=...
# 3. once the download is done: the conversion, a 2-node job that stops after prepare
src/scripts/trillium/slime/submit.sh glm5-conv 01:00:00 2 SLIME_CONVERT=1 slime-glm5-grpo -- --nodelist=...
```

First jobs (2026-11-08, 2 nodes each; nothing trained yet):

| job | what | result |
|---|---|---|
| 1263400 | 10-layer test, its conversion | the GLM-5 plugin does not import without `tilelang` (spec `None`): added, `tilelang==0.1.12` |
| 1263439 | 10-layer test | conversion done (FP8 read, PP2 6 + 4 x EP4, 18.7-21.3 GB allocated per GPU, 2 min); SGLang fell back to its Transformers backend (`GlmMoeDsaForCausalLM` is registered by `glm4_moe.py`, which imports `deep_gemm`): added `sgl-deep-gemm==0.2.0`; the run then waited on the dead engine, stopped with `scancel -b -s USR1` |
| 1263451 | full conversion | launched on both nodes; `--expert-tensor-parallel-size` defaults to TP (8 x 8 x 2 = 128 > 16 GPUs): set to 1 |
| 1263932 | full conversion | done in 6 min 31 s (job), saved 5 min after the start: 1,486,940,657,885 bytes, 16 `.distcp` + 3 files; 93.5 GB allocated per GPU |
| 1263931 | 10-layer test | SGLang sized `q_b_proj` for 384-wide heads: transformers 5.9 maps `head_dim` (192) onto `qk_rope_head_dim` (64): `transformers==5.12.1` (sglang's pin; it also checks the per-layer lists, which `truncate_hf.py` now cuts) |
| 1263962 | 10-layer test | DSA attention picked trtllm (flashinfer cubins, downloaded): FlashMLA as slime's script (`dsa`, `flashmla_sparse` / `flashmla_kv`, page 64, FP8 KV) |
| 1263975 | 10-layer test | prefills up to 2048 tokens take dense MHA on trtllm: `SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD=0` |
| 1264056 | 10-layer test | weight sync: slime packs the indexer `wk` scales as UE8M0 on B200, SGLang dequantizes FP32 ones: `SGLANG_ENABLE_JIT_DEEPGEMM=0` |
| 1264112 | 10-layer test | without DeepGEMM the FP8 GEMMs pick flashinfer_trtllm (cubins): `--sglang-fp8-gemm-runner-backend triton` |
| 1264147 | 10-layer test | weight sync: `Invalid device_uuid` on the second node's engine ranks, in the expert update (slime's rank-local expert routing into the 16-GPU engine) |
| 1264179 | the same, trainer PP1 | the same error: not PP2. The Qwen 16-GPU engine (same nodes, same expert routing) synced |
| 1264182 | PP2, one 8-GPU engine per node | weight sync 9.8 s, the first rollout (64 x 4,096 tokens, 72 s, 227 tokens/s per GPU), then the trainer's RoPE needs apex: the GLM-5 plugin imports it at every TP / CP unless `MEGATRON_USE_SGLANG_ROPE` is set, which requires TP1 and CP1: TP1 / CP1 with SGLang's RoPE, as the campaign |

The flashinfer cubins (`flashinfer-cubin` 0.6.18, FlashInfer's GitHub release, 1.57 GB) and apex (slime's Dockerfile
builds it from NVIDIA's git) would remove 3 of these workarounds; installing them was refused by the permission
check (external code) and needs the user's decision. Without them the engine runs Triton FP8 GEMMs and Triton MoE,
and the trainer TP1 / CP1 only. What the GLM-5.3 entries run instead of B200's defaults:

| path | B200 default (SGLang 0.5.21) | here | why |
|---|---|---|---|
| DSA attention, prefill and decode | `trtllm` (flashinfer cubins) | FlashMLA (`flashmla_sparse` / `flashmla_kv`, slime's GLM-5.3 setting) | cubins not on the nodes |
| prefills up to 2,048 tokens | dense MHA on trtllm-gen FMHA | the sparse FlashMLA path (`SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD=0`) | cubins |
| FP8 GEMMs | DeepGEMM (UE8M0 scales) | Triton (`SGLANG_ENABLE_JIT_DEEPGEMM=0`, `--sglang-fp8-gemm-runner-backend triton`) | slime's UE8M0 scales for the indexer `wk`; flashinfer_trtllm needs cubins |
| MoE | `flashinfer_trtllm` | Triton | cubins |

The Qwen3-30B-A3B entries share the venv: since 2026-11-08 they run with transformers 5.12.1 (5.9.0 before), tilelang
0.1.12, sgl-deep-gemm 0.2.0 and apache-tvm-ffi 0.1.11; a Qwen run after that date is not the same install as the
step 1 / step 2 runs.

`tilelang` and `sgl-deep-gemm` are at sglang 0.5.21's pins (with `apache-tvm-ffi` 0.1.11); they went into the venv
by hand (`uv pip install`, its `.slime_installed` marker stops the pack's install). With them, `GlmMoeDsaForCausalLM`
is registered and SGLang's DSA backend imports (login node). DeepGEMM's JIT cache goes to `/dev/shm`
(`DG_JIT_CACHE_DIR`, `~/.dj` by default).

`Invalid device_uuid` (jobs 1264147, 1264179), from the code: in the colocated update every trainer rank serializes
CUDA IPC handles of its own tensors and `gather_object`s them to the engine's first trainer rank, which passes the list
to the engine; engine rank i opens entry i. A rank with fewer buckets than the others in a batch is padded by the gather
source with an empty tensor on the source's own GPU (`_send_to_colocated_engine`), which an engine rank on another node
cannot open. Expert batches are capped per rank by `--update-weight-buffer-size` (512 MiB): a Qwen3-30B-A3B expert is
9.4 MB, so each layer is one batch reaching every rank; a GLM-5.3 expert is 75.5 MB, so a layer splits into several
batches (61 for 7 layers) and some ranks get none. With engines of 8 GPUs, source and targets share the node (job
1264182 synced). Fix to test: a 4 GiB buffer, one batch per layer (up to 3.6 GB per rank and layer in the 10-layer
test, about 0.9 GB at trainer EP32 with 64-GPU engines); the padding itself stays slime's (a local empty tensor per
rank would remove it: a patch to slime).

DeepGEMM: no slime or SGLang setting at these pins sends FP32 scales to SGLang with DeepGEMM on (slime's quantizer
packs UE8M0 whenever SGLang's DeepGEMM requant check is true; its `transform_ue8m0` is not an argument; SGLang always
fuses the indexer `wk` by dequantizing FP32 scales on CUDA). slime's GLM-5.3 script is for H100 (no packing). Keeping
DeepGEMM would take a one-line slime change: no UE8M0 packing for `self_attention.wk.weight`.

Cubins: every replaced trtllm path downloads them (Qwen MoE: batched_gemm; DSA prefill / decode and the dense-MHA
prefill: fmha/trtllm-gen; FP8 GEMMs: gemm). sglang 0.5.21 requires `flashinfer_python[cu13]==0.6.18` exactly. The
site's `flashinfer_cubin-0.7.0.post1+computecanada` holds 0.6.18's `fmha/trtllm-gen` artifact (same path, 70,654 files),
not its `batched_gemm` or `gemm` ones.

Step 4 at TP1 / CP1 (the campaign's GLM-5 layouts), colocated on 32 nodes, PP8 x EP32 (DP32 for the dense part; slime's
split 14 / 8 x 6 / 16), 128 prompts x 8 = 1,024 samples per step. Estimates, from the 10-layer test (job 1264931):

| per GPU (of 178.3 GiB) | |
|---|---|
| training | 85-115 GiB: BF16 weights 16.9 + FP32 main gradients 33.8 = 50.7 GiB static (9.09 B parameters on the largest stage; Adam and the master weights on the CPU), the rest activations, recompute and the allocator (the 10-layer trainer: 56.7 GiB used for 6.4 B, 8.9 GiB per billion, plus PP8's 6 more micro-batches in flight). 1F1B keeps at most 8 in flight, whatever the batch |
| rollout | about 145 GiB: SGLang's static 0.75 (`--sglang-mem-fraction-static`; 132.4 GiB of the 176.5 free) + graphs and activations (unset, SGLang computes 0.645 at the 10-layer test's TP16: 146.8 GiB peak); the KV of 4 sequences x 8.3k tokens is about 2 GB |
| sync | about 155-160 GiB: the engine + the trainer's BF16 weights (16.9) + the 4 GiB staging |

Host per node (psutil's `used`: process memory; page cache and `/dev/shm`, Ray's object store included, are not in it):
the 10-layer test's 987 GB on its first node is CPU Adam (12 B x its stage's 32.2 B = 386 GB), the trainer's weights
saved during rollouts (104 GB BF16, +208 GB if its FP32 gradients are saved too), SGLang's weights saved during training
(70 GB) and about 220-430 GB of the rest (pinned buffers, processes). At full size, on the last stage's nodes (39.7 B
parameters each: 16 MoE layers + LM head over 4 nodes): Adam 476 GB + trainer 145-435 GB + SGLang 224 GB + 220-430 GB =
about 1.3-1.35 TB of 2.0 TB. The rollout buffers of 1,024 samples are under 1 GB (+21 GB of routed experts and about
70 GB more, as Qwen, with routing replay). Above 1.8 TB, PP16 would halve the optimizer and trainer parts per node.

Per step: rollout 20-45 min (about 0.15-0.3 s per decode step at 4 sequences per DP-attention rank, the 10-layer test's
1.8-3.8 ms per layer x 78, to 8,192 tokens for the longest response); training 2.5-4.5 min (3-6 M tokens at 25-30k
tokens/s: the 10-layer test's 17.6k tokens/s on 16 GPUs x 256 / 16 GPUs x 10 / 78 layers = 36k, less an 18-30 %
bubble); sync 1.5-2 min (10 times the 10-layer test's 8-10 s); trainer and engine offload about 0.5 min. Startup to the
first sync: 10-20 min (8.5 min in the 10-layer test). `--time` = 20 + 3 x (45 + 4.5 + 2 + 0.5) + 3 (warm-up training)
+ 15 margin = 194 min: 3:15:00. The rollout measured since: 15.0-15.8 min (GLM-RL-02, below); GLM-RL-03's `--time`
is 2:05:00.

`slime-glm5-grpo` follows slime's `run-glm5.3-744B-A40B.sh` GRPO flags (no KL loss, so no reference forward pass;
per-token loss), its reward and `--sglang-moe-dense-tp-size 1`, without rollout routing replay (`slime-glm5-grpo-r3` has
it: the user's order is without first), at TP1 / CP1, PP8 x EP32. Reward: slime's `deepscaler`, which scores the answer
after `</think>`; GLM-5.3 reasons in `<think>` by default, so it fits this model, unlike Instruct-2507. Deviations from
the script:

| | slime's script | here | why |
|---|---|---|---|
| response cap | 262,144 | 8,192 | at CP1 a sample stays on one GPU (slime gives an oversized one a micro-batch of its own); the memory estimate assumes 8,192-token micro-batches |
| reasoning effort | the template's Max (slime passes no kwargs) | Low (`--apply-chat-template-kwargs '{"reasoning_effort": "low"}'`; the prompt still ends `<\|assistant\|><think>`) | with Max and an 8k cap most responses would be cut before `</think>`: reward 0, advantages 0 |
| batch | 8 prompts x 8 = 64 (its DP1) | 128 x 8 = 1,024 | at DP32: 32 samples per rank, 16-32 micro-batches through PP8 (bubble 18-30 %); 4 sequences per DP-attention rank of the 4 engines (one leaves decoding latency-bound) |

The first step-4 run's `rollout/truncated` and reward say whether Low and 8k are enough; the reward is there to show
the loop learns, not measured.

### GLM-RL-01 (job 1265179): blocked at engine load

At TP64 every SGLang scheduler raised `ValueError: Weight output_partition_size = 32 is not divisible by weight
quantization block_n = 128`: the shared experts' FP8 `gate_up_proj` split over the engine's TP (moe_intermediate_size
2048 / 64 = 32; `--sglang-moe-dense-tp-size 1` covers only the dense layers 0-2). The 2-node test's 16-GPU engine split
it to 128 and passed. slime's GLM-5.3 script gets TP1 shared experts from DeepEP (`--sglang-moe-a2a-backend deepep`,
sglang `deepseek_v2.py:753-775`), which is not installed here; the same code keeps them unsplit with
`SGLANG_SHARED_EXPERT_TP1=1` (`environ.py:737`), and SGLang then adds their output on TP rank 0 only, before the
reduce-scatter (`moe/utils.py:786-797`). Reproduced at TP32 on 4 nodes (`slime-glm5-l10-grpo-4n`, job 1265300: 2048 / 32
= 64, the same error); with the variable (job 1265370, PASS, 3 steps in 24 min) the 32-GPU engine loaded, the weight
sync took 9.3 s then 6.4-6.5 s, training 170 s then 40 s, rollouts 165-180 s; SGLang's log-probs and the trainer's agree
as at TP16 without it (job 1264931: rollout log-prob -10.26 both, TIS mean 1.0000, |TIS - 1| 0.059, clipfrac 1.1e-4
both), so the shared output is counted once. NVRM 0 on trig0115-0118.

Its cost against slime's DeepEP path: there the shared experts are unsplit too, but each GPU runs them on its own
tokens; here all 64 GPUs of an engine run them on the gathered batch. Against split shared experts (which do not load
at TP64): +2.8 GB per GPU (75 layers x 37.7 M FP8 parameters) and about +10 % weight reads per decode step (2.8 of
about 33 GB), and the fused finalize / all-reduce paths are off (`deepseek_v2.py:1055-1075`).

The same job's engines would not have had memory for a KV pool at TP64: with no `--sglang-mem-fraction-static`
SGLang reserves 512 + 1.5 x 16384 + 1024 x TP / 8 + 2 x bs + 4.5 x bs x DP MiB of 183,359 (decode graphs of bs 512,
`arg_groups/memory_hook.py:300-306, 366-371`): 0.645 at TP16, 0.433 at TP32, 0.009 at TP64 (jobs 1264931, 1265370,
1265179). The KV pool is what is free after the weights less (1 - fraction) x the 176.5 GiB free before
(`kv_cache_configurator.py:2194-2208`): 143.5 - 174.9 < 0 at TP64, "Not enough memory". `slime-glm5-grpo` now sets
0.75 and `--sglang-cuda-graph-max-bs 32` (4 sequences per DP-attention rank; 8 disaggregated), from:

| per GPU at TP64 | GiB |
|---|---|
| weights | about 33: 21.7 replicated (embeddings, attention, dense MLPs, shared experts) + 11.3 routed (the 10-layer engine: 8.70 at TP32) |
| static, 0.75 x 176.5 | 132.4: weights + about 99 of KV, against 4 x 9k tokens x 55 KB (709 B per layer, 78 layers) = 2 GB per rank |
| outside the static | 44.1, above the 38.6 SGLang would reserve itself at bs 32 |
| colocated sync | 132.4 + 16.9 (trainer BF16 weights) + 4 (staging) = 153.3 of 178 |

Checked on 4 nodes (`slime-glm5-l10-grpo-4n`, job 1265529): SGLang logs `mem_fraction_static: 0.75` and decode graphs
to 32, a KV pool of 18,688,064 tokens (123.36 GB: 132.4 less the 10-layer engine's 8.7 GiB), 28.2-28.5 GB free after
it; weight sync 9.0 s then 6.7 s, training 156 s.

`slime-glm5-rollout-8n` (8 nodes, a request row): the full GLM-5.3 on one 64-GPU engine at the real width, slime's
`--debug-rollout-only` (engines from the FP8 checkpoint, no Megatron model, no training, no weight sync), 2 rollouts of
GLM-RL-01's share per engine (1,024 samples over 4 engines: 32 prompts x 8), prefill graph off as in the colocated run
(`--sglang-cuda-graph-backend-prefill disabled`: rollout-only turns rollout offload off, slime `arguments.py:2115-2127`,
and with it the memory saver's setting, `sglang_engine.py:710-716`): checks the engine at TP64 and measures the rollout
time GLM-RL-01 only estimated (20-45 min). milabench gets no metrics from it (its hook runs in the Megatron actors): the
times are slime's `Engine startup timings (s): load_weight=` and `rollout_metrics.py:286 - perf N: {...
'perf/rollout_time': ...}` lines. `--time 2:15:00`: startup 10-20 min (each node reads the 756 GB checkpoint, about 7
min at 1.8 GB/s), 2 x 20-45 min, 15 min margin and the install.

### GLM-RL-02 (job 1265764): the engine at TP64, rollout only

8 nodes (trig[0115-0121,0123]), 08:27:40-09:07:51. Slurm says FAILED (exit 1) because milabench got no training rate,
as expected without a trainer; slime returned cleanly. Against the estimates:

| | measured | estimate |
|---|---|---|
| engine up | 7 min 47 s after the job's start; `load_weight` 132.6 s (ranks 128.5-135.6 s), decode graphs 94.2 s, scheduler 258.8 s | startup 10-20 min; load about 7 min at 1.8 GB/s per node |
| checkpoint read | every node reads all 756 GB once (SGLang's iterator `get_tensor`s every tensor, `weight_utils.py:1160-1162`; its 8 ranks share the page cache): 6.0 TB in 128-136 s, 5.7 GB/s per node, 46 GB/s in all (if cold) | |
| memory | `mem_fraction_static` 0.75, decode graphs to 32; weights 32.26 GB per GPU; KV 100.04 GB, 1,991,296 tokens per rank; 36.1-36.5 GB free after it; GPU peak 150.0 GiB; host 120 GB | weights about 33, KV about 99; rollout about 145 GiB |
| rollout 0 | 898.2 s: 424 tokens/s per engine, 6.63 per GPU | 20-45 min |
| rollout 1 | 949.9 s: 535 tokens/s per engine, 8.37 per GPU | |
| decode step | 100-113 ms (one sequence: 8.8-9.9 tokens/s; 4 per rank: 35 tokens/s) | 150-300 ms |
| response length | mean / median / max / min 1,488 / 457 / 8,192 / 62, then 1,986 / 687 / 8,192 / 95 | |

Reward and `rollout/truncated` are not logged in rollout-only mode (slime's `rollout N:` line comes after the
`debug_rollout_only` return, `rollout.py:377`); what is: groups of 8 with identical rewards (no GRPO signal), 14 all 1
and 2 all 0 of 32 in rollout 0, 10 all 1 in rollout 1.

The tail. Ray deduplicates the `Decode batch` lines across ranks (`[repeated Nx across cluster]`); each busy rank
logs one every 40 decode steps (about 4.4 s), so the lines and their repeats per minute give the busy ranks. Rollout 0:
about 60 of 64 in the first two minutes, 45 by minute 2, about 25 (one sequence each: 10 % of the 256) from minute 6,
and still about 19 in the last minute. Rollout 1: about 25 from minute 9 to minute 14. The last 10 % of the responses
kept the engine busy 9 and 6 of the 15 minutes, and about 20 (8 %) ran to the 8,192 cap. The rollout time is the
longest response: 8,192 tokens x 104-113 ms = 14.2-15.4 min, plus about 30 s of prefill and the first tokens,
whatever the batch: 4 engines (GLM-RL-03) take the same time if each has a response at the cap.

### GLM-RL-03 (draft): the 32-node colocated repost

`slime-glm5-grpo` at the current commit, 32 nodes, `--time 2:05:00` = 30 (startup) + 3 x 25 (rollout 18: the
measured 15.0-15.8 min, the max over 4 engines and a longer tail; KV onload and offload 0.5; training 2.5-4.5; sync
1.5-2) + 3 (warm-up training) + 15 margin = 123 min.

Reads from /scratch at the start: the 4 engines first (`train.py:109`), each node all of the 756 GB HF checkpoint:
24.2 TB, 2.2 min at GLM-RL-02's 5.7 GB/s per node, 9 min if /scratch stays at its 46 GB/s in all; then the trainer's
torch_dist once (fully parallel load, `megatron_utils/arguments.py:173`): 1.49 TB. About 25.7 TB in the first 10-20
min. (`--sglang-load-format dummy` would skip the 24.2 TB, the weights coming from the first sync: untested.)

Expected lines, in order: `install: ok`, `main: train.main`; per engine (4) `mem_fraction_static': 0.75`, `Load weight
end ... mem usage=32.26 GB`, `KV Cache is allocated ... #tokens:` about 1,991,296, `Engine startup timings (s):
load_weight=`; `expert_routing.py:369 - Enabled rank-local expert update: Megatron PP=8 EP=32, SGLang EP=64`; `Timer
update_weights end` (1.5-2 min); `rollout_metrics.py:286 - perf 0: {... 'perf/rollout_time': 900-1080`; `Timer train
end` (2.5-4.5 min, the first longer); `train_metric_utils.py:404 - perf 0`.

Cancel (`scancel -b -s USR1`): any `Scheduler hit an exception`, `Not enough memory`, `CUDA out of memory`, `Invalid
device_uuid`, NCCL error or `main: FAILED`; fewer than 4 `Engine startup timings` lines at 30 min; no `Timer
update_weights end` at 40 min; no `perf N` rollout line 22 min after the previous sync (measured 15.0-15.8: 8,192 tokens
x 113 ms + 0.5 min bounds it); no `Timer train end` 10 min after it; a sync over 5 min.

### GLM-RL-03 (job 1266065): GLM-5.3 GRPO on 32 nodes

trig[0101-0113,0115-0121,0123-0124,0126-0131,0133-0136], 09:17:13-10:52:02 (1:34:49), exit 0, milabench PASS, 3
steps. Step 0 is the warm-up (its training: 634 s, the 103 TileLang compiles); steps 1-2 against the estimates:

| per step | step 1 | step 2 | estimate |
|---|---|---|---|
| rollout | 1,047.3 s (17.5 min) | 1,111.0 s (18.5 min) | 18 min (GLM-RL-02: 15.0-15.8 for one engine) |
| rollout tokens/s per GPU | 5.72 | 4.46 | GLM-RL-02: 6.6-8.4 |
| training | 215.2 s | 202.7 s | 2.5-4.5 min |
| training tokens/s (global) | 7,881 (1.69 M tokens) | 7,046 (1.42 M) | 25-30k on 3-6 M tokens |
| sync | 34.7 s | 32.8 s | 1.5-2 min |
| trainer offload (`sleep`) + onload (`wake_up`) | 16.5 + 12.0 s | 11.4 + 14.8 s | 0.5 min |
| other wait (the engine's onload of weights and KV, data) | 12.6 s | 11.3 s | |
| step | 1,338.3 s (22.3 min) | 1,384.1 s (23.1 min) | 25 min |
| response mean / median, truncated | 1,496 / 589, 7.0 % | 1,239 / 530, 4.6 % | |
| raw reward, grad norm | 0.735, 0.061 | 0.767, 0.067 | |
| TIS mean, abs, clip fraction | 1.000, 0.081, 1.7 % | 1.000, 0.073, 1.5 % | (10-layer: 0.059, 1e-4) |

Step 0: rollout 1,209.7 s, training 634.1 s (2,640 tokens/s), sync 33.7 s, truncated 5.4 %, reward 0.736. Where the
step goes (steps 1-2): rollout 79 %, training 15 %, sync 2.5 %, the trainer's offload and onload 2 %, the rest 1 %.

Training took the estimated time for other reasons: 1.4-1.7 M tokens instead of 3-6 M (responses of 1.2-1.5k tokens
on average), so about 50k tokens and 7 micro-batches of 8,192 per DP rank, and a PP8 bubble of 7 / 14 = 50 % instead of
18-30 %; that brings the estimate to about 18k tokens/s, still 2.4 times the measured. The logs do not split the rest
(candidates: EP32's all-to-all over 4 nodes, where the 10-layer test's EP8 stayed in a node; the CPU Adam step, fixed
per step). Sync 33 s, not 1.5-2 min (ten times the 10-layer test's was too much).

The tail (rough: busy ranks from the `Decode batch` lines and their Ray repeat counts, at one line per 40 decode steps,
4.2-6.7 s): the last 10 % of the responses (about 100 of 1,024) ran from minute 6-9 to the end, 9-12 of the 17.5-18.5
minutes (55-70 % of the rollout, 40-50 % of the step); the last 25 or so from minute 14.

The 6.0 instead of 9.5-9.9 tokens/s at one sequence per rank: per engine. Grouping the nodes by IP 8 at a time (assumed
to be the engines; no line names them, but every group then switches at one time), every node of an engine switches
within the same minute, once per rollout, 6.5-13 min in (rollout 0: engines 0, 1, 2 at 6.5, 10.5, 13 min; rollout 1:
engines 2 and 0 at 9 and 11.5), and some engines never. Not the idle ranks' dummy batches: ranks go idle from minute 2
at full speed, and GLM-RL-02's engine kept 8.8-9.9 to its end with as many idle. Not the KV length (the fast engines
reach 8.3k tokens at 10), the CUDA graph (on in every line), a queue or a retraction (none), nor SGLang's DP padding
mode (`dp_attention.py:182-189`, by token counts, the same for all engines). On engine 0's first node (the only one
milabench monitors) the GPUs go from 331 to 297 W at the switch while their load rises from 0.92 to 0.95: they wait
longer per step, so the slower part is elsewhere in the engine. Not determined; it costs about 2.5 min per rollout (some
2.5k tokens x 62 ms more), the difference with GLM-RL-02. Per-node GPU clocks and power and CPU frequency on every node
during a rollout would tell.

Memory. GPU, the first node (stage 0, nvml): rollout 149.8-150.9 GiB (estimate about 145), training 81.3-84.2 GiB
(85-115), sync 58.9 GiB (155-160: the KV is onloaded after the sync, `train.py:19, 31`, not before). Host, slime's
`Memory-Usage` snapshots (at offload and sync, not continuous): the last stage's 4 nodes (trig0133-0136) 992-998 GB
(estimate 1.3-1.35 TB), the first node 784 GB (milabench: 816-845 GB), the others 295-557 GB.

Startup, 16.8 min to the first sync (estimate 30 with margin): install to 09:18:41; the 4 engines up at
09:24:15-09:24:33 (`load_weight` 125.8-128.6 s, decode graphs 94-95 s); the trainer's actors at 09:25:26, its model at
09:31:10, its checkpoint loaded by 09:32:18; offload, then the sync to 09:34:03. /scratch: each node reads all of the
HF checkpoint in 126-137 s (one rank 45.8 s: GLM-RL-02's 8 nodes may have had it cached), so at least 24 x 756 GB =
18.1 TB in about 2.2 min (135 GB/s or more, 5.5-6.0 GB/s per node); the trainer's 1.49 TB in at most 68 s (22 GB/s or
more).

Three things in the log that are not in the tables:
- Rollout 2's `perf/decode/throughput` mean 180,768 and max 37,964,777 tokens/s: SGLang's per-request (completion
  tokens - 1) / (finished_time - first_token_time) (`observability/req_time_stats.py:488-489, 517-519`), divided by an
  interval near 0 for a few short requests (rollout 2 has a 17-token response and a 3.75 s request, against 54-58 tokens
  and 8.6-25 s in rollouts 0-1). The mean is not usable; the median is: 9.53 tokens/s (rollouts 0-1: 9.11 and 9.56,
  means 10.24 and 10.28), as the `Decode batch` lines' 9.5-9.9 at one sequence per rank.
- About 233 `CUDA error: context is destroyed` lines (230 with Ray's repeat counts, and 4 `invalid device context`) at
  10:51:20-21, from the trainer's actors only, after slime's loop had ended (the last sync at 10:51:18, the rollout
  health monitor stopped at 10:51:19): pinned host tensors freed in `~TensorImpl` through PyTorch's
  `CachingHostAllocator` after the actor's CUDA context was gone, at process exit. Teardown order, harmless here: no
  checkpoint is saved, the metrics were logged, exit 0, milabench PASS, every node's GPUs free after 0 s, no drain.
- The fourth weight sync (31.7 s, 10:51:18) is slime's update after the last training step (`train.py:85-89`, run
  after every step, for a final evaluation). Each step's `perf/update_weights_time` is the sync before its rollout
  (33.7, 34.7, 32.8 s: the first sync, then after steps 0 and 1), so the per-step numbers have one each; the fourth
  belongs to no step (in a longer run it is the next one's) and adds about 1 min (with the offload and onload) to the
  job.

### GLM-RL-04 (draft): GLM-RL-03 with rollout routing replay

`slime-glm5-grpo-r3` (`slime-glm5-grpo` + `--use-rollout-routing-replay`; slime's sglang-router fork is installed), 32
nodes, the user's with / without comparison against GLM-RL-03. `--time 2:10:00` = 20 (startup; 16.8 measured) + step 0
(rollout 22 + training 20, the warm-up's cancel bound + 1.5) + 2 x (20 + 4.5 + 1.5) + 10 margin = 125.5 min.

Memory against GLM-RL-03. Host: SGLang pins a routed-experts cache per engine rank sized by its KV pool
(`state_capturer/base.py:69-72`: 1,982,976 tokens x 78 layers x top-8 x int32 = 4.95 GB), 39.6 GB per node, on every
node; and each step's routes, 1.4-1.7 M tokens x 78 x 8 x 4 B = 3.5-4.2 GB, through the rollout manager to the
trainer. The last stage's nodes: 998 + 40 = about 1.04 TB; the first node 0.83-0.89 TB; of 2.0 TB. (Qwen's +72 GB on
its first node is the same cache: a much larger KV pool per rank.) GPU: +0.33 GB per GPU (the capture buffer,
max(256, 2048) x 64 DP x 78 x 8 x int32), outside the static memory (44 GiB there; peak 150.9 GiB).

Expected lines and cancel rules as GLM-RL-03, with its numbers: engines up about 7 min after the start, the first sync
by 17-20 min, rollouts 17.5-20.2 min, training 3.4-3.6 min after step 0 (10.6 min), sync 33-35 s. Cancel: the
GLM-RL-03 errors; no `perf N` rollout line 22 min after a sync; step 0's training over 20 min, later ones over 10;
`SGLang routed_experts element count does not match` (slime `types.py:447`). Routing replay has not run on GLM-5.3
yet; a 4-node bring-up first (`slime-glm5-l10-grpo-4n` with it) would check the engine's capture and the trainer's
replay.

### Disaggregated bring-up (job 1266823)

`slime-glm5-l10-grpo-disagg-4n` on trig0115-0118: the trainer on 2 nodes (PP2 x EP8, DP8), one 16-GPU engine on the
other 2 (the head node), the prefill graph disabled as in the colocated engine. PASS, 3 steps in 21:29, NVRM 0, nodes
back to reserved. Against the colocated 2-node test with the same trainer and engine (job 1264931), steps 1-2:

| per step | disaggregated, 4 nodes | colocated, 2 nodes |
|---|---|---|
| rollout | 154.2 s (415 tokens/s per GPU) | 158.1 s (406) |
| training | 29.9 s (36.2k tokens/s) | 61.0 s (17.6k) |
| sync | 3.9 s (NCCL broadcast; 12.6 s the first) | 8.4 s (CUDA IPC) |
| trainer offload + onload | none | 11.5 + 3.8 s |
| step | 196.5 s | 251.1 s |
| node-seconds per step | 786 | 502 |

The NCCL sync works: from each pipeline stage's DP0 rank to the engine, 3.4-4.4 s after the first. Training takes half
the time, and not because of the offload, which is timed apart (`sleep`, `wake_up`); not determined (candidates: the
idle engine's processes beside the CPU Adam step, slime's reloadable process groups, `offload_train` only). The engine's
node holds 158.5-158.8 GiB in rollouts and syncs alike (132.4 static + 26 outside it, more than the colocated engines'
150; no visible sync peak). The trainer's nodes are not monitored by milabench. slime's `train.py` is synchronous: each
side waits while the other works, so on 4 nodes a step costs 1.57 times the node-seconds of the colocated 2.

### GLM-RL-05 (draft): disaggregated on 32 nodes

`slime-glm5-grpo-disagg`: the trainer on 16 nodes (TP1 / CP1, PP8 x EP16, DP16), 2 engines of 64 GPUs on 16 (8
sequences per DP-attention rank), no offload, NCCL sync. Estimates from GLM-RL-03 and the bring-up:
- rollout 17-21 min (GLM-RL-03's 17.5-18.5 at 4 sequences per rank; the longest response sets it, at up to 10 % more
  per decode step with 8);
- training 2.6-5.2 min: GLM-RL-03's 209 s x 2 (half the GPUs) / 1.33 (about 14 micro-batches per DP rank, a bubble of
  33 % instead of 50 %) = 5.2 min, or half that if the bring-up's 2x without offload carries over;
- sync 0.5-1.5 min (8 stage sources at once, each its stage's full weights to 128 engine GPUs);
- no offload or onload (GLM-RL-03: about 40 s per step);
- step 20-28 min, against GLM-RL-03's 22.3-23.1.

Memory: trainer GPU 77.8 GiB static (13.9 B parameters on the largest stage) + 31-33 GiB as GLM-RL-03's stage 0 above
its static = 110-115 GiB; engine GPU 150-159 GiB (GLM-RL-03's engines; the bring-up's engine without memory saver:
158.8); host on the trainer's last stage's nodes: Adam for 79 B parameters 948 GB + about 120 GB (GLM-RL-03's rest
without its offload copies) = 1.05-1.2 TB of 2.0; the engine nodes about 120 GB (GLM-RL-02). /scratch at the start: 16
nodes x 756 GB = 12.1 TB of HF checkpoint, then 1.49 TB of torch_dist.

`--time 2:10:00` = 20 (startup) + step 0 (rollout 22 + training 20 + 1.5) + 2 x 28 + 10 margin = 129.5 min. Expected
lines: as GLM-RL-03 for 2 engines, without `Timer sleep` / `wake_up`; `Timer update_weights end` 0.5-1.5 min. Cancel:
the GLM-RL-03 errors; fewer than 2 `Engine startup timings` at 25 min; no first sync at 40 min; a rollout over 25
min; step 0's training over 20 min, later ones over 12; a sync over 5 min.

### Step 4, prepared entries (not posted)

GLM-RL-01 is `slime-glm5-grpo` (posted at de8a4c63). Next, after it:

| entry | what | estimates |
|---|---|---|
| `slime-glm5-grpo-r3` | GLM-RL-01 with rollout routing replay (the user's comparison; slime's sglang-router fork is in the venv) | as GLM-RL-03; host +about 40 GB on every node (SGLang's pinned routed-experts cache) and 3.5-4.2 GB of routes per step: GLM-RL-04 above. On Qwen3-30B-A3B the times did not differ beyond the spread |
| `slime-glm5-l10-grpo-disagg-4n` | bring-up, 4 nodes: the 10-layer checkpoint, trainer on 2 nodes (TP1 / CP1, PP2 x EP8), one 16-GPU engine on the other 2 | PASS (job 1266823, above): the NCCL sync (`update_weight_from_distributed`) works, 3.4-4.4 s per step |
| `slime-glm5-grpo-disagg` | 32 nodes: trainer on 16 (TP1 / CP1, PP8 x EP16, DP16), 2 SGLang engines of 64 GPUs on 16 | trainer GPU: 13.9 B parameters on the largest stage, 77.8 GiB static, about 125-135 GiB with activations, no engine beside it; host on its last stage's nodes about 1.2-1.4 TB (Adam for 79 B parameters, no offload copies). Per step: rollout 20-50 min (8 sequences per DP-attention rank), training 5-9 min (half the GPUs, bubble 10-18 %), sync 1-3 min (8 stage sources at once), no offload. slime's `train.py` is synchronous: each side waits while the other works |

## Untested

- More than 2 nodes, the GLM-5.3 entries, disaggregated rollout across nodes.
- SGLang's flashinfer MoE kernels on B200 (cubins offline: `scripts/trillium/kimi/kernels.sh` stages FlashInfer's).
