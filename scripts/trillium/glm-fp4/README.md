# GLM-FP4: GLM-5 pretraining in NVFP4 on Trillium (B200)

The campaign's GLM-5 setup (`scripts/trillium/glm/README.md`, pack `benchmarks/megatron`, layout pp13-ep8) with
TransformerEngine's NVFP4 recipe instead of BF16 / FP8 (`ops/GLM_FP4_BRIEF.md`). Cluster directory
`$SCRATCH/trillium/new/glm-fp4` (`G` below): checkout `src` (branch `trillium-new`), venv `venv/torch`, uv cache `uv-cache`.

## Step 1: venv and feasibility (done 2026-11-08)

| what | how |
|---|---|
| venv, downloads | login node: `git clone -b trillium-new https://github.com/milabench/milabench.git $G/src`, milabench's own venv in `$G/src/.venv` (`uv venv --python 3.12.13 --seed`, `uv pip install -e ./src pybind11 nvidia-ml-py`, `UV_CACHE_DIR=$G/uv-cache UV_LINK_MODE=copy`), the campaign's TE files in `$G/wheelhouse`, then `./install-login.sh` |
| TE build, NVFP4 check | one B200 node: `te_build.sbatch` (the submit line is at its top) |

The venv has the campaign venv's versions (`$SCRATCH/trillium/glm/venv/torch`, read only): `install-login.sh` writes
`pins-campaign.txt` from its dist-info directories and installs those versions. Checked after the build: the same 283
dist-info directories (name and version), the same file hashes (RECORD) for 281 of them; the two others are benchmate
(editable, this checkout) and `transformer_engine_torch`, whose extension is compiled here from the same sdist (sha256
equal). `torch_memory_saver` is missing, as in the campaign. Megatron-LM (4b4acac, slime's patches) and slime (a4202e0)
checkouts: `diff -r` equal to the campaign's.

`nvfp4_check.py` (job 1270060, trig0152): TE 2.18.0's `NVFP4BlockScaling()` on GLM-5's GEMM shapes, one micro-batch
of pp13-ep8 (8192 tokens; 32 local experts, 2048 tokens each). Measured:

- quantizers TE builds: input RHT, weights 2D (16x16), gradients stochastic rounding and RHT. TE applies RHT to the
  column-wise copies only, which the weight-gradient GEMM uses (`transformer_engine/common/recipe/__init__.py`);
- all 3,744 GEMM calls of the NVFP4 pass ran with E2M1 operands and E4M3 scales per 16 values (cuBLASLt trace:
  `R_4F_E2M1`, `VEC16_UE4M3`), none fell back, the N = 576 and N = 128 projections included;
- per micro-batch (no-grad forward + forward and backward, as under full recompute), in ms, BF16 / MXFP8 / NVFP4:
  expert fc1 10.47 / 6.74 / 4.90, expert fc2 5.13 / 4.20 / 4.73, o_proj 4.33 / 2.69 / 1.69, shared fc1 1.10 / 0.67 /
  0.63, q_down 0.57 / 0.47 / 0.64, shared fc2 0.56 / 0.44 / 0.62, indexer wq_b 0.44 / 0.45 / 0.62, kv_down 0.29 / 0.45 /
  0.65, indexer wk 0.28 / 0.44 / 0.61; sum 23.2 / 16.6 / 15.1 (MXFP8 0.71x BF16, NVFP4 0.65x BF16 and 0.91x MXFP8);
- error against BF16 (random data, the same weights): output 3.8 % MXFP8, 14.6 % NVFP4; weight gradient 3.8 % / 16.7 %;
  no NaN.

`q_up` and the dense fc1 (LayerNormLinear) failed there: TE's RMSNorm compiles at run time and found no CUDA headers,
which milabench sets for runs and the standalone check did not; `job.sbatch check:glm5` sets them (job 1270151: all 12
modules, MoE layer 24.54 / 17.95 / 15.70 ms, NVFP4 1.14x MXFP8). Its memory column was wrong (the previous recipe's
buffers were freed after its baseline reading) and is gone.

## Not run yet

Steps 3-4 of the brief.

## The recipe (`dev.yaml`, `_tri_nvfp4`)

`--fp4-format e2m1` (Megatron's `--fp4-recipe` default, `nvfp4`: TE's `NVFP4BlockScaling` with its defaults),
`--moe-router-padding-for-quantization` (expert token counts padded to 128), and the last 5 of 78 layers in BF16
(`--first-last-layers-bf16 --num-layers-at-start-in-bf16 0 --num-layers-at-end-in-bf16 5`): the last stage of pp13, so
the BF16 layers sit on the stage that had compute to spare in BF16 and the pipeline stays balanced. 5 layers (6.4 %) is
the low end of NVIDIA's report (fewer than 15 % of the layers, the last ones; its 12B model kept 16 %, and was stable with
the last four blocks). If NVFP4's loss gap to BF16 is large, a larger N is the first thing to try. Kept as in the
campaign, not as in the report: Adam's moments in BF16 (precision-aware optimizer). `workload.precision` is `nvfp4`,
`precision_detail` lists the recipe's parts and the BF16 layers.

## Step 2: one node, the truncated model

`megatron-glm5-fp4-test-n1-{bf16,nvfp4}`: the campaign's B200 smoke model (8 layers: 1 dense + 7 MoE, 64 experts, seq
4096, batch 32) at PP1 x EP8 on one node, 50 steps with the C-loss settings in small (seed 1234, warm-up 5, lr 3e-4
cosine); NVFP4 keeps the last layer of 8 in BF16. `-nvfp4-lt`: 3 steps, run as `lt:` for cuBLASLt's trace.
Data, once, from a login node: `MILABENCH_BASE=$G milabench prepare --config $G/src/benchmarks/megatron/dev.yaml --select
megatron-glm5-fp4-test-n1-bf16` (GLM-5.3's config and tokenizer, 200,000 FineWeb documents, under `$G/data`). Job:
`job.sbatch megatron-glm5-fp4-test-n1-bf16 megatron-glm5-fp4-test-n1-nvfp4 lt:megatron-glm5-fp4-test-n1-nvfp4-lt`
(submit line at its top). The campaign ran this model on 2 nodes only (PP2), so the BF16 step time has no campaign
reference at one node: that comparison is step 3's, on `megatron-glm5-tri-test-b200-n2-gbs32`.

Measured (jobs 1270151 and 1270295, trig0145; rank 0's step rows, step time over steps 6-50; peak allocated over the 8
ranks; `-mxfp8` is the campaign's recipe on the same model, added for its memory):

| | BF16 | MXFP8 | NVFP4 |
|---|---|---|---|
| loss step 1 / 6 / 21 / 50 | 13.16 / 22.43 / 9.16 / 7.894 | 13.16 / 21.72 / 12.79 / 7.899 | 13.16 / 24.57 / 12.60 / 7.912 |
| loss, mean of steps 41-50 | 7.919 | 7.941 | 7.972 |
| NaN, max grad norm | none, 184 | none, 171 | none, 197 |
| step time median (sd) | 3.227 s (0.33) | 2.958 s (0.20) | 2.916 s (0.08) |
| peak allocated, mean (max) of 8 ranks | 43.98 (46.24) GiB | 59.22 (59.22) GiB | 48.06 (49.66) GiB |

The loss goes down in all three; NVFP4 runs further from BF16 between steps 6 and 25 and ends 0.053 above it over steps
41-50, MXFP8 0.022 (one run each: no repeat to read the gaps against). Memory over BF16 per quantized parameter
(3.88 B per rank in MXFP8, 3.38 B in NVFP4 without the BF16 layer): MXFP8 4.2 bytes (the campaign's pp13-ep8: 4.2-5.8),
NVFP4 1.3 bytes (its two copies: 1.125). Scaled to pp13-ep8 (8.43 B per rank in the middle stages), an estimate: NVFP4
about 100 GiB at the largest rank, against 90.0 BF16 and 132.3 MXFP8 measured (job 1242058).

Which GEMMs ran in E2M1 (`-nvfp4-lt`, job 1270295, rank 0, 3 steps, cuBLASLt trace): 6,960 calls in E2M1 with E4M3 scales
per 16 values, 85.6 % of the traced FLOPs: the attention projections, the indexer's `wq_b` and `wk`, the dense MLP of
layer 1 and the expert GEMMs (their token counts all multiples of 128: the padding); 1,104 in BF16 (14.4 %) at the same
attention and expert shapes, the BF16 last layer (`o_proj`: 336 E2M1, 24 BF16), and a few BF16-only shapes; the router
in FP32. The output layer and the plugin's einsums go through torch's cuBLAS path, which this trace does not show.

## Step 3: four nodes, pp13-ep8's stages (job 1270436)

`megatron-glm5-fp4-n4-{bf16,mxfp8,nvfp4}`: 24 layers (3 dense + 21 MoE), 256 experts, EP8, seq 8192, PP4 split 7 / 6 /
6 / 5 (pp13's first stage, two middle ones, its last), 256 sequences (32 micro-batches per data-parallel rank, as
pp13-ep8), 8 steps; NVFP4's BF16 tail is the last stage (5 layers). trig[0122-0123,0133,0145], 28.5 min. Rank 0's step
rows (steps 3-8), peak allocated over each stage's 8 ranks:

| | BF16 | MXFP8 | NVFP4 |
|---|---|---|---|
| step time median (min-max) | 38.87 s (38.79-39.72) | 36.99 s (36.95-37.35) | 37.06 s (36.95-38.44) |
| peak allocated, stage 0 / 1 / 2 / 3 (GiB) | 68.3-69.0 / 82.7-83.2 / 82.1-82.7 / 73.5-73.7 | 101.5 / 132.3 / 132.3 / 115.2 | 76.4-76.9 / 92.5-92.9 / 91.9-92.5 / 74.4-74.9 |
| loss step 1 / 8 | 13.160 / 12.209 | 13.164 / 12.210 | 13.162 / 12.246 |

NVFP4 is as fast as MXFP8 (0.2 % apart), both 4.8 % faster than BF16; at 13 nodes the campaign measured MXFP8 5.4 % faster
than BF16 (45.97 against 48.61 s, job 1242058). The BF16 step time is the 13-node one scaled by the bubble, (32 + 3) /
(32 + 12) x 48.61 = 38.7 s. NVFP4 adds 9.7 GiB on a middle stage over BF16 (8.43 B quantized parameters per rank: 1.23
bytes each), nothing on the last (its layers are the BF16 tail); MXFP8 adds 49.6 GiB, its stage peaks equal to the
campaign's at 13 nodes. Estimate for pp13-ep8 NVFP4: about 100 GiB at the largest rank (the middle stages hold more
micro-batches in flight at PP13: +5 to +7 GiB in BF16). Loss at a learning rate of 1e-6 (the speed entries) only shows
that all three train.

## GLM-FP4-01: C-speed at 13 nodes (job 1271210, 2026-11-08 21:03-22:21, COMPLETED)

`job.sbatch exp:C-speed megatron-glm5-tri-pp13-ep8-n13-gbs256{,-mxfp8,-nvfp4}` at 57b77658 on
trig[0129-0131,0136-0137,0139-0144,0147,0149], pp13-ep8, 256 sequences, 20 steps, all three PASS. Timer: rank 0's
per-rank metrics `elapsed` (CUDA events), steps 3-20 (the `.0.data` rate events use the same timer here); peak allocated
over all 104 ranks:

| | BF16 | MXFP8 | NVFP4 |
|---|---|---|---|
| step time mean (sd), median | 48.59 s (1.06), 48.38 | 45.82 s (0.61), 46.00 | 46.05 s (0.96), 45.65 |
| the campaign's job 1242058, mean | 48.61 s | 45.97 s | |
| against BF16 (means) | | 0.943 | 0.948 |
| steps 3-5 then 18-20 | 47.29 then 50.17 s | 44.93 then 46.41 | 45.02 then 47.02 |
| peak allocated, largest stage / last stage | 90.03 / 78.9 GiB | 132.34 / 115.2 | 98.64 / 80.0 |
| loss at step 20 | 9.446 | 9.444 | 9.526 |

NVFP4 equals MXFP8 on GLM-5 at 13 nodes, for this split: +0.23 s by the means, -0.35 s by the medians, within the
steps' spread. Estimated before the run: 45.7 s, about 1 % faster than MXFP8 (steps 1 and 3); measured 46.05 s. The
largest-stage peak, 98.6 GiB, is the about 100 GiB estimated from the 4-node test. The BF16 and MXFP8 times repeat the
campaign's within 0.4 %. Loss at step 20: one run each, 20 steps into training at the speed entries' constant learning
rate of 1e-6: not a precision comparison.

Recipe as run (`run_start` `precision_detail`, Megatron's argument dump): MXFP8 `fp8` e4m3, recipe mxfp8; NVFP4 `fp4`
e2m1, recipe nvfp4, Hadamard transforms on the weight-gradient inputs, 16x16 weight blocks, stochastic rounding, 5 BF16
layers at the end. With `--decoder-last-pipeline-num-layers 5` the last stage (trig0149) runs entirely in BF16.

trig0149 and the BF16 tail: trig0149 is at the software power cap (`clocks_event_reasons` 0x4) in 17 / 8 / 17 % of its
training-window samples in BF16 / MXFP8 / NVFP4 (the results review, with another window: 22 / 10 / 22 %), on all 8
GPUs, down to 1,485 MHz; it is the only node capped in MXFP8 and NVFP4 (the others at most 0.9 %), while in BF16 every
node is (8-13 %). In NVFP4 it is the slowest stage (the review, from the straggler lines: GEMMs 5.2-5.7 s), so the
all-BF16 last stage probably sets NVFP4's step (medium confidence); the cap costs at most about 0.5 s per step (the
review's estimate). Not a hardware fault: no Xid, ECC or drains.

The drift: the step time rises through the run in all three entries and in 1242058 (above). Not throttling: the share
of capped samples falls as the steps grow longer (the review). The all-to-all time grows from step 10 in every stage,
with expert balancing off (`--moe-aux-loss-coeff 0`, `--moe-router-bias-update-rate 0`): likely a growing load imbalance
(the review, medium confidence). It is the same in every entry, so it does not bias the comparison.

## Qwen3 32B's GEMMs (`nvfp4_check.py qwen3`, job 1270151)

The same modules for a Qwen3 32B layer (te.LayerNormLinear for QKV 5120 -> 10240 and gate+up 5120 -> 51200, te.Linear
for O 8192 -> 5120 and down 25600 -> 5120), forward + dgrad + wgrad per call, weights cached (in parentheses: quantized
again in every call), ms, BF16 / MXFP8 / NVFP4:

| tokens | QKV | O | gate+up | down | layer | speed-up MXFP8 / NVFP4 |
|---|---|---|---|---|---|---|
| 4096 | 1.02 / 0.68 / 0.58 | 0.71 / 0.43 / 0.47 | 4.69 / 2.96 / 2.20 | 2.13 / 1.59 / 1.07 | 8.55 / 5.66 / 4.32 (9.04 / 6.16 / 4.96) | 1.51x / 1.98x (1.47x / 1.82x) |
| 8192 | 2.24 / 1.40 / 1.15 | 1.32 / 0.88 / 0.70 | 9.78 / 5.56 / 3.97 | 4.58 / 2.81 / 1.72 | 17.91 / 10.64 / 7.54 (17.91 / 11.25 / 8.02) | 1.68x / 2.37x (1.59x / 2.23x) |

One run of 10 calls each after 3 warm-up calls: no repeat.

## MoE expert GEMMs of other models (`nvfp4_check.py moe`, job 1270575)

GEMMs only: the routed experts' fc1 (gate + up fused) and fc2 through te.GroupedLinear, the path Megatron's
TEGroupedLinear takes (TE's per-expert GEMMs; its fused grouped GEMM needs cuBLASLt 13.3, these pins have cuBLAS
13.1.1.3), forward + dgrad + wgrad, weights cached, activation and gradient quantization and the Hadamard transforms
in every call. Shapes from each model's `config.json`; experts per GPU at EP8. Attention differs per model (sparse MLA
in GLM-5, MLA in DeepSeek-V3 and Kimi K3, GQA in Qwen3 and gpt-oss) and is not measured here, nor are the router, the
token dispatch and the shared experts.

NVFP4 over BF16 / NVFP4 over MXFP8 (fc1 + fc2 time), by tokens per expert:

| model (hidden, expert FFN, experts, top-k; experts per GPU) | 128 | 512 | 2048 | 8192 | tokens per expert at 8192 tokens per micro-batch, EP8 |
|---|---|---|---|---|---|
| DeepSeek-V3 (7168, 2048, 256, 8; 32) | 0.54x / 0.80x | 0.73x / 0.79x | 1.84x / 1.28x | 2.49x / 1.48x | 2048 |
| GLM-5 (6144, 2048, 256, 8; 32) | 0.50x / 0.80x | 0.63x / 0.79x | 1.65x / 1.13x | 2.32x / 1.47x | 2048 (the campaign's C-speed) |
| Qwen3-235B-A22B (4096, 1536, 128, 8; 16) | 0.48x / 0.78x | 0.48x / 0.77x | 0.70x / 0.79x | 1.83x / 1.29x | 4096 |
| Qwen3-30B-A3B (2048, 768, 128, 8; 16) | 0.48x / 0.77x | 0.48x / 0.77x | 0.48x / 0.77x | 0.68x / 0.81x | 4096 |
| gpt-oss-120b (2880, 2880, 128, 4; 16) | 0.38x / 0.65x | 0.38x / 0.65x | 0.74x / 0.76x | 1.52x / 1.12x | 2048 |
| Kimi K3 (3584 latent, 3072, 896, 16; 112) | NVFP4 fails | | | | 1170 |
| Kimi K3 at EP16 (56 experts per GPU; job 1270744) | 0.50x / 0.79x | 0.62x / 0.79x | 1.62x / 1.13x | 2.26x / 1.40x | 2341 |

Below a few thousand tokens per expert the times hardly change with the token count (Qwen3-30B-A3B: 2.1 / 3.3 / 4.3 ms
BF16 / MXFP8 / NVFP4 from 128 to 2048): the GPU waits for the host to issue the per-expert GEMMs and quantization
kernels, and NVFP4 issues the most. In a training step part of that issue time can overlap other GPU work, which this
microbenchmark does not show. BF16 and MXFP8 run for Kimi K3 at EP8 (8192 tokens per expert: fc1 97.4 / 60.7 ms, fc2
50.0 / 32.5 ms); NVFP4 does not, in TE 2.18 (not fixed):

Kimi K3, EP8, 112 experts per GPU, NVFP4: `RuntimeError: transformer_engine/common/hadamard_transform/group_hadamard_transform.cu:408 in function group_hadamard_transform_amax: Assertion failed: num_tensors <= kMaxTensorsPerKernel`

## Qwen3 32B dense (`megatron-qwen3-32b-*`, `dev.yaml` `_qwen3_32b`)

Megatron's own TE layer spec (`model.spec: default`), existing options only: GQA (`--group-query-attention
--num-query-groups 8 --kv-channels 128`), `--qk-layernorm`, SwiGLU, RMSNorm, RoPE 1e6, untied embeddings; Qwen3's
tokenizer (`Qwen/Qwen3-32B`, config and tokenizer files); random weights; FineWeb packed as the GLM-5 entries. TP2
(sequence parallel) x PP2 x data parallel, no recompute, 8192 tokens per micro-batch, 16 micro-batches per data-parallel
rank; NVFP4 keeps the last 4 of 64 layers in BF16. Megatron's packed-sequence batches already go through TP and CP
(`get_batch`), so QW-CP can add `--context-parallel-size`.

1 node, 8 layers, 50 steps (job 1270443, trig0152): the loss goes down in all three, labels right (`workload.precision`
bf16 / mxfp8 / nvfp4, `bf16_layers_last: 1`):

| | BF16 | MXFP8 | NVFP4 |
|---|---|---|---|
| loss step 1 / 50, mean of 41-50 | 12.95 / 7.680, 7.742 | 12.95 / 7.645, 7.709 | 12.95 / 7.684, 7.739 |
| step time median | 1.170 s | 0.949 s | 0.960 s |
| peak allocated stage 0 / 1 | 17.0 / 16.5 GiB | 19.4 / 19.1 GiB | 17.0 / 17.0 GiB |

Full depth, 20 steps (rank 0's step rows, steps 3-20; jobs 1270575 and 1270744, 1 node, trig0152 and trig0121; 1270732,
4 nodes, trig[0124,0140,0143,0152]):

| | BF16 | MXFP8 | NVFP4 | NVFP4, no BF16 layer |
|---|---|---|---|---|
| 1 node, 8192 tokens: step, tokens/s per GPU | 8.044 s, 4,074 | 5.263 s, 6,226 (1.53x) | 5.119 s, 6,401 (1.57x) | 4.796 s, 6,832 (1.68x) |
| 1 node, peak stage 0 / 1 | 115.0 / 87.8 GiB | 129.4 / 106.2 | 112.2 / 90.0 | 112.2 / 90.4 |
| 4 nodes, 8192 tokens: step, tokens/s per GPU | 8.359 s, 3,920 | 5.715 s, 5,734 (1.46x) | 5.468 s, 5,994 (1.53x) | |
| 4 nodes, peak stage 0 / 1 | 97.8 / 70.6 GiB | 106.6 / 83.3 | 95.0 / 72.9 | |
| 1 node, 4096 tokens: step, tokens/s per GPU | 3.930 s, 4,169 | 3.440 s, 4,763 (1.14x) | 4.459 s, 3,674 (0.88x) | |
| 1 node 4096, peak stage 0 / 1 | 84.3 / 70.6 GiB | 102.9 / 91.4 | 87.0 / 75.3 | |

NVFP4 over MXFP8 per step: 1.03x (1 node), 1.05x (4 nodes) at 8192 tokens, 1.10x without the BF16 layers, 0.77x at 4096:
below the GEMM ratio at the shapes each GPU runs: TP2 halves the GEMMs' width, so a GPU's 8192-token GEMMs do the work of
the full-width 4096-token ones, 1.31x NVFP4 over MXFP8 per layer (`nvfp4_check.py qwen3`; 1.41x at full width, TP1); its
4096-token GEMMs are half that again, not measured. GPU telemetry over the training window (8 GPUs
of the 1-node runs, nodemon `gpu.tsv` and `gpm.tsv`): busy 82.5 % BF16, 75.1 % MXFP8, 75.3 % NVFP4, 64.7 % NVFP4 at
4096; tensor-pipe activity 38.6 / 25.5 / 16.0 % (8.7 % NVFP4 at 4096). The GPUs wait more as the GEMMs get faster: TP2
halves each GPU's GEMMs, NVFP4 issues the most kernels per GEMM (quantization, Hadamard transform), and with PP2 the
last stage, which holds the BF16 layers and the BF16 output layer, sets the pace (no BF16 layer: 6.3 % faster). The
GEMM share of a BF16 step is about 60-65 % (estimated: tensor activity 38.6 % at the about 58 % of peak these GEMMs reach;
32 layers x about 9 ms of a 473 ms micro-batch slot), not the 75-85 % estimated before measuring. Not measured: a
profile, a layout without PP (TP2 x data parallel fits from 4 nodes, an estimate), CUDA graphs.

### Without the pipeline, with CUDA graphs (jobs 1271063, 4 nodes trig[0122-0123,0133,0145]; 1271160, 1 node trig0152)

`megatron-qwen3-32b-n4-tp2[-cg]-*`: TP2 x data parallel 16, 128 x 8192 tokens per step (8 micro-batches per
data-parallel rank); `-n1-tp2-cg-*`: TP2 x data parallel 4, 32 sequences; `-cg`: `--cuda-graph-impl local` (128 graphs,
64 layers forward and backward, built in 48 s). Step time median over steps 3-17 (5-17 with graphs; 18-19 are profiled):

| | BF16 | MXFP8 | NVFP4 | NVFP4 / MXFP8 |
|---|---|---|---|---|
| 4 nodes, TP2 x DP16: step, tokens/s per GPU | 8.341 s, 3,929 | 5.894 s, 5,560 (1.42x) | 4.968 s, 6,596 (1.68x) | 1.19x |
| the same with CUDA graphs | 7.109 s, 4,609 | 5.371 s, 6,101 (1.32x) | 4.464 s, 7,341 (1.59x) | 1.20x |
| peak allocated, without / with graphs | 131.6 / 125.9 GiB | 155.0 / 151.2 | 136.5 / 130.7 | |
| 1 node, TP2 x DP4, CUDA graphs | 6.440 s, 5,088 | out of memory | 3.828 s, 8,560 (1.68x) | |

(Speed-ups against the BF16 of the same row.) 1-node MXFP8 with graphs stops at its first optimizer step:
`torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 500.00 MiB ... 167.48 GiB is allocated by PyTorch, with
53.91 GiB allocated in private pools (e.g., CUDA Graphs)`; it was not tried without graphs. With graphs the peaks come in
the first step, before the optimizer state exists, so they are the same on 1 and 4 nodes (allocated after it: 84.1 GiB
on 1 node against 66.9 on 4, BF16).

Without the pipeline NVFP4 over MXFP8 per step is 1.19x, close to the GEMMs' 1.31x at these shapes; CUDA graphs speed
every recipe up by 9 to 17 % and leave it at 1.20x. Where the time goes, from Megatron's profiler (rank 0, step 18,
one step; kernel time by class, summed over streams; idle: no kernel running):

| run | window | busy | GEMM | quantize + Hadamard | NCCL | other | attention | kernels |
|---|---|---|---|---|---|---|---|---|
| 1 node PP2 4096 BF16 | 4.14 s | 80.3 % | 50.2 % | 0 | 17.2 % | 14.7 % | 3.4 % | 41,370 |
| 1 node PP2 4096 MXFP8 | 4.22 s | 55.1 % | 24.3 % | 3.0 % | 12.4 % | 13.5 % | 3.0 % | 50,461 |
| 1 node PP2 4096 NVFP4 | 5.83 s | 41.7 % | 9.4 % | 4.8 % | 15.6 % | 9.8 % | 2.2 % | 76,186 |
| 4 nodes TP2 MXFP8 | 5.98 s | 80.5 % | 36.4 % | 3.9 % | 22.8 % | 16.8 % | 5.0 % | 50,102 |
| 4 nodes TP2 NVFP4 | 6.27 s | 69.8 % | 21.2 % | 7.4 % | 24.0 % | 15.6 % | 4.6 % | 74,222 |
| 4 nodes TP2 BF16, graphs | 7.10 s | 96.8 % | 58.4 % | 0 | 24.3 % | 15.0 % | 4.5 % | 44,702 |
| 4 nodes TP2 NVFP4, graphs | 4.53 s | 92.0 % | 29.6 % | 10.6 % | 28.9 % | 21.9 % | 6.4 % | 85,504 |
| 1 node TP2 BF16, graphs | 6.50 s | 97.3 % | 64.5 % | 0 | 16.8 % | 17.0 % | 5.0 % | 45,145 |
| 1 node TP2 NVFP4, graphs | 4.00 s | 89.9 % | 33.9 % | 12.0 % | 17.5 % | 25.5 % | 7.2 % | 85,947 |

The idle time is launch gaps: the gaps between kernels of 10 us to 1 ms make most of it (3.2 of 3.4 s at 4096 NVFP4,
1.7 of 1.9 s on 4 nodes), and NVFP4 issues 1.8-1.9x the kernels of BF16 per step (quantization, Hadamard transform,
scale swizzles around each GEMM). The pipeline adds waits on stage 0 (NCCL SendRecv 0.55 s at 4096 NVFP4 against 0.21
in BF16) for the last stage, which holds the BF16 layers and the output layer. With graphs the GPUs are 90-97 % busy and
the remaining limits are TP2's sequence-parallel all-gather and reduce-scatter (17-29 %, little of it overlapped with
compute), elementwise kernels (15-26 %) and, in NVFP4, quantization (11-12 %); the GEMMs are a third of the NVFP4 step
and two thirds of the BF16 one (measured GEMM share of a BF16 step: 58-65 % with graphs, 50 % at 4096 tokens with PP2). The 4-node TP2 BF16 trace without graphs
was overwritten by a concurrent job's (fixed in 18fc60af); its step times are not affected.

NVFP4 at 0.88x BF16 (1 node, PP2, 4096 tokens; profiled again in 1271160: 4.365 against 4.141 s): the GPU does the
work faster (GEMMs 0.55 s against 2.08 s per step) and waits for the host for 58 % of the step. Checked: the recipe is
applied (cuBLASLt trace of `-s4k-nvfp4-lt`, a stage-0 rank: all 18,432 GEMM calls in E2M1 with E4M3 scales per 16, 32
layers x 4 modules x 3 GEMMs x 16 micro-batches x 3 steps); no memory pressure (87 GiB of 178); no fallback (the GEMMs
are cuBLASLt's block-scaled FP4 kernels); no re-quantization of the weights per micro-batch (Megatron sets
`is_first_microbatch` on the first micro-batch of each step; quantization kernels 0.28 s per step). Each GPU's work per
kernel is small there (TP2 halves the GEMMs, 4096 tokens), and NVFP4's extra kernels per GEMM make the host the limit.

TP comm overlap (`--tp-comm-overlap`, TE userbuffers; `-tpo` entries, job 1271416, the 4-node TP2 x DP16 layout with
CUDA graphs): BF16 6.948 s (2.3 % faster than without it), MXFP8 5.328 s (0.8 %); NVFP4 stops at its first forward,
`KeyError: 'UB for qkv_fprop with use_fp8=True is not registered.'`: Megatron registers quantized userbuffers only for
`--fp8-format` (`initialize.py` `_initialize_tp_communicators`), TE asks for them under NVFP4. The speed layout is CUDA
graphs without it, for all three recipes.

## Proposed rows

GLM-FP4-01, GLM-5 C-speed at 13 nodes: `scripts/trillium/glm-fp4/job.sbatch exp:C-speed
megatron-glm5-tri-pp13-ep8-n13-gbs256 megatron-glm5-tri-pp13-ep8-n13-gbs256-mxfp8
megatron-glm5-tri-pp13-ep8-n13-gbs256-nvfp4` (BF16 first: it compiles the kernels), 13 nodes, `--time=01:45:00` (the
campaign's 1242058: 2,062 s for the first entry, 1,559-1,658 s for each next one). Venv and data under
`$SCRATCH/trillium/new/glm-fp4` (staged), run data in its `runs`. First lines: `[glm-fp4] job=... ulimit -c 0`, `===
megatron-glm5-tri-pp13-ep8-n13-gbs256`, Megatron's arguments, the first step about 15 min in (918 s in 1242058), then
BF16 46-51 s per step (48.61 s, sd 1.02), MXFP8 about 46 s (45.97), NVFP4 about 46 s (est.: the 4-node test had it at
MXFP8's); peak allocated BF16 about 90, MXFP8 132.3, NVFP4 about 100 GiB (est.). Cancel: no first step 30 min into an
entry (Megatron's timeout), steps above 60 s after step 3 (a capped or throttled node: nodemon), NaN loss.

Qwen3 32B C-loss, four 1-node jobs (they can run at once): `job.sbatch exp:C-loss megatron-qwen3-32b-loss-bf16-n1`, then
`-loss-bf16-rep2-n1`, `-loss-mxfp8-n1`, `-loss-nvfp4-n1`. TP2 x PP2 x data parallel 2 without CUDA graphs (MXFP8 does not
fit one node with them), 32 x 8192 tokens per step, 500 steps, seed 1234, lr 3e-4, warm-up 50, cosine to 3e-5, 600,000
FineWeb documents at stride 4 (`stage-loss-data.sh`). Times from the 20-step speed runs: BF16 and its repeat 67 min of
steps each (8.04 s), `--time=01:40:00`; MXFP8 44 min (5.26 s) and NVFP4 43 min (5.12 s), `--time=01:10:00` (max_duration
5,400 s). First lines: `[glm-fp4] job=...`, `=== megatron-qwen3-32b-loss-...`, Megatron's arguments, the first step
about 1 min in (28 s in the speed runs); loss at step 1 about 12.95 (the 8-layer test: 12.95, then 7.7 by step 50). Peak
allocated: BF16 115, MXFP8 129, NVFP4 112 GiB (the speed runs). Cancel: no first step 15 min into a run, NaN loss, steps
above 1.5x those times. Checked on `megatron-qwen3-32b-loss-check-n1` (the NVFP4 entry for 20 steps, warm-up 5; job
1271699, trig0145, on the 200,000 documents then staged): passed, `run.experiment` from `exp:`, `precision` nvfp4, loss
12.94 at step 1, 27.4 at step 3 (the 5-step warm-up to 3e-4), 8.98 at step 20, 4.8-5.5 s per step; `data_loading`:
stride 4, 5,120 slots from 13,815 documents (2.7 per slot), highest index 20,477, no wrap. 500 steps read about 128,000
slots up to about index 512,000. Staged on tri-dm1 at 22:30 (`stage-loss-data.sh`, about 1 min): 600,000 documents,
415.9 M GPT-2 tokens, 3 parts, the Arrow cache built; the check again on them (job 1271942, trig0145): passed,
`docs_available` 600,000, the same 13,815 documents read (the first 200,000 are unchanged), no wrap.

Qwen3 32B C-speed at 13 nodes, one job: `job.sbatch exp:C-speed megatron-qwen3-32b-n13-bf16 megatron-qwen3-32b-n13-mxfp8
megatron-qwen3-32b-n13-nvfp4`, `--time=00:45:00`. TP2 x data parallel 52, CUDA graphs, 416 x 8192 tokens per step, 20
steps. First lines per entry: Megatron's arguments, `Creating 128 CUDA graphs`, `built 128 cuda graph(s) in` about 48 s,
the first step at about 85 s; then (est. from 4 nodes, the larger data-parallel reduce added) BF16 7.1-7.8 s, MXFP8
5.4-5.9 s, NVFP4 4.5-4.9 s per step; peaks below the 4-node ones (the optimizer state over 52), at most about 151 GiB
(MXFP8). Cancel: no first step 15 min into an entry, steps above 2x those, NaN.
