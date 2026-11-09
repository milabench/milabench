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

Below a few thousand tokens per expert the times hardly change with the token count (Qwen3-30B-A3B: 2.1 / 3.3 / 4.3 ms
BF16 / MXFP8 / NVFP4 from 128 to 2048): the GPU waits for the host to issue the per-expert GEMMs and quantization
kernels, and NVFP4 issues the most. In a training step part of that issue time can overlap other GPU work, which this
microbenchmark does not show. Kimi K3 at EP8 (112 experts per GPU): NVFP4 stops in TE's grouped Hadamard transform
(`group_hadamard_transform.cu:408`, `num_tensors <= kMaxTensorsPerKernel`); BF16 and MXFP8 run (8192 tokens per
expert: fc1 97.4 / 60.7 ms, fc2 50.0 / 32.5 ms).
