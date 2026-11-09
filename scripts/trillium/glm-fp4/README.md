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

Not measured by the check: `q_up` and the dense fc1 (LayerNormLinear: TE's RMSNorm compiles at run time and found no
CUDA headers; milabench sets them, the standalone check did not), and memory (its "kept" column is wrong: the previous
recipe's buffers are freed after its baseline reading).

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
