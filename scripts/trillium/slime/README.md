# slime GRPO on Trillium (B200)

GRPO post-training with slime (`benchmarks/slime`, Megatron-LM trainer, SGLang rollouts, Ray) for the GLM-RL
workload (`ops/GLM_RL_BRIEF.md`). Bring-up on a Qwen3-30B-A3B stand-in first, then the GLM-5.3 entries.

Everything lives in `B = $SCRATCH/trillium/new/glm-rl`: `src` (this repository, branch `trillium-new`, with the
milabench CLI venv in `src/.venv`), `venv/torch` (the pack's venv), `data`, `runs`, `slurm` (job logs), `uv-cache`,
`wheelhouse`. Not the campaign's `$SCRATCH/trillium/glm`: the slime entries install into `venv/torch` too.

| step | command (login node) | what |
|------|------------------|------|
| install | `src/scripts/trillium/slime/install.sh` | downloads (below) |
| 1 | `src/scripts/trillium/slime/submit.sh q30-1n 00:50:00 1 slime-qwen3-30b-a3b-grpo -- --nodelist=trig0145` | 1 node: GPU build of TE and torch_memory_saver (first job only), the HF -> torch_dist conversion (once), 6 GRPO rollouts |

`submit.sh NAME TIME NODES ENTRY... [-- sbatch args]` submits one job named `nw-glm-rl-NAME` that runs `job.sbatch`:
power check, `milabench install` (offline), `milabench prepare` on the first node, then one `milabench run` per entry.
The job runs `$B/src` as it is: `git -C $B/src pull --ff-only` first.

## Install

Compute nodes have no internet. `install.sh` (login node) creates the CLI venv, runs the pack's install with
`SLIME_DOWNLOAD_ONLY=1` (everything but TransformerEngine and torch_memory_saver, which need a GPU node), puts TE
2.18.0's three files from PyPI in `wheelhouse`, fetches torch_memory_saver's pinned git commit into the uv cache (a
dry-run install; offline, uv takes a pinned commit from its cache) and downloads the entry's HF checkpoint and
dataset (`milabench prepare` with `SLIME_DOWNLOAD_ONLY=1`). The first job then builds TE (`NVTE_PYTORCH_FORCE_BUILD`,
`NVTE_CUDA_ARCHS=100`) and torch_memory_saver offline (`UV_OFFLINE`, `UV_FIND_LINKS=wheelhouse`), as the campaign's
GLM install did (`scripts/trillium/glm/README.md`, Install).

- Own uv cache, `UV_LINK_MODE=copy`: the pack's `patch_sglang` rewrites sglang's installed files, and with uv's default
  hardlinks it rewrites the cache's files too. The campaign's `venv/torch` shares its inodes with `$SCRATCH/uv-cache`
  (link count 2, 2026-11-07), so the sglang 0.5.21 in that cache is already patched: a venv installed from it
  would refuse `sglang.patch` (it does not apply twice).
- torch_memory_saver is required here (`Slime.memory_saver_required`): `--colocate` turns `--offload-train` on, which
  pauses the trainer's GPU memory with it while SGLang generates. The campaign's GLM venv went without it.

## Entries

`slime-qwen3-30b-a3b-grpo` (`benchmarks/slime/dev.yaml`): slime's `scripts/run-qwen3-30B-A3B.sh` on one node,
colocated: the trainer (TP4 x EP8, DP 2, CPU-offloaded Adam) and one SGLang engine of 8 GPUs take turns on the
GPUs. DAPO-Math-17k with the `deepscaler` rule-based reward, the GLM-5 entry's GRPO flags (score centering,
REINFORCE loss, TIS, rollout routing replay), 32 prompts x 8 samples = 256 samples per rollout, one training step
per rollout, responses capped at 4,096 tokens (the recipe: 131,072), 6 rollouts, no eval.

## Measured

Nothing yet.

## Untested

- More than one node (Ray across nodes through `RayCluster`), the GLM-5.3 entries.
