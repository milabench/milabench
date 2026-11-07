# Kimi K3 pre-training (TorchTitan) on Trillium B200

Brief: `~/projects/tri-work/ops/K3_TRAINING_BRIEF.md`. Upstream TorchTitan's `kimi_k3` model as is (the commit the
torchtitan_native pack pins, 51b37c2): KDA (attn-gym kernels), gated MLA, LatentMoE (896 experts, top-16, 2 shared),
Attention Residuals, text only, random init. Entries: `config/torchtitan_native_kimi.yaml` (`--model kimi_k3`).

| step | command | what |
|---|---|---|
| 2 | `k3_train.sh --nodelist <node> k3-1n-l4` (`-bf16`) | 1 node, 4 layers (KDA dense, KDA-MoE x2, MLA-MoE), EP8: loss on FineWeb |
| 3 | `k3_train.sh --nodelist <2 nodes> k3-pp2-l4` (`-bf16`) | PP2 x EP8, 2 layers per stage, 16 micro-batches: memory per stage, pipeline hop |
| 3b | `k3_train.sh --nodelist <4 nodes> k3-pp4-l6-gpipe` (`-4k`) | PP4 GPipe, 64 micro-batches in flight: PP48's worst in-flight memory |
| hop | `python hop_times.py $K3/runs/<run>/traces/iteration_20` | NCCL kernel times of the traced step, per rank and stream |

`k3_train.sh --dry-run ...` prints the sbatch command. Jobs are named `nw-k3trn-<bench>`, bring-up rules in
`~/projects/tri-work/ops/IMPL_BRIEF.md`.

## Install (login node: it downloads)

```bash
unset PYTHONPATH PIP_CONFIG_FILE; export PATH=$HOME/.local/bin:$PATH UV_CACHE_DIR=$SCRATCH/trillium/new/k3-trn/uv-cache
K3=$SCRATCH/trillium/new/k3-trn; mkdir -p $K3 && cd $K3
git clone -b trillium-new https://github.com/milabench/milabench.git
uv venv --python-preference only-managed --python 3.12 --seed env
VIRTUAL_ENV=$PWD/env uv pip install -e './milabench[cuda]'
export PATH=$PWD/env/bin:$PATH MILABENCH_BASE=$PWD/output MILABENCH_GPU_ARCH=cuda
milabench install --config milabench/config/torchtitan_native_kimi.yaml --select k3-1n-l4
milabench prepare --config milabench/config/torchtitan_native_kimi.yaml --select k3-1n-l4
```

The bench venv is the Qwen pack's (same requirements and torch nightly). `prepare` downloads FineWeb sample-10BT
(500,000 documents) and DeepSeek-V3.1's tokenizer: K3's own is tiktoken (`tiktoken.model` with remote code), which
TorchTitan's `HuggingFaceTokenizer` cannot read; DeepSeek's ids (< 129,280) index K3's 163,840-entry embedding.

## Measured (2026-11-07, trig0145 / trig0146, throttled step times)

Seq 8192, one sequence per GPU per micro-batch, EP8 inside a node, full AC. Memory: largest rank, GiB, peak allocated /
peak reserved (summary `per_rank`). Tokens/s per GPU from the median step; MFU = TorchTitan's FLOPs per token x
tokens/s / 2.25e15 per GPU (the campaign's `PEAK_BF16`, `tri_analysis/machine.py`), as the run summary.

| job | entry | result |
|---|---|---|
| 1261073 | `k3-1n-l4-bf16` (1 node, 3 MoE layers per GPU) | 0.738 s per step, 11,100 tokens/s per GPU, MFU 17.4%; 122.6 / 151.7 GiB; FineWeb loss 12.49 -> 7.20 at step 100 |
| 1261072 | `k3-1n-l4` (FP32 master) | out of memory: model 56.4 GiB, 136 GiB at step 1, then `expandable_segments: memory mapping failed` every ~15-25 s and no step 2; cancelled |
| 1261155 | `k3-pp2-l4` (FP32 master) | stage 1 (2 MoE layers + output): 146.5 GiB at step 1, then the same failures and no step 2 for 4 min; cancelled |
| 1261179 | `k3-pp2-l4-bf16` | 6.625 s per step (16 micro-batches, 1.05 M tokens), 9,890 tokens/s per GPU, 349 TFLOP/s per GPU, MFU 15.5%; stage 0 76.4 / 108.0, stage 1 116.8 / 167.2 GiB; loss 12.50 -> 6.52 at step 45 |

Pipeline hop (1261179, step 20 traced, `hop_times.py`): every micro-batch sends 2 tensors of 117.4 MB (hidden state and
the AttnRes stack, one block this shallow) from each of the 8 GPUs of a node to the next node; a send whose receiver is
waiting takes 4.24-4.96 ms (24-28 GB/s per GPU, ~200 GB/s per node), so ~9.6 ms per micro-batch against ~390 ms of
stage compute. Expert all-to-all (inside the node): 217-229 ms of the 6.9 s step on stage 0 (1 MoE layer), 586-647 ms
on stage 1 (2 MoE layers).

Upstream fix needed at this pin: K3's pipeline stage does not set FSDP's manual backward finalization (job 1261124,
`configs._patch_kimi_pp_stage`).

## Untested

More than 2 stages: micro-batches in flight on early stages (up to 47 at PP48), AttnRes stacks of up to 8 blocks per hop,
the P2P rate at 40-48 nodes (D2 measured a drop from 13-node jobs on), startup at 48 nodes. `k3-pp48-full` and
`k3-pp40-l78` have only been dry-run.
