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

`--optimizer muon` imports upstream's Kimi K3 recipe (`torchtitan_recipes.tests.models.kimi_k3`), whose multimodal
imports need torchvision, which the pack does not install: add the nightly of the torch build, into the K3 venv only:

```bash
uv pip install --python output/venv/torchtitan_native/bin/python --no-deps \
    --index-url https://download.pytorch.org/whl/nightly/cu130 "torchvision==0.30.0.dev20261005+cu130"
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

### Stages and memory at depth (4 nodes, PP4 x EP8, BF16, review of ed08aaa6)

| job | entry | result |
|---|---|---|
| 1261381, 1261433, 1261501, 1261567 | `k3-pp4-l6(-gpipe)`: stage 0 = embedding + dense layer 0, no MoE layer | step 1 gradient norm differs by stage (1.27 on stage 0, 0.00 elsewhere; GPipe: not finite), then no step 2 and `CUDA error: Invalid access of peer GPU memory over nvlink` on rank 24 or 16 (trig0152, trig0148, trig0145: three nodes), with both AttnRes transports |
| 1261618 | `k3-pp4-l7` (an MoE layer on every stage), 1F1B, 16 micro-batches | 5.90 s per step, 5,550 tokens/s per GPU; allocated / reserved GiB: stage 0 77.0 / 108.6, stages 1-2 (2 MoE layers, 3 / 2 in flight) 106.3 / 158.0 and 107.6 / 158.6, stage 3 (1 MoE + output) 73.0 / 107.1; reserved grows ~31 GiB after step 1 (Adam state) |
| 1261657 | `k3-pp4-l7-gpipe`, 64 micro-batches of 8192 in flight | step 1 reserved 159-161 GiB on stages 1-3, then allocator failures (`memory mapping failed`) and no step 2 in 4 min: out of memory |
| 1261745 | `k3-pp4-l7-gpipe-4k`, 64 of 4096 in flight | step 1 reserved 141 GiB on stages 1-2, then the same failures on ranks 8 and 16, no step 2 in 4 min: out of memory |

So every stage needs an MoE layer (the scale entries set `--pp-first/last-stage-less-layers 0`; which collective
diverges is not established), and a stage of 2 MoE layers does not hold the in-flight activations of PP48's worst
stages (13-24, ~35 micro-batches with 3-4 AttnRes blocks), neither at 8192 nor at 4096 tokens per micro-batch.

Upstream fix needed at this pin: K3's pipeline stage does not set FSDP's manual backward finalization (job 1261124,
`configs._patch_kimi_pp_stage`).

## Untested

PP48 / PP40 as entered do not fit (above). Not measured: the step-time cost of 4096-token micro-batches; why
reserved exceeds allocated by ~50 GiB on MoE stages (fragmentation; allocated stays ~107 GiB at 2-3 in flight); the
P2P rate at 40-48 nodes; startup at 48 nodes.
