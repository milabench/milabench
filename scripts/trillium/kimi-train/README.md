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

### Stages and memory at depth (4 nodes, PP4 x EP8, BF16, review of ed08aaa6)

| job | entry | result |
|---|---|---|
| 1261381, 1261433, 1261501, 1261567 | `k3-pp4-l6(-gpipe)`: stage 0 = embedding + dense layer 0, no MoE layer | step 1 gradient norm differs by stage (1.27 on stage 0, 0.00 elsewhere; GPipe: not finite), then no step 2 and `CUDA error: Invalid access of peer GPU memory over nvlink` on rank 24 or 16 (trig0152, trig0148, trig0145: three nodes), with both AttnRes transports |
| 1261618 | `k3-pp4-l7` (an MoE layer on every stage), 1F1B, 16 micro-batches | 5.90 s per step, 5,550 tokens/s per GPU; allocated / reserved GiB: stage 0 77.0 / 108.6, stages 1-2 (2 MoE layers, 3 / 2 in flight) 106.3 / 158.0 and 107.6 / 158.6, stage 3 (1 MoE + output) 73.0 / 107.1; reserved grows ~31 GiB after step 1 (Adam state) |
| 1261657 | `k3-pp4-l7-gpipe`, 64 micro-batches of 8192 in flight | step 1 reserved 159-161 GiB on stages 1-3, then allocator failures (`memory mapping failed`) and no step 2 in 4 min: out of memory |
| 1261745 | `k3-pp4-l7-gpipe-4k`, 64 of 4096 in flight | step 1 reserved 141 GiB on stages 1-2, then the same failures on ranks 8 and 16, no step 2 in 4 min: out of memory |

| 1261880 | `k3-pp4-l7-gpipe-gc`: + `garbage_collection_threshold:0.8`, `per_process_memory_fraction:0.99` (on top of the pack's `expandable_segments:True`) | step-1 peaks identical to 1261657 to 0.01 GiB, no step 2: no effect |
| 1261917 | Muon: DistMuon + AdamW (upstream's Kimi K3 recipe; option removed after the test, commit 3b0c87f3) | step 1 reserved 164.5-166.8 GiB on stages 1-3 (+5 against AdamW), then a worker exited (its output is filtered: local rank 0 only) |
| 1261943 | Muon + the allocator setting | the same step-1 peaks, no step 2 in 8 min |
| 1262141 | `k3-pp4-l7-gpipe-noexp`: native allocator (`expandable_segments:False`) | step 1 reserved 162-166 GiB; then on rank 24, in FSDP's `finalize_backward` (gradient dtype cast in `foreach_reduce`): "Tried to allocate 6.89 GiB. GPU 0 has a total capacity of 178.34 GiB of which 3.36 GiB is free ... 81.34 GiB is allocated by PyTorch, and 86.45 GiB is reserved by PyTorch but unallocated" |
| 1262143 | `k3-pp4-l7-noexp`: the same, 1F1B | allocated as with expandable segments (106.3 / 107.4 GiB on stages 1-2), reserved 161.4 / 161.5 (+3), stage 0 125.4 (+17); 5.93 s per step |
| 1262015 | `k3-pp4-l5-gpipe`: one MoE layer per stage, 64 of 8192 in flight | step 1 reserved 104.7 / 106.4 GiB on stages 1-2; stage 3 (output and loss, 64 in flight, which PP48's last stage never holds) hit the limit and held the run at step 1 |
| 1262121 | `k3-pp4-l5`: the same split, 1F1B, 16 micro-batches | 4.76 s per step; stages 1-2: step 1 84.6 / 82.3 GiB, after step 2 98.1 / 95.3 reserved (62.0 / 59.4 allocated); stage 0 77.0 / 108.7, stage 3 73.3 / 107.1 |

One MoE layer per stage with PP48's worst in-flight load: ~120 GiB reserved after step 2 (1262121's steady state + the
+22 GiB that 64 micro-batches in flight add at step 1 in 1262015), stage 0 (embedding + layers 0-1) ~138 GiB. Hence
`k3-pp48-l49` (49 layers on 48 nodes) and `k3-pp40-l41`; `k3-pp48-full` and `k3-pp40-l78` do not fit.

The limit is fragmentation, not memory: with expandable segments the failures report the driver's free memory at
~18 MiB of 178.3 GiB while ~35-50 GiB of torch's reserved memory is unallocated, and the native allocator fails with
81 GiB allocated and 86 GiB reserved but unallocated, at a 6.89 GiB request (one MoE layer's expert weights in BF16
per GPU) while FSDP casts gradients for the reduction (FP32 by default, on a one-rank group for the experts).
`TORCH_NCCL_AVOID_RECORD_STREAMS` is not read by this torch build (not among its `TORCH_NCCL_*` variables).

So every stage needs an MoE layer (the scale entries set `--pp-first/last-stage-less-layers 0`; which collective
diverges is not established), and a stage of 2 MoE layers does not hold the in-flight activations of PP48's worst
stages (13-24, ~35 micro-batches with 3-4 AttnRes blocks), neither at 8192 nor at 4096 tokens per micro-batch.

Upstream fix needed at this pin: K3's pipeline stage does not set FSDP's manual backward finalization (job 1261124,
`configs._patch_kimi_pp_stage`).

## Untested

`k3-pp48-l49` and `k3-pp40-l41` only dry-run; their memory is composed from two 4-node runs, not measured at depth.
Not measured: why reserved exceeds allocated by ~35-50 GiB
on MoE stages; the step-time cost of 4096-token micro-batches; the P2P rate at 40-48 nodes; startup at 48 nodes.
