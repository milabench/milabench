# Kimi K3 pre-training (TorchTitan) on Trillium B200

Brief: `~/projects/tri-work/ops/K3_TRAINING_BRIEF.md`. Upstream TorchTitan's `kimi_k3` model as is (the commit the
torchtitan_native pack pins, 51b37c2): KDA (attn-gym kernels), gated MLA, LatentMoE (896 experts, top-16, 2 shared),
Attention Residuals, text only, random init. Entries: `config/torchtitan_native_kimi.yaml` (`--model kimi_k3`).

| step | command | what |
|---|---|---|
| 2 | `k3_train.sh --nodelist <node> k3-1n-l4` (`-bf16`) | 1 node, 4 layers (KDA dense, KDA-MoE x2, MLA-MoE), EP8: loss on FineWeb |
| 3 | `k3_train.sh --nodelist <2 nodes> k3-pp2-l4` (`-bf16`) | PP2 x EP8, 2 layers per stage, 16 micro-batches: memory per stage, pipeline hop |
| 3b | `k3_train.sh --nodelist <4 nodes> k3-pp4-l7-gpipe-rbf16` (`k3-pp4-l7-rbf16`: 1F1B) | PP4 GPipe, 64 micro-batches in flight on 2-MoE-layer stages: stand-in for PP48's worst stages |
| 4 | `k3_train.sh --nodelist <48 nodes> k3-pp48-full-rbf16` (`k3-pp40-l78-rbf16`, `k3-pp48-l49`) | one copy, 93 layers (40 nodes: 78; fallback: 49 layers): `REQUESTS.md` K3-TRN-01..03 |

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

Pipeline hop (1261179, step 20 traced with `--profile-freq`; NCCL kernels per CUDA stream read from the trace by
`hop_times.py`, removed after commit 9d792a39): every micro-batch sends 2 tensors of 117.4 MB (hidden state and
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
| 1262207 | `k3-pp4-l7-gpipe-rbf16`: BF16 gradient reduction (`--grad-reduce bf16`) | reaches step 13, 22.2 s per step (5,900 tokens/s per GPU); allocated / reserved GiB: stages 1-2 (2 MoE layers, 64 in flight) 110.1 / 141.9 and 111.3 / 143.3, stage 0 84.1 / 108.8, stage 3 (output, 64 in flight) 147.2 / 160.6; no allocator failure |
| 1262230 | `k3-pp4-l7-rbf16`: the same, 1F1B | 5.55 s per step (5.90 with the FP32 reduction); stages 1-2 90.6 / 109.9 and 91.3 / 111.2 (FP32: 106.3 / 158.0, 107.6 / 158.6); losses as the FP32 reduction's to 0.003 at step 13 |
| 1262015 | `k3-pp4-l5-gpipe`: one MoE layer per stage, 64 of 8192 in flight | step 1 reserved 104.7 / 106.4 GiB on stages 1-2; stage 3 (output and loss, 64 in flight, which PP48's last stage never holds) hit the limit and held the run at step 1 |
| 1262121 | `k3-pp4-l5`: the same split, 1F1B, 16 micro-batches | 4.76 s per step; stages 1-2: step 1 84.6 / 82.3 GiB, after step 2 98.1 / 95.3 reserved (62.0 / 59.4 allocated); stage 0 77.0 / 108.7, stage 3 73.3 / 107.1 |

One MoE layer per stage with PP48's worst in-flight load: ~120 GiB reserved after step 2 (1262121's steady state + the
+22 GiB that 64 micro-batches in flight add at step 1 in 1262015), stage 0 (embedding + layers 0-1) ~138 GiB. Hence
`k3-pp48-l49` (49 layers on 48 nodes); 2 layers per stage do not fit with the FP32 reduction.

The limit is fragmentation, not memory: with expandable segments the failures report the driver's free memory at
~18 MiB of 178.3 GiB while ~35-50 GiB of torch's reserved memory is unallocated, and the native allocator fails with
81 GiB allocated and 86 GiB reserved but unallocated, at a 6.89 GiB request (one MoE layer's expert weights in BF16
per GPU) while FSDP casts gradients for the reduction (FP32 by default, on a one-rank group for the experts).
`TORCH_NCCL_AVOID_RECORD_STREAMS` is not read by this torch build (not among its `TORCH_NCCL_*` variables).
With the BF16 reduction (no cast), the 2-MoE-layer stages hold the stand-in's load with ~22 GiB to spare, hence
`k3-pp48-full-rbf16` (93 layers, 48 nodes) and `k3-pp40-l78-rbf16` (78 layers, 40 nodes). On Qwen the campaign measured the same option at +2% (13 nodes) and +0.2%
(50 nodes) tokens/s per GPU and 5-6 GiB less memory.

So every stage needs an MoE layer (the scale entries set `--pp-first/last-stage-less-layers 0`; which collective
diverges is not established), and with the FP32 reduction a stage of 2 MoE layers does not hold the in-flight activations of
PP48's worst stages, neither at 8192 nor at 4096 tokens per micro-batch; with the BF16 reduction it does (above).

### Before step 1 at depth (K3-TRN-01, job 1262377, 48 nodes)

All 48 stages came up; before step 1 torch's pipeline stage infers the stage shapes one stage after another (a real
forward on each, then a backward in reverse, `recv_object_list` between them). Stages 29-47 timed out in that
receive after 600 s: TorchTitan gives `--init-timeout-seconds` to the default group only, the mesh groups (the
pipeline's included) keep torch's 10-min NCCL default until its `set_pg_timeouts` lowers the mesh groups and the
default group to the train timeout after step 1. `configs._patch_init_timeouts` (K3 only) calls that same function
with the init timeout when training starts. The pipeline's 2-rank P2P child groups are created during step 1 and copy
their parent's timeout then (torch `distributed/pipelining/_p2p.py`): they get the init timeout and keep it for the
whole run, as the lowering after step 1 does not reach them (job 1261501, at 600 s: lowered to 0:03:00 at 18:59:02,
rank 0 timed out in a P2P wait after 600 s at 19:09:40). A stage that dies after step 1 can hold the others for up to
the init timeout (1 h): see the cancel rule below. Check (job 1262556, `k3-pp4-l7-rbf16-t30`, 30 s): local rank 0 of
each node logs "adjusting timeout for all ProcessGroups to 0:00:30", and stage 2 fails 39 s after training starts on
its 2-rank pipeline communicator ("[TC][rank=1] ... Operation timed out after 30661 ms") while stages 0-1 infer their
shapes.

Per-stage shape inference (4 nodes, node telemetry: the active stage runs ~1 core per rank with its GPUs idle, the
waiting ones sit in NCCL receives):

| job | setting | forward inference per stage, s (0 / 1 / 2 / 3) |
|---|---|---|
| 1262230, 1262576 | compiled, expandable segments (the rows) | 10 / 20 / 25 / 15, 11 / 25 / 25 / 22; backward ~5 each |
| 1262562 | `--compile eager` | 10 / 20 / 26 / 14 (not torch.compile: the caches are warm, 7 hits) |
| 1262143, 1262643 | native allocator | 9 / 10 / 15 / 12, 10 / 16 / 9 / 16 |

The 2-MoE-layer stages spend ~10 s more with expandable segments (largely a phase with the GPUs at ~0.88 while the
first large blocks are mapped, inferred); ~9-12 s of single-core work per stage remains with either allocator, not
identified. The native allocator would shorten the 48-stage chain but reserves more: 155.8 GiB on the stand-in's worst
2-MoE-layer stage (job 1262643, BF16 reduction) against 143.3 with expandable segments, so the rows keep expandable
segments and a 60-min init timeout. Static stage shapes would skip the inference, but TorchTitan builds K3's
`AttnResPipelineStage` without them, and the first micro-batch would still meet each cold stage in turn.

### Running the scale rows (K3-TRN-01..03)

Step 1 is expected about 25-30 min after the job starts (the shape-inference chain, est.); cancel if there is none by
45 min. After step 1, cancel if there is no new step for 5 min: the P2P groups keep the 60-min init timeout, so a stage
that dies would otherwise hold every node until it expires. The rows log every step (`--log-freq 1`); K3-TRN-04..06
(job 1263259) logged steps 1, 10 and 20 only, so there a step line comes ~10 min after step 1 (est.).

Upstream fix needed at this pin: K3's pipeline stage does not set FSDP's manual backward finalization (job 1261124,
`configs._patch_kimi_pp_stage`).

### Last stage at depth (K3-TRN-04, job 1262896, 48 nodes)

Every stage came up and the shape inference passed; stage 47 (trig0152: layer 92 + output, 19.6-19.9 GiB of model at
init like the other one-layer stages, 26.7 on two-layer ones) then ran out of memory before step 1: rank 376 logged
10 "memory mapping failed" between 23:55:15 and 23:55:56 with 4-12 MiB free of 178.3 GiB, and at 23:56:01 another
worker of that node exited with code 1. Its error is not in the log: local ranks 1-7 print nothing
(`--local-ranks-filter=0`), and torchrun's summary of the failure (rank, exit code, the traceback TorchTitan's `@record`
writes) was lost because benchrun's worker launcher dropped torch's `start_new_session`, so the agent's shutdown raised
`ProcessLookupError` on its first `killpg` (reproduced on CPU with the K3 venv; with the fix the summary prints the
failing rank's exception). Node telemetry records no GPU memory; trig0152's GPUs computed steadily from 23:54:32 to
23:55:14 (~42 s: about step 1's 128 micro-batches at the stand-in's ~0.29 s per micro-batch, est.) and were nearly idle
from the first failure on; GPU 6 went idle at ~23:55:58 while the others spun at full utilization (rank 382 the one
that exited: inferred). So the limit came at about the end of step 1's micro-batches on stage 47 (inferred from the
timing); what was allocated, by what, is not recorded.

The stand-ins' last stage received a 1-block AttnRes stack (74.7 GiB reserved, 62.0 allocated, job 1262576); layer
92 receives 8 blocks. `k3-pp4-l8-depth` measures that stage at depth: 8 layers in 1-layer AttnRes blocks
(`--attn-res-block-size 1`), split so that stage 3 is layer 7 (MLA + MoE, as layer 92) + output, receiving 7 blocks and
holding 8 after its own, with PP48's 128 micro-batches. `--pp-memory-log` prints the last stage's memory around each
micro-batch; `--memory-snapshot-freq` dumps TorchTitan's CUDA memory snapshot every N steps and, through the pack's
allocator observer, at the first out-of-memory (TorchTitan's own exit dump never fires at this pin: its `train()`
closes the profiler with `__exit__(None, None, None)`, so job 1263416 left no snapshot).

Result (job 1263416, last stage = rank 24, GiB): allocated at the start of each micro-batch's forward grows by 0.875
per micro-batch (8.2 at micro-batch 0, 142.1 after the backward of micro-batch 127), each micro-batch adds a
transient of ~20 on top (running peak 162.0), step 1 reserved 169.2 (stages 0-2: 98.6-102.4); it all drops before
the next step (22.7 at step 2's first forward, +14.5 of optimizer state and kept gradients), and step 2 ran out of
memory at micro-batch 120 ("Tried to allocate 896.00 MiB", all 8 ranks of the stage, printed by torchrun's summary).
0.875 GiB is the stage's received inputs, hidden + 7 blocks of [8192, 7168] BF16 (PP48's stage 47: hidden + 8, 1.0
GiB, 126 GiB over 128 micro-batches; K3-TRN-06's: hidden + 4, 0.55, ~70). The holder: torch's schedule keeps every
micro-batch's loss tensor until the step ends (`_internal_losses`, then TorchTitan's `losses` list, cleared after the
schedule step), and a live loss keeps its autograd graph, whose AccumulateGrad nodes keep the received leaves after
the backward. Reproduced on CPU (2-rank gloo, torch's `Schedule1F1B`, the bench venv's torch): received inputs alive
after each backward 1, 2, ..., 8 and 8 after the step; with `--pp-detach-losses` (the schedule keeps a detached loss
once the backward has its own reference) 1 each time and 0 after the step, the losses' values unchanged. Middle stages
drop their outputs after the backward, hence no growth there. The loss and logits (TorchTitan's chunked loss, 8
chunks: ~0.1 GiB per micro-batch at the loss) are not what fills the stage.

`k3-pp4-l8-depth-m64ga2` (job 1263463: 64 micro-batches per schedule step, 2 per optimizer step) drops the retention
between the two: last stage 126.8 GiB reserved at step 1, 139.8 at step 2 (127.6 allocated). `k3-pp4-l8-depth-detach`
measures the fix at 128 micro-batches.

## Untested

`k3-pp48-full-rbf16`, `k3-pp40-l78-rbf16` and `k3-pp48-l49` only dry-run; their memory comes from 4-node stand-ins, not
measured at depth. The review estimates the worst PP48 stages at the block boundaries (stages 6, 12, 18, 24, 30, where
the second layer also keeps the concatenated AttnRes stack): 144-157 GiB reserved, within 165 GiB but closer than the
stand-in's 143.3; PP40 about 1 GiB above the stand-in (est.). The shape-inference chain at 48 stages after the timeout
fix: ~21-26 min (est., from 21 s per stage at 48 and the 4-node backward).
Not measured: why reserved exceeds allocated by ~35-50 GiB
on MoE stages; the step-time cost of 4096-token micro-batches; the P2P rate at 40-48 nodes; startup at 48 nodes.
