# QW-CP: Qwen3 32B long-context pretraining with context parallelism (Trillium B200)

`ops/QW_CP_BRIEF.md`. The Qwen3 32B dense entry of GLM-FP4 (`scripts/trillium/glm-fp4/README.md`, "Qwen3 32B dense":
Megatron's TE layer spec, random weights, FineWeb, BF16) with the sequence split over GPUs (`--context-parallel-size`).
Entries `megatron-qwen3-32b-cp-*` in `benchmarks/megatron/dev.yaml` (`_qwcp`).

Cluster directory `$SCRATCH/trillium/new/qw-cp` (`Q`): checkout `src` (branch `trillium-new`; its `Megatron-LM` copied
from GLM-FP4's checkout, 4b4acac with slime's patches; milabench's CLI venv `src/.venv`), runs in `runs`, node monitor in
`nodemon`, Slurm logs in `slurm`. Read only, from GLM-FP4 (`$SCRATCH/trillium/new/glm-fp4`): the venv `venv/torch` (TE
2.18.0, cuDNN 9.20, no flash-attn) and the data `data` (Qwen3's config and tokenizer, 600,000 FineWeb documents); the
venv's benchmate is GLM-FP4's checkout (editable install). Hugging Face datasets may write lock files beside the
FineWeb Arrow cache in GLM-FP4's `data`, as GLM-FP4's own runs do.

## How to run

`job.sbatch ENTRY...` (the submit line is at its top): one `milabench run` per entry, in order, on the nodes of the job.
`exp:NAME` sets `run.experiment` for the entries after it. Results per run: `$Q/runs/qw-cp.<job>.<entry>`.
Step 1 ran `megatron-qwen3-32b-cp-t8-*` (8 layers, about 100 s each, 30 s to fail) and `megatron-qwen3-32b-cp-n1-*` (full
depth, 20 steps: 4.5-7.5 min each, 10-14 min at 128k; the 128k loss run 22 min), 1 node per job, as in the tables below.

## The input

The 8k entries pack 8 slots of 1,024 tokens per sequence (`docs_per_sample: 8`) in Megatron's THD input: attention runs
inside each 1,024-token slot. Here one slot per sequence (`data.docs_per_sample: 1`, FineWeb documents joined with EOS):
one causal mask over the whole 32k / 64k / 128k sequence, attention's cost growing with its square. In Megatron's sbhd
input: `--dataloader-inter-document-masking` off, and `main.py`'s dataset then leaves out `cu_seqlens` (TP rank 0 would
keep them and run THD while the other TP ranks run sbhd). TE 2.18 refuses THD with `a2a+p2p` (`attn_forward_func_with_cp`:
"thd format is not supported with hierarchical CP implementation yet"); with p2p, THD and sbhd give the same loss at
every step with one slot per sequence (job 1272968, below). Slots start 256 documents apart (`doc_stride`; a 128k slot holds
about 190 documents). RoPE as the 8k entries (base 1e6): Qwen3's YaRN for 128k changes the frequencies, not the compute.
Real long-context data has long documents; these are short ones under one causal mask (no document masking).

32,768 tokens per GPU per step at every length, as the 8k entries: on 8 GPUs 8 x 32k, 4 x 64k or 2 x 128k sequences.

## FLOPs per token, causal attention counted

Megatron's `num_floating_point_operations` (what `main.py` reports as `mfu_inputs.flops_per_token`), per token, forward
and backward: `6 x (64 x (h (q + 2 kv) + q h + 3 h f) + h V) + 6 x 64 x q x L`, with h 5,120, q 8,192 (64 heads of 128),
kv 1,024 (8 groups), f 25,600 (SwiGLU: 3 GEMMs), V 151,936, L the length of the causal mask; the attention term is
`3 (forward and backward) x 2 (multiply-add) x 2 (QK^T and PV) x q x L / 2 (causal)` per layer. Recompute is not counted.
MFU against 2.25 PFLOP/s per GPU (dense BF16).

| causal mask | linear | attention | per token | attention share |
|---|---|---|---|---|
| 1,024 (the 8k entries' slots) | 191.9 G | 3.2 G | 195.1 G | 1.7 % |
| 8,192 | 191.9 G | 25.8 G | 217.7 G | 11.8 % |
| 32,768 | 191.9 G | 103.1 G | 295.0 G | 34.9 % |
| 65,536 | 191.9 G | 206.2 G | 398.1 G | 51.8 % |
| 131,072 | 191.9 G | 412.3 G | 604.2 G | 68.2 % |

The 8k references (4,609 tokens/s per GPU on 4 nodes, 4,580 on 13, CUDA graphs) are 40.0 / 39.7 % MFU with their
1,024-token slots.

## Step 1: CP options on one node, 8 layers (job 1272968, trig0145, 2026-11-09 01:57-02:13; 1273064, 1273173)

`megatron-qwen3-32b-cp-t8-*`: 8 layers, 10 steps, TP2 (sequence parallel) x CP4 unless named, 32,768 tokens per GPU per
step. Step time: rank 0's step rows, median of steps 3-10. Peak allocated: all 8 ranks.

| entry | runs | step | tokens/s per GPU | MFU | peak allocated |
|---|---|---|---|---|---|
| 32k p2p | yes | 1.784 s | 18,370 | 33.4 % | 25.3 GiB |
| 32k p2p, THD input | yes | 1.920 s | 17,068 | 31.1 % | 25.4 |
| 32k a2a | yes | 1.759 s | 18,634 | 33.9 % | 25.8 |
| 32k a2a+p2p (2 x 2) | yes | 1.811 s | 18,098 | 32.9 % | 25.8 |
| 32k all_gather (job 1273064) | yes | 1.859 s | 17,626 | 32.1 % | 25.3 |
| 32k TP1 x CP8, p2p | yes | 1.906 s | 17,196 | 31.3 % | 35.4 |
| 128k p2p, no recompute | yes | 2.620 s | 12,505 | 44.2 % | 59.2 |
| 32k p2p, CUDA graphs | yes | 1.458 s | 22,480 | 40.9 % | 21.5 |
| 32k a2a, CUDA graphs | yes | 1.453 s | 22,548 | 41.0 % | 22.0 |
| 128k p2p, full recompute, full-iteration graph | step 1, then fails | | | | |

- Same loss: the six TP2 x CP4 runs (p2p on sbhd and THD input, a2a, a2a+p2p, p2p and a2a with graphs) agree at every
  step within 2e-4 (12.853 at step 1, 11.265 at step 10, learning rate 1e-6).
- Attention backend: TE picks FusedAttention (cuDNN, sub-backend 1) in every run (`NVTE_DEBUG`); flash-attn is not
  installed, and TE's unfused attention does not support CP.
- `allgather`: Megatron's help lists it, TE 2.18 refuses it ("Unsupported communication type: allgather!", job 1272968)
  and calls it `all_gather` (as Megatron's `transformer_config.py` does); the entry passes `all_gather`.
- CUDA graphs (`--cuda-graph-impl local`, 16 graphs) run with p2p and a2a: 18 % faster, same loss at every step. Megatron
  allows full recompute only with a full-iteration graph (`--cuda-graph-impl full_iteration`, NaN check off); with CP it
  stops at step 2: `RuntimeError: The size of tensor a (32768) must match the size of tensor b (131072)` in
  `full_cuda_graph.py`'s static data loader (likely: `get_batch` splits the batch dict in place for CP, so the static
  buffer kept after step 1 is the CP shard; not checked further).
- Memory on 8 layers: 25.3 GiB at 32k, 59.2 at 128k: about 14 GiB that does not grow with the sequence, plus 11.3 GiB per
  8k tokens on a CP rank, which is 0.84 GiB per layer (as GLM-FP4's 8k runs) and about 4.6 GiB of logits.
- No CP (`-t8-32k-cp1`, TP2 x data parallel 4, the same 8 sequences per step; job 1273173): the same loss as the CP runs
  at every step, within 2e-4.

## Step 1: full depth, one node (jobs 1273064 trig0152, 1273065 trig0121, 1273068 and 1273173 trig0145, 1273195 trig0152)

`megatron-qwen3-32b-cp-n1-*`: 64 layers, BF16, 20 steps (the 128k p2p point is the 50-step loss run: learning rate
3e-4 instead of 1e-6, the same work per step), 32,768 tokens per GPU per step. Step time: rank 0's step rows, median of
steps 3-20 (5-20 with CUDA graphs). MFU with the FLOPs above (recompute not counted). Power cap: share of the 8 GPUs'
samples over the same steps with the software power cap (`clocks_event_reasons` 0x4), and their mean power.

| length, layout | recompute | step | tokens/s per GPU | MFU | peak allocated | power cap, mean power |
|---|---|---|---|---|---|---|
| 8k (one 8k mask), TP2 x DP4 | none | 8.176 s | 4,008 | 38.8 % | 148.8 GiB | 78 %, 881 W |
| 32k, TP2 x CP4 p2p | none | 12.123 s | 2,703 | 35.4 % | 148.8 | 80 %, 867 W |
| 32k, a2a | none | 12.024 s | 2,725 | 35.7 % | 152.8 | 78 %, 868 W |
| 32k, p2p, CUDA graphs | none | 10.337 s | 3,170 | 41.6 % | 125.9 | 99 %, 970 W |
| 64k, p2p | none | out of memory in the first forward (172.8 GiB allocated) | | | | |
| 64k, p2p | selective (MLP) | 14.774 s | 2,218 | 39.2 % | 133.5 | 98 %, 972 W |
| 64k, p2p, CUDA graphs | selective (MLP) | out of memory building the graphs (171.6 GiB) | | | | |
| 128k, p2p | selective (MLP) | step 1 (160.2 GiB), out of memory in step 2 (Adam's moments) | | | | |
| 128k, p2p (loss run, 50 steps) | full | 24.252 s | 1,351 | 36.3 % | 108.6 | 99 %, 979 W |
| 128k, a2a | full | 25.964 s | 1,262 | 33.9 % | 108.6 | 99 %, 974 W |
| TP1 x CP8, 32k and 128k | full | out of memory before step 1: 122 GiB allocated, then the 61 GiB gradient buffer (32.8 B parameters per GPU) | | | | |
| 8k (one 8k mask), TP2 x DP4, CUDA graphs | none | 7.089 s | 4,622 | 44.7 % | 125.9 | 98 %, 966 W |
| 64k, p2p | full | 17.558 s | 1,866 | 33.0 % | 96.3 | 98 %, 957 W |
| 128k, p2p | full on 48 of 64 layers (block) | 36.712 s (32.0-41.3) | 893 | 24.0 % | 167.2 | 65 %, 722 W |

- Recompute each length needs on 8 GPUs at TP2 x CP4: none at 32k, selective (MLP) at 64k (16 % faster than full there),
  full at 128k. 128k with 16 layers left out of the recompute is slower than full and irregular, at 167 GiB allocated and
  171.5 reserved of 178: likely the allocator near the limit (not checked).
- p2p against a2a: 32k within 1 % (a2a 12.024 s, p2p 12.123); at 128k p2p is 7 % faster (24.25 against 25.96 s).
- CUDA graphs: 15 % at 32k (10.337 against 12.123 s), 13 % at 8k; not with full recompute (above), and not at 64k with
  selective (out of memory while building the 128 graphs).
- Against 8k: per GPU, 32k keeps 0.69x of 8k's tokens/s (with graphs, both one causal mask), 64k 0.55x, 128k 0.34x
  (without graphs, against 8k without graphs); against GLM-FP4's 8k references with 1,024-token slots (4,609 tokens/s
  per GPU on 4 nodes, 5,088 on 1, graphs) 128k keeps 0.29x / 0.27x. MFU falls less: 44.7 % at 8k, 41.6 % at 32k, 39.2 %
  at 64k, 36.3 % at 128k (recompute not counted).
- Power: with graphs, and from 64k on, the GPUs sit at the software power cap in 98-99 % of the samples at 957-979 W;
  8k and 32k without graphs 78-80 %, 867-881 W. These step times are throttled. No Xid in any run.
- Estimated before the runs: 32k 12.9 s and 142 GiB (measured 12.123 s, 148.8), 64k selective 17.5 s and 140 GiB (14.774,
  133.5), 128k full 26-29 s and about 120 GiB (24.25, 108.6); 64k without recompute and 128k selective out of memory
  (both were). TP1 x CP8's 153 GiB of weights, gradients and optimizer state was estimated to fit at 32k with full
  recompute; it does not: the gradient buffer is allocated while 122 GiB are already in use.

## 128k loss, 50 steps (job 1273065, trig0121)

`megatron-qwen3-32b-cp-n1-128k-loss`: TP2 x CP4 p2p, full recompute, 2 x 128k tokens per step, seed 1234, learning rate
3e-4 after a 5-step warm-up, cosine to 3e-5. Loss (rank 0, the last stage; averaged over the CP group): 12.948 at step 1,
24.91 at step 3 (the warm-up spike, as GLM-FP4's 8k check: 27.4), 11.21 at 10, 8.27 at 20, 7.83 at 30, 7.74 at 40, 7.64
at 50; mean of steps 41-50 7.753. No NaN; the highest gradient norm 1,412 (warm-up). Data: 100 sequences of about 193
documents each, none read past the stride, highest document index 25,506 of 600,000.

## Not run, untested

- CP across nodes, a2a+p2p across nodes, profiles (step 2); MXFP8 and NVFP4 with CP.
- all_gather and a2a+p2p at full depth (8 layers only).
- The THD input with a2a and all_gather (p2p only); the 8k entries' THD input with 8 slots under CP is not used: with
  `--dataloader-inter-document-masking` Megatron splits the whole sequence into 2 x CP chunks while TE's THD path expects
  each slot split (read from the code, not run).
- Why the full-iteration graph and the 48-layer block recompute fail or slow down (the causes above are readings).
- `--cross-entropy-loss-fusion` and activation offloading (memory options not tried).

## Step 2, proposed (128k, BF16, 32,768 tokens per GPU per step, 12 steps, one profiled step per layout)

| job | layout | recompute | why |
|---|---|---|---|
| 2 nodes, 4 x 128k | TP2 x CP8 p2p (CP across nodes) | full | against CP4 x DP2 |
| | TP2 x CP8 a2a+p2p, sizes 4 x 2 (a2a in the node, p2p across) | full | against p2p |
| | TP2 x CP4 x DP2 p2p (CP in the node) | full | the reference |
| | TP2 x CP8 p2p | selective | 16k tokens per rank: est. about 120 GiB, what CP across nodes saves in recompute |
| 4 nodes, 8 x 128k | TP2 x CP16 p2p, a2a+p2p 4 x 4, CP4 x DP4 | full | as on 2 nodes |
| | TP2 x CP16 p2p | none, CUDA graphs | 8k tokens per rank, the 32k CP4 shapes, optimizer over 16: est. 125-150 GiB |
| 1 node | TP2 x CP4 p2p, MXFP8 | full | does FP8's gain shrink when attention is 68 % of the FLOPs |

Estimates: full recompute 24-26 s per step on 2 and 4 nodes if the KV exchange overlaps attention (the same work per GPU
as on one node); selective at CP8 about 20 s, no recompute with graphs at CP16 about 17 s. Bytes, p2p at CP8 over 2 nodes:
per layer and micro-batch each CP rank passes on 7 KV chunks of 16k tokens x 2 KiB (4 local KV groups x 128 x K and V,
BF16) = 224 MiB forward, about twice that backward (KV and dKV), again for the recomputed forward; 2 of the 8 hops of each
ring cross nodes. To check against the profiles.
