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
Step 2 ran `megatron-qwen3-32b-cp-n{1,2,4}-*` at 128k (12 steps, profiled: 7-11 min each), one job per node count;
Megatron's profiler writes rank 0's trace to `$Q/prof/<name>/torch_profile`.

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

## Step 2: 128k across 2 and 4 nodes (jobs 1273313 trig[0140-0143], 1273314 trig[0136-0137], 1273316 trig0145; 2026-11-09 03:12-04:07)

`megatron-qwen3-32b-cp-n{1,2,4}-*`, BF16 (one MXFP8 point), 32,768 tokens per GPU per step: global batch 2 / 4 / 8 x 128k
on 1 / 2 / 4 nodes, micro-batch 1. CP across nodes (CP8, CP16, data parallel 1): one micro-batch per CP group per step,
4 / 8 of them. CP in the node (CP4 x DP2, x DP4): 2 micro-batches per data-parallel rank, as on one node. 12 steps,
Megatron's profiler on rank 0 at steps 10-11 (it records step 11); step time: rank 0's median of steps 3-9 (5-9 with CUDA
graphs). Bytes across nodes per node and step: NCCL's 4 ports' tx counters (node monitor, steps 3-9; 0.0 on the 1-node runs,
so storage does not use them) against the count from the layout (below). Power cap: share of the GPUs' samples with the
software power cap (0x4); mean power; SM clock median (lowest) over the nodes' samples.

| nodes, layout | recompute | step | tokens/s per GPU | MFU | peak allocated | across nodes, GiB per node and step: measured / counted | power cap, mean power, SM clock |
|---|---|---|---|---|---|---|---|
| 1, TP2 x CP4 p2p | full | 24.999 s | 1,311 | 35.2 % | 108.6 GiB | 0 / 0 | 99.4 %, 976 W, 1,447 (1,275) MHz |
| 1, TP2 x CP4 p2p, MXFP8 | full | 22.557 s | 1,453 | 39.0 % | 146.2 | 0 / 0 | 98.0 %, 973 W, 1,455 (1,282) |
| 2, CP8 p2p | full | 27.349 s | 1,198 | 32.2 % | 91.5 | 574 / 555 | 96.0 %, 958 W, 1,466 (1,320) |
| 2, CP8 a2a+p2p (4 x 2) | full | 28.124 s | 1,165 | 31.3 % | 91.5 | 365 / 363 | 97.0 %, 969 W, 1,534 (1,357) |
| 2, CP4 x DP2 | full | 25.844 s | 1,268 | 34.0 % | 97.1 | 108 / 107 | 97.5 %, 968 W, 1,432 (1,162) |
| 2, CP8 p2p | selective (MLP) | 23.130 s | 1,417 | 38.0 % | 122.1 | 451 / 443 | 93.1 %, 950 W, 1,462 (1,312) |
| 2, CP4 x DP2 | selective (MLP) | out of memory in step 2 (162.4 GiB allocated, 9.3 more asked) | | | | | |
| 4, CP16 p2p | full | 36.354 s | 901 | 24.2 % | 91.5 | 1,095 / 1,074 | 79.5 %, 857 W, 1,627 (1,402) |
| 4, CP16 a2a+p2p (4 x 4) | full | 32.543 s | 1,007 | 27.0 % | 91.5 | 890 / 882 | 86.5 %, 886 W, 1,537 (1,372) |
| 4, CP4 x DP4 | full | 25.953 s | 1,263 | 33.9 % | 91.5 | 117 / 114 | 99.5 %, 965 W, 1,432 (1,117) |
| 4, CP16 p2p, CUDA graphs | none | 25.252 s | 1,298 | 34.8 % | 125.9 | 838 / 834 | 96.0 %, 951 W, 1,612 (1,417) |
| 4, CP4 x DP4 | selective (MLP) | 34.373 s (31.6-35.0) | 953 | 25.6 % | 165.9 | 118 / 114 | 61 %, 714 W, 1,470 (1,185) |
| 4, CP16 p2p | none | 27.783 s | 1,179 | 31.7 % | 131.6 | 850 / 834 | 82 %, 884 W, 1,620 (1,410) |
| 4, CP8 x DP2 p2p (job 1273445) | selective (MLP) | 23.236 s | 1,410 | 37.9 % | 116.4 | 448 / 450 | 93 %, 947 W, 1,459 (1,305) |

The 1-node rows are this job's 12-step profiled runs (24.999 s; the 50-step loss run on trig0121: 24.25 s), so the
multi-node rows compare with them.

Bytes counted from the layout, per node and step. K/V chunk per token on a GPU: 2 KiB (TP2: 4 KV groups x 128, K and V,
BF16). TE's p2p sends a rank's chunk on (CP - 1) hops per layer forward, and in backward K/V then K/V with dK/dV then dK/dV
(2 (CP - 1) chunks): 4 (CP - 1) chunks per layer and micro-batch with full recompute (the forward twice), 3 (CP - 1)
without recompute of attention (selective recomputes the MLP only). In a p2p ring of CP ranks over n nodes, n of the CP
sends of each hop cross a node; in a2a+p2p (a2a over the node's 4 CP ranks, a p2p ring of n across) every p2p send
crosses, with the same chunk size per GPU and n - 1 hops. Gradients: the distributed optimizer's reduce-scatter of the BF16 gradients and all-gather of the BF16
parameters, 30.5 GiB each per GPU (16.38 B parameters per TP rank), over the data-parallel x CP group (8 GPUs on 2 nodes,
16 on 4), counted as rings: (N - 1) / N x 30.5 GiB per node per collective and TP rank. CP8 p2p, 2 nodes: 2 rings (TP) x
64 layers x 4 micro-batches x 2 crossings x 28 chunks of 32 MiB = 896 GiB, 448 per node, plus 107 of gradients. Rank 0's
profile agrees with the per-rank count: 96 / 224 / 480 GiB of K/V sent per step at CP4 / CP8 / CP16 (`record_param_comms`),
30.5 GiB in the gradient reduce-scatter. The measured bytes are within 3.5 % of the count in every layout: 0.5-3.5 % above,
0.6 % below for CP8 x DP2 on 4 nodes (the counters are read at the node monitor's 1 s samples nearest the window's ends,
scaled to the window).

Where the time goes (rank 0, the profiled step, kernel time by class; "CP exchange": NCCL SendRecv kernels, the p2p rings
and a2a+p2p's all-to-all; "alone": no other kernel running; "no kernel": the GPU idle):

| layout | profiled step | attention | GEMM | CP exchange busy (share under compute) | CP exchange alone | TP all-gather + reduce-scatter alone | no kernel |
|---|---|---|---|---|---|---|---|
| 1 node CP4, full | 25.83 s | 66.3 % | 20.6 % | 20.3 % (97.7 %) | 0.5 % | 3.5 % | 3.5 % |
| 1 node CP4, full, MXFP8 | 22.92 s | 73.7 % | 12.5 % | 20.8 % (98.6 %) | 0.3 % | 3.2 % | 2.7 % |
| 2, CP8 p2p, full | 28.43 s | 61.5 % | 19.6 % | 36.1 % (95.8 %) | 1.5 % | 4.4 % | 5.8 % |
| 2, CP8 a2a+p2p, full | 28.71 s | 56.5 % | 19.1 % | 14.0 % (36.0 %) | 9.0 % | 4.8 % | 3.1 % |
| 2, CP4 x DP2, full | 25.94 s | 66.8 % | 21.6 % | 7.0 % (99.6 %) | 0.0 % | 5.0 % | 0.6 % |
| 2, CP8 p2p, selective | 23.70 s | 59.1 % | 22.5 % | 39.4 % (96.2 %) | 1.5 % | 5.1 % | 4.4 % |
| 4, CP16 p2p, full | 37.74 s | 46.9 % | 14.4 % | 50.2 % (82.0 %) | 9.0 % | 4.5 % | 18.6 % |
| 4, CP16 a2a+p2p, full | 33.27 s | 50.2 % | 17.0 % | 23.5 % (82.7 %) | 4.1 % | 4.8 % | 17.5 % |
| 4, CP4 x DP4, full | 26.06 s | 65.9 % | 21.4 % | 7.1 % (99.3 %) | 0.0 % | 6.1 % | 0.6 % |
| 4, CP16 p2p, none, graphs | 27.15 s | 52.1 % | 15.0 % | 56.2 % (85.6 %) | 8.1 % | 11.2 % | 6.7 % |
| 4, CP8 x DP2 p2p, selective | 23.78 s | 58.6 % | 22.3 % | 40.1 % (94.8 %) | 2.1 % | 5.7 % | 3.8 % |

- Same recompute (full): CP in the node with data parallel across is fastest on 2 and 4 nodes, 25.84 / 25.95 s, 3.4 /
  3.8 % above one node (the gradient reduce-scatter and parameter all-gather at the end of the step: `--overlap-grad-reduce`
  is off in every entry). CP across nodes: CP8 p2p 27.35 s on 2 nodes (+5.8 % against CP4 x DP2); on 4 nodes CP16 p2p
  36.35 s (+40 %) and CP16 a2a+p2p 32.54 s (+25 %).
- Each layout's best fitting recompute: CP across nodes needs less. On 2 nodes CP8 with selective recompute, 23.13 s, is
  10.5 % faster than CP4 x DP2 with full (25.84; CP4 x DP2 with selective runs out of memory in step 2) and faster than one
  node. On 4 nodes CP8 x DP2 with selective recompute, 23.24 s (job 1273445: as CP8 on 2 nodes, 23.13), is the fastest:
  CP16 without recompute and with CUDA graphs takes 25.25 s, CP4 x DP4 with full 25.95.
  CP4 x DP4 with selective fits (165.9 GiB, the optimizer over 16) but takes 34.37 s: 34 % of its profiled step has no
  kernel while the host spends 9.4 s in `cuMemCreate` (1,439 calls), 2.7 s in `cuMemRelease` and 1.4 s in `cuMemMap`: the
  expandable-segments allocator maps and unmaps memory every step near the limit (no such call in the full-recompute or
  CP8 selective profiles). The same likely explains step 1's 128k block recompute (167 GiB, 36.7 s).
- p2p against a2a+p2p: on 2 nodes p2p is faster (27.35 against 28.12 s): a2a+p2p's all-to-all inside the node (216 GiB per
  rank per step) runs mostly alone (9 % of the step), for 256 GiB across per node and step instead of 448. On 4 nodes
  a2a+p2p is faster (32.54 against 36.35 s): every GPU carries 96 GiB across per step instead of 480 on each of the 2 boundary GPUs a
  node has in the p2p rings, and its 4-rank ring does 4 times the attention per hop for the same chunk.
- Overlap: on 1 and 2 nodes the p2p exchange runs under compute (96-98 % of its kernel time, 0.5-1.5 % of the step
  alone). At CP16 over 4 nodes the per-hop attention block is 8k x 8k and no longer hides the chunk: 82 % under compute,
  9 % of the step alone, and 18.6 % of the step with no kernel at all on rank 0 (17.5 % with a2a+p2p, 6.7 % without
  recompute and with graphs).
- Power: the layouts that keep the GPUs busy are capped in 93-100 % of samples at 950-976 W; CP16 without graphs (p2p,
  a2a+p2p, no recompute) in 80-87 % at 857-886 W, with median SM clocks of 1,537-1,627 MHz against 1,432-1,466 for CP4 x DP
  and CP8 p2p: there the network, not the power cap, sets the step. CP4 x DP4 selective: 61 %, 714 W (the allocator). No Xid.
- MXFP8 at 128k (1 node): 22.56 against 25.00 s, 1.11x (8k: 1.32x at 13 nodes): its GEMMs take 2.87 s instead of 5.32, and
  attentio## The 1-node 128k step: 24.25 s in step 1, 25.0 s in step 2

Not the node: the same entry (`megatron-qwen3-32b-cp-n1-128k-p2p-full`, learning rate 1e-6, 20 steps) took 25.239 s on
trig0121 (job 1273446) and 25.111 s on trig0145 (job 1273448), 0.5 % apart; step 2's profiled run on trig0145 took 24.999
s. Step 1's 24.25 s is the 50-step loss run (job 1273065, trig0121), at learning rate 3e-4: its steps fall from 24.82 s at
step 3 (warm-up) to 24.2-24.3 s from step 5. Two same-node pairs, one run per arm, in separate jobs:

| node | 1e-6 | 3e-4 | 1e-6 slower by | median SM clock, 1e-6 / 3e-4 | mean power, 1e-6 / 3e-4 |
|---|---|---|---|---|---|
| trig0121 | 25.239 s (1273446, steps 3-20) | 24.252 s (1273065, the loss run, steps 3-50) | 4.1 % | 1,432 / 1,485 MHz | 984 / 979 W |
| trig0145 | 25.122 s (1273448, steps 5-12) | 23.859 s (1273463, `-128k-lr3e-4`, the loss settings for 12 steps, steps 5-12) | 5.3 % | 1,447 / 1,507 MHz | 976 / 978 W |

The steps are 4-5 % slower at 1e-6, consistent with lower clocks at the same power (every run is capped in 99-100 % of
samples): a mechanism inferred, not measured per kernel (the values the higher learning rate produces would draw less
power). Every speed entry here, as the campaign's, runs at a constant 1e-6.

5 % above those of training at 3e-4
(one comparison).

## 32k across nodes, and the 13-node row

`megatron-qwen3-32b-cp-n2-32k-cg` (job 1273420, trig[0136-0137], COMPLETED 6:54): the 13-node 32k entry's layout on 2
nodes, TP2 x CP4 x DP2 with CUDA graphs, 16 x 32k per step (8 micro-batches per data-parallel rank): 10.733 s (steps 5-20),
3,053 tokens/s per GPU, 40.0 % MFU, peak 125.9 GiB; 3.8 % above one node (10.337 s), estimated about 11.2 s. 108 GiB per
node and step across nodes (107 counted: gradients only); power cap 94 %, 940 W, SM clock median 1,530 MHz.

At 13 nodes (104 GPUs, 52 TP pairs) CP can only be 1, 2 or 4, so CP stays in the node and data parallel crosses, which on
2 and 4 nodes cost 3.4-3.8 % over one node at 128k (full recompute) and 3.8 % at 32k (graphs). To see whether CP across
nodes holds up with many CP groups crossing at once and data parallel on top, a third entry runs CP8 on 12 of the 13
nodes: `placement: {nodes: leaf, group_nodes: 2}` keeps whole pairs of nodes (one CP8 group each) and leaves the last node
out (an existing option: Slurm has no topology on Trillium, so the main node comes first and the others by name). Checked
on 3 of 4 nodes (`megatron-qwen3-32b-cp-t8-n3of4`, job 1273445: 24 ranks on trig0140-0142, trig0143 idle, 1.697 s per step).

Proposed row: `job.sbatch exp:C-speed megatron-qwen3-32b-cp-n13-32k megatron-qwen3-32b-cp-n13-128k
megatron-qwen3-32b-cp-n12-128k-cp8`, BF16, 20 steps, 32,768 tokens per GPU per step:

| entry | layout | recompute | per step | estimate (from 2 and 4 nodes) | peak |
|---|---|---|---|---|---|
| `-n13-32k` | TP2 x CP4 x DP13, CUDA graphs | none | 104 x 32k | 10.7-11.0 s, 2,980-3,060 tokens/s per GPU | 125.9 GiB |
| `-n13-128k` | TP2 x CP4 x DP13 | full | 26 x 128k | 25.9-26.5 s, 1,240-1,265 | 91-97 GiB |
| `-n12-128k-cp8` | TP2 x CP8 x DP6, 12 nodes | selective (MLP) | 24 x 128k | 23.2-24.0 s, 1,365-1,410 | about 116 GiB |

About 27 min in all (first steps 1-2.5 min in, 128 CUDA graphs built in 60-90 s for 32k). Across nodes per node and step
(est.): about 108-120 GiB for the CP4 entries (gradients), 450-460 for CP8 (336 of K/V and the gradients).
## Not run, untested

- Above 4 nodes; CP across more than 4 nodes; anything but CP4 at 13 nodes (the 52 TP pairs).
- `--overlap-grad-reduce` (the data-parallel reduce runs at the end of the step in every entry) and the TP comm overlap.
- The all_gather and a2a+p2p implementations at full depth on one node; a2a+p2p with selective recompute or CUDA graphs.
- MXFP8 across nodes; NVFP4 with CP.
- The THD input with a2a and all_gather (p2p only); the 8k entries' THD input with 8 slots under CP is not used: with
  `--dataloader-inter-document-masking` Megatron splits the whole sequence into 2 x CP chunks while TE's THD path expects
  each slot split (read from the code, not run).
- Why the full-iteration graph fails; the allocator explanation of the slow near-limit runs is from one profile.
- `--cross-entropy-loss-fusion`, activation offloading, `PYTORCH_ALLOC_CONF` without expandable segments.
