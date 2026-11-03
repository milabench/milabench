# Trillium B200: experiment design

Large-scale LLM pre-training on Trillium's B200 partition: what the experiments are, which question each
answers, which figure each feeds, and what they cost. The Rorqual material (`scripts/analysis/`) is a
separate campaign; nothing here is meant to reproduce it.

Status 2026-11-02: design, before access. Every number marked *(est.)* is an estimate, not a measurement.

## The machine

| | known | source |
|---|---|---|
| Nodes | 52 (`trig0101`-`trig0152`), exclusive access during the window; **5 of them (trig0148-0152) are reserved for SciNet's inference service (`AI_inference-nodes`): we may get only 47** | SciNet talk 2026-10-21; `/etc/hosts`; `scontrol show reservation` |
| Per node | 8 x B200 180 GB on NVLink, 2 x 48-core Intel, 2 TB RAM, liquid cooled | SciNet talk |
| GPUs | 416 in total | |
| Network | InfiniBand, 800 Gb/s per node, "non-blocking" (user); HCA count unknown | user; SciNet |
| Switch layout | **unknown** (asked to the admins). `/etc/hosts` puts the nodes in 8 subnets of 7, 7, 7, 7, 4, 7, 7, 6 nodes (racks?) | `/etc/hosts` |
| Shared with others | the distributed filesystem and the interconnect (CPU and H100 nodes stay in production) | user |
| Storage | `/scratch`, `/project`: VAST over NFS (65 PB). Nodes are diskless (`TmpDisk=2048` MB): no local NVMe, only /dev/shm in the 2 TB RAM | `df`, `scontrol show node` |
| Slurm | partition `compute_b200_full_node`, 1-day limit; no topology plugin (`scontrol show topology` empty) | `sinfo` |
| Software | `module load cuda/12.9`, `--gpus-per-node=b200:8`; account `rrg-bengioy-ad` (to confirm on trillium-gpu) | SciNet talk; user |
| Matmul (SciNet, 8192^2) | B200: BF16 1,668, FP8 3,041 TFLOP/s; H100: BF16 785 | SciNet talk (their measurement) |
| Window | a few days *(est.)*, date unknown | user |

What sets it apart, and drives the design:

1. **Network per FLOP like Rorqual, but nodes twice as large.** 100 Gb/s per GPU against 1,668 TFLOP/s
   BF16, i.e. ~0.060 Gb/s per TFLOP (Rorqual: 50 / 785 = 0.064). A standard DGX B200 has 4x more network.
   The best way to use this machine keeps traffic inside the 8-GPU NVLink domain.
2. **FP8 / MXFP8 doubles compute, not the network.** Whatever is network-bound in BF16 becomes more so.
3. **73 TB of GPU memory.** Qwen3.8-27B (~430 GB of training state) fits on one node; GLM-5 (744B) fits on
   ~13 nodes *(est.)*: up to 4 model copies on the machine. Activation checkpointing can be reduced.
4. **Topology unknown, fabric shared.** Placement is not a factor of the design unless D2 shows a structure;
   a reference run repeated through the window separates our configs from background load.

## Workloads

| | Qwen3.8-27B | GLM-5 (744B MoE, 256 experts, top-8) |
|---|---|---|
| Framework | TorchTitan (`benchmarks/torchtitan_native`) | Megatron-LM (`benchmarks/megatron`) |
| Parallelism available today | FSDP, HSDP (`dp_replicate`), TP (`tp`, not with the `rows` layout; **at most 4**: 4 KV heads) | EP, PP, DP; **TP = 1 and CP = 1** (the plugin's RoPE); EP is carved from each stage's data-parallel ranks: 8N % PP = 0 and 8N / PP % EP = 0 |
| To add | PP, MXFP8, gradient-reduction dtype, TP with the `rows` layout, checkpointing; fix the pushed loss and the token count under PP | 8 GPUs per node, checkpoint save / load, FP8 / MXFP8 entries, B200 layouts, loss from the last pipeline stage |
| Precision recipes | `bf16`, `mxfp8` (the pinned TorchTitan removed Float8 tensorwise / rowwise on 2026-10-01) | `bf16`, `fp8-tensorwise`, `fp8-blockwise` (runs as MXFP8 GEMMs on B200), `mxfp8` |
| Sequence | 4096 x 3 per GPU (12,288 tokens) by default | 8192, micro-batch 1 |
| Global batch | grows with the GPUs (weak scaling) | **256 sequences per model copy** (weak scaling); 64 per copy as a small-batch point (8 micro-batches against 13 stages: ~60% bubble *(est.)*) |
| Training | everything trains | everything trains (no frozen experts: memory allows it) |

Network arithmetic for Qwen *(est.)*: any data-parallel layout (FSDP, HSDP, TP + FSDP) moves ~8 bytes per
parameter per node per step (BF16 gathers forward + backward, FP32 gradient reduction): ~215 GB, so
**>= 2.2 s per step at 800 Gb/s**, against ~3 s of compute at 12,288 tokens per GPU and ~50% of 1,668
TFLOP/s. Levers that change the ratio: tokens per GPU, BF16 reduction (~6 B/param), PP across nodes
(volume / stages). HSDP vs FSDP changes when the traffic happens, not its volume.

GLM-5 layout that matches the machine: PP13 x EP8 = 104 GPUs = **13 nodes per copy, 4 copies on 52
nodes** (78 layers split 7 / 11 x 6 / 5: the first stage holds the dense layers, the last the loss; the
plugin's DSA index sharing is turned off, as on Rorqual, since it only allows some stage boundaries) (memory: Rorqual's full run peaked at 66.6 GiB x 200 GPUs ~ 13 TiB; 13 B200
nodes have ~17 TiB *(est.)*).

## Experiments

Groups run in this order. IDs are used by the launch scripts (`run.variant`) and the figures.

### D. Health and network (first: decides whether placement matters)

| ID | what | scale | est. time |
|---|---|---|---|
| D1 | Per-node health check (srygaard tool): per-GPU matmul BF16 / FP8, NVLink all-reduce, per-HCA loopback bandwidth, GPU-HCA affinity | every node | 30 min |
| D2 | Pairwise sweep: latency and bandwidth for every node pair, 26 disjoint pairs per round, 51 rounds (all 1,326 pairs) | 52 nodes | 15 min |
| D3 | NCCL collectives vs scale: all-reduce, all-gather, reduce-scatter, all-to-all at 1, 2, 4, 8, 13, 26, 52 nodes, 1 GiB messages (+ a size sweep at 2 and 52 nodes) | 1-52 | 45 min |
| D4 | Bisection: the two halves (26 + 26, random) all-to-all with each other, 3 random cuts | 52 | 15 min |

If D2 shows groups (latency or bandwidth clusters), add **P**: Qwen and GLM at 13 nodes inside one group vs
spread over groups. Otherwise placement is dropped.

### E. Jobs side by side (second: decides whether runs can be packed)

| ID | what | est. time |
|---|---|---|
| E1 | Qwen best guess (HSDP-8, 24k tokens/GPU) on 4 disjoint random sets of 13 nodes: each set alone, then all 4 together; 2 repetitions with new sets | 1.5 h |
| R  | Reference run: Qwen on 4 fixed nodes, FSDP, 12,288 tokens/GPU, 10 min; **every ~3 h through the window** | 10 min each |

If E1 shows no slowdown (< 2%), runs of <= 26 nodes are packed in parallel for A, B, C, F.

### A. Parallel layout (core of the design)

**A-Q, Qwen** (5-min measurement after 30 warm-up steps, as on Rorqual):

| factor | levels |
|---|---|
| layout | `fsdp` (all GPUs) / `hsdp8` (shard in a node, replicate across) / `hsdp16` / `tp2-fsdp` / `tp4-fsdp` / `pp2-fsdp` / `pp4-fsdp` |
| tokens per GPU | 12k (4096 x 3) / 24k (4096 x 6) / 48k (4096 x 12) |
| gradient reduction | fp32 / bf16 (on the 2 best layouts) |
| activation checkpointing | selective / full / none (1 node and 13 nodes; none only where it fits) |
| nodes | 13 (full cross); 2 and 52 for the 3 best; 1 for the non-PP layouts |

~45 runs x 2 repetitions x ~10 min *(est.)*.

**A-G, GLM-5** (20 iterations, first 5 skipped; startup ~15 min at scale *(est.)*), global batch 256 per
copy. A model copy is PP x EP GPUs; EP can only use the GPUs of one stage, so the smallest run differs:

| layout | EP group | smallest run (copies) | at 52 nodes |
|---|---|---|---|
| `pp13-ep8` | in a node (NVLink) | 13 nodes (1) | 4 copies |
| `pp13-ep16` | across 2 nodes | 26 nodes (1) | 2 copies |
| `pp13-ep32` | across 4 nodes | 52 nodes (1) | 1 copy |
| `pp26-ep8` | in a node | 26 nodes (1) | 2 copies |
| `pp6-ep16` (13 layers per stage) | across 2 nodes | 12 nodes (1) | 48 nodes, 4 copies (416 GPUs don't split in 6) |

The EP comparison (8 / 16 / 32) is made at 52 nodes; `pp13-ep8` and `pp6-ep16` also at 13 / 12 nodes. Plus
recompute full vs selective, micro-batch 1 vs 2, and global batch 64 per copy on the best layout.
~14 runs x 2 repetitions x ~25 min *(est.)*.

### B. Scaling (with the best layouts from A)

| ID | what |
|---|---|
| B-Q-weak | Qwen, 1, 2, 4, 8, 13, 26, 52 nodes, best layout and `fsdp` baseline, fixed tokens per GPU |
| B-Q-strong | Qwen, fixed global batch (52 x 8 x 24k tokens), 13, 26, 52 nodes |
| B-G | GLM-5 `pp13-ep8`, 1, 2, 3, 4 copies (13, 26, 39, 52 nodes), batch 256 per copy (weak) and 256 in total (strong) |

### C. Precision

| ID | what |
|---|---|
| C-speed | Qwen: BF16 / MXFP8; GLM: BF16 / FP8 tensorwise / FP8 blockwise / MXFP8; best layout, at 1 node (Qwen) or 13 nodes (GLM) and at 52 nodes |
| C-loss | same recipes, 500 steps from the same seed and data (FineWeb, a realistic learning rate) at 13 nodes: loss difference to BF16 |

### F. Operations

| ID | what |
|---|---|
| F1 | Checkpoint save and load: Qwen (~0.38 TB) at 1, 13, 52 nodes; GLM-5 weights only (~1.5 TB) and full (~10.4 TB) at 13 and 52 nodes; synchronous vs asynchronous (time training is blocked vs total write); 3 repetitions at different times of day. Escalation with the admins' agreement: Qwen 1 node -> Qwen 52 -> GLM weights -> GLM full. **Only one GLM-5 checkpoint on disk at a time**, deleted after the load is timed. No local disk: staging only in host memory (asynchronous mode) |
| F2 | Long run: best Qwen layout on 52 nodes, >= 1 h (stability, stragglers, clocks, power) |
| F3 | Startup: time to an initialized model and to the first step, from every run (no extra runs) |
| F4 | Energy: tokens per kWh from GPU power (nodemon), from every run (no extra runs) |

### Budget *(est.)*

| group | serial | packed (if E1 holds) |
|---|---|---|
| bring-up on B200 (install, 1-node smoke of both workloads, FP8 / MXFP8 smoke) | 4 h | |
| D | 2 h | |
| E + R | 2.5 h | |
| A-Q | 15 h | 6 h |
| A-G | 12 h | 6 h |
| B | 8 h | 5 h |
| C | 6 h | 4 h |
| F | 6 h | |
| total | ~55 h | ~35 h |

Priorities if the window is short: **tier 0** bring-up, D1; **tier 1** A-Q at 13 nodes, B-Q-weak, A-G at
13 and 52 nodes, B-G; **tier 2** D2-D4, E1, C, F1; **tier 3** F2, B-Q-strong, the rest of A.

## Figures

Figure IDs are the names in `analysis/tri_analysis/figures.py` and `report/generated/figures/`.

| figure | question | x / y (panels) | experiments |
|---|---|---|---|
| `health_gpus` | Is every B200 alike? | per-GPU matmul BF16 and FP8, by node (strip), with the median and a -5% line | D1 |
| `health_links` | Is every NVLink and HCA alike? | per-node NVLink all-reduce bus bandwidth; per-HCA bandwidth | D1 |
| `pair_matrix` | Does the fabric have a structure? | 52 x 52 heatmap of pairwise bandwidth (and latency), nodes ordered by subnet | D2 |
| `collectives_scale` | Does bandwidth hold up to 52 nodes? | bus bandwidth vs nodes, one line per collective, 800 Gb/s line; bisection points | D3, D4 |
| `interference` | Do jobs slow each other? | tokens/s per GPU, each 13-node set alone vs together | E1 |
| `reference_drift` | Does the background load move our numbers? | reference throughput vs time of the window | R |
| `qwen_layouts` | Which layout, at which batch? | tokens/s per GPU (or MFU) by layout (rows) x tokens per GPU (columns), panels 2 / 13 / 52 nodes | A-Q |
| `qwen_levers` | What buys the most: batch, reduction dtype, PP, AC? | throughput change vs the `fsdp` 12k reference at 13 nodes | A-Q |
| `step_decomposition` | How much of the step is network? | step = compute (1-node) + exposed communication, by layout and scale | A-Q, B-Q |
| `glm_layouts` | Expert parallelism inside or across nodes? | tokens/s per GPU and all-to-all time by layout, at 13 and 52 nodes | A-G |
| `qwen_scaling` | How far does Qwen scale? | tokens/s per GPU and efficiency vs GPUs (8-416), best layout vs `fsdp`; strong scaling panel | B-Q |
| `glm_scaling` | Do model copies scale? | tokens/s per GPU vs copies (1-4), weak vs strong batch | B-G |
| `precision_speed` | What does FP8 / MXFP8 buy, and does it hold at scale? | speedup over BF16 by recipe, 1 node vs 52 nodes, both workloads | C-speed |
| `precision_loss` | Does it train the same? | loss difference to BF16 vs step, by recipe | C-loss |
| `checkpoint` | What does a checkpoint cost? | save / load time and blocked time vs nodes, sync vs async, full vs weights only; write throughput | F1 |
| `stability` | Does throughput hold over an hour? | tokens/s per step and GPU clocks / power vs time | F2 |
| `stragglers` | Are some GPUs slower? | per-GPU activity, per-node CPU clock (nodemon) | F2 |
| `startup` | How long until training? | time to the first step vs nodes, both workloads | F3 |
| `energy` | What does a token cost in energy? | tokens per kWh by layout, precision and scale | F4 |

## Data

Formats: `scripts/analysis/FORMAT.md` (`milabench.pretrain.v1`, health checks, nodemon, per-rank
metrics), plus the Trillium additions in `FORMAT.md` here (layout and precision fields, loss per step,
checkpoint events, collective sweeps). Raw data: `$SCRATCH/trillium/` on Trillium, copied read-only to
`analysis/data/raw/`. Synthetic data stands in for every experiment not run yet and is drawn hollow and
labelled as such in the figure itself.
