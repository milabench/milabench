# Trillium B200: experiment design

Large-scale LLM pre-training on Trillium's B200 partition: what the experiments are, which question each
answers, which figure each feeds, and what they cost. The Rorqual material (`scripts/analysis/`) is a
separate campaign; nothing here is meant to reproduce it.

Status 2026-11-03: design, before access, updated with what the code tests on H100 (truncated models, Rorqual
and Fir) established. Every number marked *(est.)* is an estimate, not a measurement; nothing here is measured on B200
except what SciNet measured.

## The machine

| | known | source |
|---|---|---|
| Nodes | 52 (`trig0101`-`trig0152`), **50 usable on 2026-11-04**: `trig0146` is in maintenance (7 of 8 GPUs), `trig0122` excluded (see Health checks). The admins expect all 52 eventually. Reservation `b200-friendly-users` until 2026-12-03, shared with other users | `sinfo`, `scontrol`, admins |
| Per node | 8 x B200, 179.1 GiB each, on NVLink, 1,000 W limit; 2 x 48-core Intel, ~2 TB RAM, liquid cooled | health checks; SciNet talk |
| GPUs | 416 in total | |
| Network | InfiniBand. NCCL uses 4 ports at 400 Gb/s, 1,600 Gb/s per node (`mlx5_0`, `1`, `4`, `9`: site `NCCL_IB_HCA`, port counters agree), not the 800 Gb/s first quoted. GPUs 0, 2, 4, 6 each share a PCIe switch with one port; 1, 3, 5, 7 have none. Measured results below | D3, D2 (jobs 1239634, 1239766, 1239768) |
| Switch layout | **unknown** (asked to the admins). `/etc/hosts` puts the nodes in 8 subnets of 7, 7, 7, 7, 4, 7, 7, 6 nodes (racks?) | `/etc/hosts` |
| Shared with others | the distributed filesystem and the interconnect (CPU and H100 nodes stay in production) | user |
| Storage | `/scratch`, `/project`: VAST over NFS (65 PB), 25 TiB quota on `$SCRATCH`. Nodes are diskless; `/tmp` is ~1.5 TiB of zram, RAM like `/dev/shm`, although Slurm reports `TmpDisk=2048` MB | health checks, `diskusage_report`, `scontrol show node` |
| Slurm | partition `compute_b200_full_node`, 1-day limit; no topology plugin. QOS `normal` stops at 100 GPUs per user (12 nodes); use `bump`. **Jobs above 25 nodes are refused** (removal asked to the admins) | `sinfo`, `sacctmgr`, `sbatch --test-only` |
| Software | Driver 595.91.07, CUDA 13.2 runtime (>= 580, as the cu130 wheels need); `module load cuda/12.9`, `--gpus-per-node=b200:8`; account `rrg-bengioy-ad` | health checks; SciNet talk |
| Matmul (SciNet, 8192^2) | B200: BF16 1,668, FP8 3,041 TFLOP/s; H100: BF16 785 | SciNet talk (their measurement) |
| Window | a few days *(est.)*, date unknown | user |

What sets it apart, and drives the design:

1. **About twice Rorqual's network per FLOP for collectives, nodes twice as large.** 200 Gb/s per GPU against 1,668 TFLOP/s
   BF16, i.e. ~0.12 Gb/s per TFLOP (Rorqual: 50 / 785 = 0.064). Point-to-point traffic is far slower than that: see the measured results.
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
| Added for Trillium, tested on H100 | PP, MXFP8 (stops cleanly without a Blackwell GPU), gradient-reduction dtype, TP with the `rows` layout, checkpointing, loss per step (merged) | 8 GPUs per node, checkpoint save / load, FP8 entries (MXFP8 needs a Blackwell GPU: not run), B200 layouts, loss from the last pipeline stage (branch `trillium-glm`) |
| Precision recipes | `bf16`, `mxfp8` (the pinned TorchTitan removed Float8 tensorwise / rowwise on 2026-10-01) | `bf16`, `fp8-tensorwise`, `fp8-blockwise` (runs as MXFP8 GEMMs on B200), `mxfp8` |
| Sequence | 4096 x 3 per GPU (12,288 tokens) by default | 8192, micro-batch 1: **micro-batch 2 is impossible** (the GLM-5 plugin's attention takes THD input only) |
| Global batch | grows with the GPUs (weak scaling) | 32 micro-batches per data-parallel rank (weak scaling and A-G): **256 / 512 / 768 / 1024 sequences at 13 / 26 / 39 / 52 nodes**, 27% bubble *(est., from the 1F1B formula)*; strong scaling 768 at all four; 64 per copy as a small-batch point at 13 nodes (8 micro-batches against 13 stages: 60% bubble *(est.)*) |
| Recompute | selective by default; full and none as variants | full by default; selective is `moe_act layernorm` (**`mla_up_proj` is impossible**: the plugin's attention never sets the checkpoint it asserts on) |
| Training | everything trains | everything trains (no frozen experts: memory allows it) |

Network arithmetic for Qwen *(est.)*: any data-parallel layout (FSDP, HSDP, TP + FSDP) moves ~8 bytes per
parameter per node per step (BF16 gathers forward + backward, FP32 gradient reduction): ~215 GB, so
**>= 1.1 s per step at 1,600 Gb/s**, against ~3 s of compute at 12,288 tokens per GPU and ~50% of 1,668
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

**Health checks, measured on 2026-11-04** (51 nodes, jobs 1238238, 1238239, 1238240, 1238571; 25 nodes per job; srygaard's tool, not the matmul / per-HCA checks D1 lists, which it does not have):

- All GPUs pass DCGM, with no Xid or uncorrected ECC. Medians: 727 GB/s intra-node, 337 GB/s pair, 14.5 GB/s loopback, about 1,295 TFLOPS (FP16 diagnostic).
- `trig0122`: excluded in the launchers until the admins confirm the firmware update, then rerun the health check. `mlx5_1` firmware 28.35.2000 (others 28.47.1026), pairs 289-292 GB/s against 337; 25-node all-reduce 165 GB/s with it, 177 without.
- `trig0109` GPU 7: 177,903 corrected ECC and 2 corrected remapped rows, unchanged over two runs. Kept; watch nodemon's ECC.
- **CPU clocks at full GPU power (~8 kW), 50 of 51 nodes:** mean 1,640-2,270 MHz against ~3,150 before, a quarter of samples at 200 MHz on the median node, cause unknown; `trig0114` holds ~3,190. Measured under DCGM's power test with CPU load, not training: check in the first runs (`stragglers`, nodemon's CPU collector).

**Network, measured on 2026-11-04** (up to 25 nodes, the job limit):

- D3: all-gather and reduce-scatter hold 174-192 GB/s from 2 to 25 nodes; all-reduce 171-188 from 4 nodes. All-to-all falls to 10.0 GB/s at 2 nodes and 4.3 at 25.
- D2, send/recv: on all 8 GPUs 45 GB/s per node on 2-4 nodes and 10.9 from 13 nodes. On GPUs 0, 2, 4, 6 alone: 148-163 on 2-4 nodes and about 64 within a group from 13 nodes. The drop comes with the job's size, not with traffic; cause unknown. It may affect pipeline stages and EP across nodes.
- **Two bandwidth groups**, no exception in 600 pairs on GPUs 0, 2, 4, 6: `trig0101-0128` (27 usable nodes) and `trig0129-0152` (23). Within a group 62.9-66.9 GB/s, across 45.0-47.5. Latency shows no groups (~25 us). Pairs across the two 25-node test sets (625 of 1,225) are unmeasured. So experiment P applies.

### E. Jobs side by side (second: decides whether runs can be packed)

| ID | what | est. time |
|---|---|---|
| E1 | Qwen best guess (HSDP-8, 24k tokens/GPU) on 4 disjoint random sets of 13 nodes: each set alone, then all 4 together; 2 repetitions with new sets | 1.5 h |
| R  | Reference run: Qwen on 4 fixed nodes, FSDP, 12,288 tokens/GPU, 10 min; **every ~3 h through the window** | 10 min each |

If E1 shows no slowdown (< 2%), runs of <= 26 nodes are packed in parallel for A, B, C, F.

### A. Parallel layout (core of the design)

**A-Q, Qwen** (5-min measurement after 30 warm-up steps, as on Rorqual; the cross runs selective recompute, the
entries' default):

| factor | levels |
|---|---|
| layout | `fsdp` (all GPUs) / `hsdp8` (shard in a node, replicate across) / `hsdp16` / `tp2-fsdp` / `tp4-fsdp` / `pp2-fsdp` / `pp4-fsdp` |
| tokens per GPU | 12k (4096 x 3) / 24k (4096 x 6) / 48k (4096 x 12) |
| gradient reduction | fp32 / bf16 (on the 2 best layouts) |
| activation checkpointing | selective (default) / full / none (1 node and 13 nodes; none only where it fits) |
| nodes | 13 (full cross); 2 and 52 for the 3 best; 1 for the non-PP layouts |

~45 runs x 2 repetitions x ~10 min *(est.)*.

**A-G, GLM-5** (20 iterations, the first left out as warm-up; startup ~15 min at scale *(est.)*). Every layout at a
node count takes that count's weak-scaling batch. A model copy is PP x EP GPUs; EP can only use the GPUs of one
stage, so the smallest run differs. Bubble = (PP - 1) / (micro-batches + PP - 1) *(est.: the formula)*:

| layout | EP group | smallest run (copies), batch | largest run (copies), batch | micro-batches, bubble |
|---|---|---|---|---|
| `pp13-ep8` | in a node (NVLink) | 13 nodes (1), 256 | 52 nodes (4), 1024; 26 nodes (2), 512: B-G-weak's run | 32, 27% |
| `pp13-ep16` | across 2 nodes | 26 nodes (1), 512 | 52 nodes (2), 1024 | 32, 27% |
| `pp13-ep32` | across 4 nodes | 52 nodes (1), 1024 | same | 32, 27% |
| `pp26-ep8` | in a node | 26 nodes (1), 512 | 52 nodes (2), 1024 | 64, 28% |
| `pp6-ep16` (13 layers per stage) | across 2 nodes | 12 nodes (1), 256 | 48 nodes (4), 1024 (416 GPUs don't split in 6) | 16, 24% |

The EP comparison (8 / 16 / 32) is made at 52 nodes, EP 8 / 16 and PP 13 / 26 also at 26 nodes; `pp13-ep8` and `pp6-ep16`
also at 13 / 12 nodes. Plus recompute full vs selective (`moe_act layernorm`, at 13 and 52 nodes) and the small batch, 64 at
13 nodes (60% bubble *(est.)*). No micro-batch comparison: the plugin only takes micro-batch 1.
~12 runs x 2 repetitions x ~25 min *(est.)*.

### B. Scaling (with the best layouts from A)

| ID | what |
|---|---|
| B-Q-weak | Qwen, 1, 2, 4, 8, 13, 26, 52 nodes, best layout and `fsdp` baseline, fixed tokens per GPU |
| B-Q-strong | Qwen, fixed global batch (52 x 8 x 24k tokens), 13, 26, 52 nodes |
| B-G-weak | GLM-5 `pp13-ep8`, 1, 2, 3, 4 copies (13, 26, 39, 52 nodes), 32 micro-batches per data-parallel rank: global batch 256 / 512 / 768 / 1024, bubble 27% throughout *(est.)* |
| B-G-strong | same nodes, global batch 768 at all four: 96 / 48 / 32 / 24 micro-batches, bubble 11 / 20 / 27 / 33% *(est.)*, the cost of spreading a fixed batch |

**Runs serving two experiments.** `run.experiment` holds one value, the launcher's tag, and these runs are not repeated;
the analysis reads them under both (`analysis/tri_analysis/select.py`, one place; `tests/test_series.py` and `analyze.py`
check that each series has all its node counts):

| entry | `run.experiment` | also serves |
|---|---|---|
| `megatron-glm5-tri-pp13-ep8-n13-gbs256` | A-G | B-G-weak, 13 nodes |
| `megatron-glm5-tri-pp13-ep8-n52-gbs1024` | A-G | B-G-weak, 52 nodes |
| `megatron-glm5-tri-pp13-ep8-n39-gbs768` | B-G-weak | B-G-strong, 39 nodes |
| `megatron-glm5-tri-pp13-ep8-n26-gbs512` | B-G-weak | A-G, the pp13-ep8 reference at 26 nodes |

**To look at in weak scaling: a one-off drop from 1 to 2 copies.** The first copy added brings cross-node data parallelism,
with a cost per step that does not depend on the batch. A truncated GLM-5 on H100 looked like this from 2 to 4 nodes: 0.76
efficiency, which a +0.76 s per-step constant accounts for (**the cause is inferred, not measured**). So one
efficiency number can mislead: report efficiency against 1 copy **and** against 2 copies (the first run with that traffic),
and likewise Qwen against 1 node and 2 nodes. A drop from 1 to 2 followed by a flat line is not a scaling limit.

### C. Precision

| ID | what |
|---|---|
| C-speed | Qwen: BF16 / MXFP8; GLM: BF16 / FP8 tensorwise / FP8 blockwise / MXFP8; best layout, at 1 node (Qwen) or 13 nodes (GLM) and at 52 nodes |
| C-loss | same recipes, 500 steps from the same seed and data (FineWeb, a realistic learning rate) at 13 nodes: loss difference to BF16. **Same-recipe repeats**: one more BF16 with the same seed, one with another seed, so that the FP8 vs BF16 gap can be compared with noise (Qwen: `TT_REPETITION=2 tri-closs-bf16`, `tri-closs-bf16-seed2`; GLM: `-rep2`, `-seed2` entries) |

GLM's loss curve starts with a warm-up: the learning rate starts at 0 (50 warm-up steps), so steps 1 and 2 see the same
weights, then an early unstable phase. Compare recipes after it, not on the first steps.

### F. Operations

| ID | what |
|---|---|
| F1 | Checkpoint save and load: Qwen (full ~0.32 TB *(est.)*, weights ~54 GB *(est.)*) at 1, 13, 52 nodes; GLM-5 weights only (~1.5 TB *(est.)*) and full (~8.9 TB *(est.)*) at 13 and 52 nodes; synchronous vs asynchronous (time training is blocked vs total write); 3 repetitions at different times of day. **GLM runs weights first**; fallback for full: `pp13-ep16` on 26 nodes (see below). Escalation with the admins' agreement: Qwen 1 node -> Qwen 52 -> GLM weights -> GLM full. **Only one GLM-5 checkpoint on disk at a time**, deleted after the load is timed. No local disk: staging only in host memory (asynchronous mode) |
| F2 | Long run: best Qwen layout on 52 nodes, >= 1 h (stability, stragglers, clocks, power) |
| F3 | Startup: time to an initialized model and to the first step, from every run (no extra runs) |
| F4 | Energy: tokens per kWh from GPU power (nodemon), from every run (no extra runs) |

**Checkpoint sizes.** Measured on truncated models (H100): full 12.0 B/param, weights 2.0 B/param. The whole-model sizes
above are estimates from those ratios (GLM-5 744e9 parameters, Qwen 26.9e9).

**Full checkpoints raise GPU memory, on saves too.** Truncated GLM-5 on Fir: at PP2 x EP4, 38.4 GiB in training rose to
56.5-60.5 GiB on saves, and a load ran out of memory at 79 GiB; at PP2 x EP8 on B200, the load peak was 1.95 x the training
peak (50.28 against 25.85 GiB, job 1239618). So `full 13` (one copy on 13 nodes) may not fit in 180 GB: ~190 GiB *(est.)*. Hence weights first, and the full checkpoint's
first load read through `peak_allocated_gib` before the 52-node one. If it does not fit: `pp13-ep16` on 26 nodes (half the
expert state per GPU), entries `megatron-glm5-tri-ckpt-full-ep16-*-n26`.

### If only 47 nodes

No longer expected (2026-11-04): `trig0148-0152` joined the reservation. Kept for the plan's shape on fewer nodes.

`N_NODES` in `analysis/tri_analysis/machine.py` is the one setting: 52, or 47 if trig0148-0152 stay reserved. The synthetic
data, figures and scales of the report follow it. What changes at 47 (376 GPUs):

| | 52 nodes | 47 nodes |
|---|---|---|
| GLM largest `pp13-ep8` | 52 nodes, 4 copies | **39 nodes, 3 copies** (it serves A-G's top as well) |
| `pp26-ep8`, `pp13-ep32` | run | **dropped** |
| `pp13-ep16` | 26 and 52 nodes | 26 nodes: the EP comparison is 8 vs 16 at 26 nodes |
| `pp6-ep16` | 12 and 48 nodes | 12 nodes only (48 does not fit; 36 is the next valid count, no entry) |
| B-G | 1-4 copies | 1-3 copies |
| GLM C-speed, F1 top | 52 nodes | 39 nodes: **needs new entries** (dev.yaml holds n13 and n52) |
| Qwen and network top (A-Q, B-Q-weak, C-speed, F1, F2, D3) | 52 | **47** |
| D2 | 26 pairs per round, 51 rounds, 1,326 pairs | 23 pairs per round, 47 rounds, 1,081 pairs |
| D4 | 26 + 26 | 23 + 23 (one node out) |
| E1 | 4 sets of 13 | 3 sets of 13 |
| B-Q-strong | 13, 26, 52 nodes | 13, 26 only: **no top point at 47.** The global batch (52 x 8 x 24k tokens, 2,496 sequences) does not divide 376 GPUs (6.64 sequences per GPU), and a batch that does (47 x 8 x 6 = 2,256 sequences) does not divide 104 or 208 GPUs. The largest count holding this batch is 39 nodes (8 sequences per GPU; no entry, and more tokens per GPU than the other points' 24k) |
| `hsdp16` | 26, 52 | needs a multiple of 16 GPUs: 26 or 46, not 47 (`tt_trillium.sh` refuses it) |

The Qwen entries other than B-Q-strong use every allocated node: `--nodes 47` needs no new entry.

### Budget *(est.)*

| group | serial | packed (if E1 holds) |
|---|---|---|
| bring-up on B200 (install, 1-node smoke of both workloads, FP8 / MXFP8 smoke) | 4 h | |
| D | 2 h | |
| E + R | 2.5 h | |
| A-Q | 15 h | 6 h |
| A-G | 12 h | 6 h |
| B | 8 h | 5 h |
| C (with the BF16 repeats of C-loss) | 10 h | 5 h |
| F | 6 h | |
| total | ~59 h | ~36 h |

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
| `glm_layouts` | Expert parallelism inside or across nodes? | tokens/s per GPU and all-to-all time by layout, at 12-13, 26 and 48-52 nodes; recompute, batch 64 | A-G (with B-G-weak's 26-node run) |
| `qwen_scaling` | How far does Qwen scale? | tokens/s per GPU and efficiency vs GPUs (8-416), best layout vs `fsdp`; strong scaling panel | B-Q |
| `glm_scaling` | Do model copies scale? | tokens/s per GPU vs copies (1-4), weak vs strong batch, bubble of each point | B-G (with A-G's 13 and 52 nodes) |
| `precision_speed` | What does FP8 / MXFP8 buy, and does it hold at scale? | speedup over BF16 by recipe, 1 node vs 52 nodes, both workloads | C-speed |
| `precision_loss` | Does it train the same, beyond the noise? | loss difference to one BF16 run vs step, by recipe; noise band from the BF16 repeats (same seed, other seed); GLM's lr warm-up marked | C-loss |
| `checkpoint` | What does a checkpoint cost? | save / load time and blocked time vs nodes, sync vs async, full vs weights only (pinned `blocked_s` from repetition 2 when present); write throughput; load and save peak GPU memory against 180 GB | F1 |
| `stability` | Does throughput hold over an hour? | tokens/s per step and GPU clocks / power vs time | F2 |
| `stragglers` | Are some GPUs slower? | per-GPU activity, per-node CPU clock (nodemon) | F2 |
| `startup` | How long until training? | time to the first step vs nodes, both workloads | F3 |
| `energy` | What does a token cost in energy? | tokens per kWh by layout, precision and scale | F4 |

## Data

Formats: `scripts/analysis/FORMAT.md` (`milabench.pretrain.v1`, health checks, nodemon, per-rank
metrics), plus the Trillium additions in `FORMAT.md` here (layout and precision fields, loss per step,
checkpoint events, collective sweeps). Raw data: `$SCRATCH/trillium/` on Trillium, copied read-only to
`analysis/data/raw/`. Synthetic data stands in for every experiment not run yet and is drawn hollow and
labelled as such in the figure itself. A measured run replaces the synthetic runs of its configuration; F1 per checkpoint
operation. A failed measured run replaces too: a configuration whose only measured runs failed has no synthetic point; it gets a red
cross (`checkpoint`, `glm_layouts`, `glm_scaling`) and is named in the report's draft box and ledger. Runs written on another cluster stop
`analyze.py build`: H100 test runs are not report data.
