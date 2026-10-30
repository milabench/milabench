# Experiment checklist: Rorqual evaluation (Qwen3.8-27B, GLM-5)

Every experiment the report (`report/main.pdf`) needs, split into **before October 30** and the
**October 30 window**, then by group (health checks, Qwen, GLM). Each item names the figure it feeds.
Status 2026-10-26. Formats the runs must produce: `FORMAT.md`.

Constraints to keep in mind:

- Rorqual: reservation MILABENCH2026 (rg[21701-21707,21801-21807,31601-31607]) ends **10-27 04:00**;
  cluster-wide outage (POWEROUT20261027) from **10-27 05:00**; full-cluster access expected **10-29 17:00 ->
  10-30 17:00** (24 h). Anything on Rorqual "before Oct 30" must finish by 10-27 04:00.
- Fir and Tamia: normal queue only (no reservation); runs there are "shared links" (other users' traffic).
- Traffic isolation: placement comparisons are only valid within one level (alone / reserved one switch /
  shared). In the window, **runs that cross the spine must run alone**; one-switch runs on different leaves
  may run at the same time.
- Measured cost of one Qwen run: ~11-12 min (30 warm-up steps of ~10 s + 5 min measurement + ~1 min
  startup), e.g. job 25153034: P12 12 min, S6 11 min; rehearsal 25189970 (10-26): 11.4-11.5 min per block,
  first step ~20 s after launch.

Owner: fill in per item (`Owner:`).

---

## Before October 30

### Health checks

- [ ] **Access to srygaard results.** The window health check writes to the colleague's `$SCRATCH`, not
      readable by us. Get read access or an agreed copy location (`analyze.py collect --srygaard`). Copy the
      full results directories, not only `report.json`: the per-GPU TFLOPS (Fig. 3) are only in each node's
      output. Owner:
- [ ] **Dry run of the three window modes on reserved nodes** (before 10-27 04:00), to time them and check
      our adapter parses real output: `--job-per switch`, `--job-per cluster`, `--switches ihpc217u25
      ihpc316u25 --job-per cluster`. Figs. 2, 4. Owner:
- [ ] **nodemon (node telemetry)**: both launchers write
      `BENCHMATE_NODEMON_DIR=$SCRATCH/nodemon/<run_name>`. GLM-5
      (`scripts/rorqual/job.sbatch`) always sets it; the pack starts the sampler. TorchTitan
      (`scripts/slurm/tt_native.sh`) sets the same path; an empty value disables it. Still to check:
      one dry-run job's TSVs parse (FORMAT §3),
      and on Rorqual's H100s that `<host>.gpm.tsv` fills during a training step. An idle sample
      has an empty first tick, then zeros and a tiny `dram_bw_util`; tensor-pipe activity in the
      straggler figure needs a busy GPU, otherwise `util_mean`. Owner:

### Qwen3.8-27B (TorchTitan)

- [x] **Benchmark-side format changes** (FORMAT "Changes requested", items 1-4, 3b, 3c): summary
      `milabench.pretrain.v1`, `run_start` event (pushed before distributed init), launcher variables
      (campaign, variant, repetition, family, health dir, topology snapshot, reservation, exclusive_cluster),
      `gpu_uuid` + `local_rank` per rank, `CUDA_DEVICE_ORDER=PCI_BUS_ID`. Item 5 (`step` in rate events,
      optional) not done. Commits 07303b6e, dc310bfa. Owner: Qwen
- [x] **Node picker** (`tt_pick_nodes.py`): family **O** (1 node per switch, 3-node leaves first: O12 stays
      off ihpc316, O16 takes every leaf). Owner: Qwen
- [ ] **Node picker, GLM part**: "even number per leaf" selection and the aligned / mixed / crossed node
      orders. Owner: GLM
- [x] **Format validation run**: Rorqual 25189411 (2 nodes, 10-26): O1 x 2 + lowmem smoke through both
      launchers; `analyze.py all --no-synthetic` parses every field with no manual step. Owner: Qwen
- [x] **Rorqual 1-node `tt-w-smoke` rerun** (3 x 4096): 25171133 (10-25), 3 nodes x 3 reps on ihpc316:
      **8,413-8,514 tok/s/GPU (~39% MFU), spread +-0.6%**; confirmed on rg21702 / rg21705 (8,469-8,474,
      10-26). rg21701 is noisy on its own (6,089-8,386 over 5 runs) while passing the health check: keep
      it out of 1-node runs. Owner: Qwen
- [x] **Leaf assumption, pre-window check**: 25171129 (10-25): S7 on ihpc316 alone / next to P14 (217+218) /
      alone again = 1176 / 1176 / 1177 tok/s/GPU; P14 alone / paired = 1154 / 1155. No effect beyond the
      ~0.1% run-to-run spread: pairing is safe. The in-window check (blocks 4 / 14) stays. Owner: Qwen
- [x] **Bridge and gradient accumulation at 6 nodes**: 25171134 (10-25): 3 x 4096 = 1217, 2 x 8192 = 1512
      tok/s/GPU (+24%; +21% at 3 nodes). Accumulation over 4 microbatches: 1218 tok/s/GPU, step exactly 4x
      longer: FSDP2 all-gathers and reduce-scatters on every microbatch, so **accumulation gives no measured
      compute / communication split** (skipping the sync would keep ~108 GB of unsharded fp32 gradients per
      GPU). Fig. 9 stays estimated; accumulation removed from block 36. Owner: Qwen
- [ ] **Collect Rorqual 25045108** (H6 on 3-node leaves ihpc317/318, normal queue): still pending 10-26 and
      unlikely to start before the outage; no longer needed. Owner:
- [ ] **Tamia 639741 (H6) / 639742 (S6)**: pending, est. start 10-27 11:30 (nodes reserved for another
      job). Keep (4-HCA comparison, Fig. 8) or cancel. Fir 6398 / 6399 were cancelled. Owner:
- [ ] **Fir and Tamia nodes-per-switch at 3-12 nodes** (S, H, O; O up to 8 on Fir, 5 on Tamia), 2 reps:
      feeds the 4-HCA lines of Fig. 8. Pinned node lists queue for days without a reservation (Fir
      6398/6399 estimated 10-29 / 10-31): decide whether to try or drop the 4-HCA lines. Owner:
- [x] **Compile cache and venv archives on Rorqual** with the final code and shapes: every window shape
      (16-layer smoke, 64 layers at 3 x 4096 and 2 x 8192, lowmem) hit the cache with no miss in 10-25 / 10-26
      runs; the summary now reports `startup.compile_cache` (warm in 25189411 and 25189970). Owner: Qwen

### GLM-5 (Megatron / slime)

Full description for the GLM owner: `GLM_RUNS.md`.

- [ ] **Confirm the full model fits on 208 GPUs at EP8** (52 nodes). If not, choose one truncated depth for
      all expert-placement runs. Owner:
- [ ] **Output in milabench format**: voirfile or converter writing `rate` per step + `summary` (FORMAT §1),
      with `placement.ep_groups`, `params_trainable_fraction`, and per-EP-group all-to-all time if Megatron's
      MoE timers allow (`per_ep_group`). Fig. 12 needs the per-group timers. Owner:
- [ ] **Node orders for the 52-node set**: generate aligned / mixed (~1 group in 4 crossing) / crossed node
      lists from the healthy set; check them on the login node (no GPU). Owner:
- [ ] **Small truncated runs** (Fir or Rorqual reservation): EP4 / EP8 / EP16, BF16 vs FP8, to validate the
      pipeline and the format end to end (Fig. 10 small-scale points). Owner:
- [ ] **Measure GLM-5 run length** (startup + warm-up + measurement) at a few nodes: the window queue below
      assumes ~15 min per run until measured. Owner:

### Window preparation (all groups)

- [ ] **One command per block**: each window block below launches from a prepared command (config, node list
      from the picker, run tag), so the next block starts within minutes of the previous one. **Qwen: done**
      (`tt_block.sh`, commands in "Qwen block commands" below; every block dry-run on the 66 nodes, no node
      shared within a block). GLM: to do. Owner (per group):
- [ ] **Node lists recomputed from the healthy set** right after the first health check, for every block
      (deterministic picker; re-run it after any exclusion). Owner: health
- [ ] **Allocation strategy**: hold the nodes in long allocations and launch blocks as job steps
      (`tt_variants.sh` pattern), instead of one Slurm job per run (no queue wait between blocks). Qwen side
      validated in the rehearsal (health check + 3 blocks as job steps of one 9-node job). Still to check: the
      maximum walltime available in the window. Owner:
- [ ] **Window rehearsal** on Fir (2-3 nodes): health check -> Qwen block -> GLM block -> `analyze.py quick`,
      with the real commands. **Qwen part done on Rorqual** (25189970, 10-26, 9 nodes, 35 min): health 51 s;
      `H:6 S:3:ihpc316u25` (1215 ‖ 1227 tok/s/GPU, 11.5 min); `O:3 O:1::tt-w-smoke` (1221 ‖ 8413, 11.4 min);
      `ALL` (1197, 11.5 min); `analyze.py quick` 32 s. GLM part and the combined sequence: to do. Owner (all):

---

## Figure priority

Two purposes: the campaign began as a validation of the two new milabench workloads at large scale; access
to the full cluster adds the cluster questions for the admins, probed with those same workloads. Ranking
criteria: shows each workload runs and measures correctly at full scale; answers a cluster question;
informs a decision; only the window can produce it. (The first criterion leaves the order unchanged: tier 1
already runs each workload on every healthy node.)

The window queue follows this order. When a Qwen and a GLM-5 item are about equally important, **GLM-5
goes first**: its runs are more complex and more likely to fail, which leaves more time to fix them before
the window ends.

| tier | rank | figure | why |
|---|---|---|---|
| 0 | - | **Window health check** (data behind Fig. 2) | Sanity check and state of the cluster before any workload: healthy set, exclusions, node lists. Runs first |
| 1 | 1 | **11 GLM-5 expert placement** | Is using every node worth it vs the largest aligned subset; full scale only. Ahead of Fig. 7 (near-equal; GLM first) |
| 1 | 2 | **7 Weak scaling** | Central result: throughput per GPU up to the full machine vs the no-network ceiling |
| 1 | 3 | **5 At scale: balance or pack** | Allocation guidance at scale |
| 1 | 4 | **8 More nodes vs crossing switches** | Separates node count from crossing switches; explains Fig. 7 |
| 2 | 5 | **4 Bandwidth by scope** | Physical reason (one HCA per node, spine) |
| 2 | 6 | **2 Window health check** (as a figure) | Validity of the rest |
| 2 | 7 | **10 GLM-5 scaling, EP size, BF16 vs FP8** | MoE at scale, FP8 |
| 2 | 8 | **12 All-to-all per EP group** | Mechanism behind Fig. 11 (comes with its runs) |
| 2 | 9 | **9 Step decomposition** | Communication-bound step (estimated: gradient accumulation does not separate the two, measured 10-25) |
| 3 | 10 | 6 Per-GPU batch | Answered by preliminary data |
| 3 | 11 | 13 Startup | Every run feeds it |
| 3 | 12 | 15 Stragglers | node telemetry from regular runs |
| 3 | 13 | 14 Stability (>= 1 h) | Conditional in the brief; over 1 h of the full machine |
| 3 | 14 | 3 Per-GPU compute by cluster | Premise already shown by preliminary data |
| 3 | 15 | 1 Preliminary health checks | Background |

## Window: 10-29 17:00 -> 10-30 17:00 (Rorqual, alone on the cluster)

**To run the window, follow `WINDOW.md`**: the queue below with the command of every block, the GLM-5 blocks
as run by PR #46's scripts (fit ladder first, G1-G5), and how blocks are chained. This section keeps the
reasoning behind the order.

### Roles

- **Health and monitoring**: runs the health-check blocks, watches the node telemetry / logs during every
  run, runs
  `analyze.py quick` after each block and compares with expectations (preliminary numbers, synthetic
  model), and flags anything suspicious (node to exclude, throughput off, run stalled).
- **Qwen**: prepares and launches the Qwen blocks (Q).
- **GLM**: prepares and launches the GLM-5 blocks (G).

### Rules

- **Alternate Q and G blocks**: one group prepares while the other runs.
- **Skip on failure**: if a block fails (broken workload, suspicious result), log it and launch the next block
  of the *other* workload; the failing group fixes it while the other runs, and the block goes back in the
  queue after the next one.
- **Exclusions**: a node flagged by health is excluded, the picker recomputes node lists, and the ledger
  records which runs used it. Runs that crossed the spine run alone; 1-node runs (smoke, re-checks) may run
  on idle nodes at the same time.
- **First pass, then repeats**: pass 1 gives one run of every key comparison (each figure has data early);
  pass 2 adds the second repetition; then P2 and P3. Each block is useful on its own.

### Queue (in order)

Times: Qwen ~12 min per run (measured); GLM ~15 min per run (assumed until measured); cumulative time
excludes fixes and waiting. Within each pass, blocks follow the figure priority, GLM first. **"‖"** marks
runs launched at the same time (pairing rules: "Running tests in parallel" below): a block's time is its
longest run.

**Pass 1: one run of every tier-0 and tier-1 comparison (~2.5 h)**

| # | group | block | runs | feeds | est. | cum. |
|---|---|---|---|---|---|---|
| 0 | H | Health `--job-per switch` on all 66 nodes; topology snapshot; nodemon check; node lists for all blocks | 1 | tier 0; Figs. 2, 3 | 10 min | 0:10 |
| 1 | G+Q | **GLM EP8 aligned**, 52 nodes ‖ Qwen 1-node `tt-w-smoke` on an idle node ‖ 1-node health re-checks on the other idle nodes | 1 + 1 | Figs. 11, 7, 9 | 15 min | 0:25 |
| 2 | Q | **ALL** (every healthy node) | 1 | Fig. 7 | 12 min | 0:37 |
| 3 | G | **EP8 crossed**, same 52 nodes ‖ 1-node re-checks on idle nodes | 1 | Fig. 11 | 15 min | 0:52 |
| 4 | Q | **H27** (9 x 3-node leaves) ‖ **S12** (ihpc316; leaf-assumption check, part 1) | 2 | Figs. 5, 8, 7 | 12 min | 1:04 |
| 5 | G | **EP8 mixed**, same 52 nodes | 1 | Figs. 11, 12 | 15 min | 1:19 |
| 6 | Q | **P27** (12+8+7) ‖ **H3** control (a free 3-node leaf) | 2 | Figs. 5, 8 | 12 min | 1:31 |
| 7 | G | **EP8 mixed, every healthy node** | 1 | Fig. 11 | 15 min | 1:46 |
| 8 | Q | **O12** (1 node on 12 of the 3-node leaves) ‖ **S3** (ihpc316) | 2 | Figs. 8, 7 | 12 min | 1:58 |
| 9 | Q | **H12** (4 x 3-node leaves) ‖ **S6 on ihpc218** (control) | 2 | Fig. 8 | 12 min | 2:10 |
| 10 | H | Health `--job-per cluster` and `--switches ihpc217u25 ihpc316u25 --job-per cluster` | 2 | Fig. 4 | 20 min | 2:30 |

**Pass 2: second repetition of pass 1 (~2.4 h)**

| # | group | block | runs | est. | cum. |
|---|---|---|---|---|---|
| 11 | G | EP8 aligned (rep 2) ‖ 1-node re-checks on idle nodes | 1 | 15 min | 2:45 |
| 12 | Q | ALL (rep 2) | 1 | 12 min | 2:57 |
| 13 | G | EP8 crossed (rep 2) | 1 | 15 min | 3:12 |
| 14 | Q | **S12 alone** (rep 2; leaf-assumption check, part 2: compare with block 4) | 1 | 12 min | 3:24 |
| 15 | G | EP8 mixed (rep 2) | 1 | 15 min | 3:39 |
| 16 | Q | H27 (rep 2) ‖ **S6** (ihpc316, first run; P2 point pulled forward) | 2 | 12 min | 3:51 |
| 17 | G | EP8 mixed, every node (rep 2) | 1 | 15 min | 4:06 |
| 18 | Q | P27 (rep 2) ‖ H3 (rep 2) | 2 | 12 min | 4:18 |
| 19 | Q | O12 (rep 2) ‖ S3 (rep 2) | 2 | 12 min | 4:30 |
| 20 | Q | H12 (rep 2) ‖ S6 on ihpc218 (rep 2) | 2 | 12 min | 4:42 |
| 21 | H | Quick health re-check (`--job-per switch`) | 1 | 10 min | 4:52 |

If block 14 differs from block 4 by more than the run-to-run spread, stop pairing: from block 22 on, run every
"‖" item in sequence (~2.5 h more over P2 and P3), and treat paired results so far with caution.

**P2: the rest of the tier-1 curves and tier-2 GLM-5 (~4.5 h)**

| # | group | block | runs | feeds | est. | cum. |
|---|---|---|---|---|---|---|
| 22 | G | Full model **FP8** at the aligned scale (BF16 = block 1) ‖ 1-node re-checks | 1 | Fig. 10 | 15 min | 5:07 |
| 23 | Q | **H15** (5 x 3-node leaves) ‖ **S9** (ihpc316) | 2 | Figs. 5, 8 | 12 min | 5:19 |
| 24 | G | FP8 (rep 2) | 1 | Fig. 10 | 15 min | 5:34 |
| 25 | Q | **P15** (12+3; uses ihpc316) | 1 | Fig. 5 | 12 min | 5:46 |
| 26 | Q | **H48** (fallback: largest size H can build, e.g. H45) | 1 | Fig. 5 | 12 min | 5:58 |
| 27 | Q | **P48** (same size as block 26) | 1 | Fig. 5 | 12 min | 6:10 |
| 28 | Q | **H6** ‖ S6 (ihpc316, rep 2) | 2 | Fig. 8 | 12 min | 6:22 |
| 29 | Q | **O6** ‖ S9 (ihpc316, rep 2) | 2 | Fig. 8 | 12 min | 6:34 |
| 30 | Q | **H9** | 1 | Fig. 8 | 12 min | 6:46 |
| 31 | Q | **O9** | 1 | Fig. 8 | 12 min | 6:58 |
| 32 | Q | **O16** (one node on every leaf) | 1 | Fig. 8 | 12 min | 7:10 |
| 33 | Q | **O3** | 1 | Fig. 8 | 12 min | 7:22 |
| 34 | Q | Second repetition of blocks 23, 25-27, 30-33 (H15, P15, H48, P48, H6, O6, H9, O9, O16, O3); interleave GLM block 35 when it is ready | 10 | Figs. 5, 8 | 2 h | 9:22 |

**P3: if time remains (~2.1 h)**

| # | group | block | runs | feeds | est. | cum. |
|---|---|---|---|---|---|---|
| 35 | G | EP16 at full scale | 1-2 | Fig. 10 | 30 min | 9:52 |
| 36 | Q | Bridge **H12 at 2 x 8192** ‖ bridge **S12 at 2 x 8192** (ihpc316) | 2 | Fig. 6; Table 2 | 12 min | 10:04 |
| 37 | Q | Long stability run, every healthy node, >= 1 h, with node telemetry | 1 | Figs. 14, 15 | 70 min | 11:14 |
| 38 | H | Final health check (`--job-per switch`) | 1 | Figs. 2, 3 | 10 min | 11:24 |

Pass 1 + pass 2 + P2 + P3 is ~11.5 h of execution, leaving ~12.5 h of the 24 h for fixes, failures and
waiting. After a failure, the queue keeps the order but skips to the next block of the other workload.

### Qwen block commands

Inside one allocation holding the nodes, from `$SCRATCH/tt38/milabench` (each block is job steps, no queue
wait). Once per session:

```bash
S=scripts/slurm
export TT_CAMPAIGN=window-1030 TT_EXCLUSIVE_CLUSTER=1 TT_CONFIG=window TT_LOCAL_VENV=1
bash $S/tt_healthcheck.sh            # healthy.txt of this job: tt_block.sh picks from it
```

`tt_block.sh FAMILY:NODES[:LEAF][:BENCH] ...` picks each run's nodes among the healthy nodes the block's
earlier specs left (H / S / P / O: `tt_pick_nodes.py`; ALL: every remaining healthy node) and launches them
together. Bench defaults to `tt-w-nodes` (any node count). The run records campaign, variant (family + node
count), repetition (counted per campaign), family, health check and topology snapshot. `TT_EXCLUDE=<hostlist>`
leaves nodes out (e.g. GLM's 52 nodes in block 1, or a node flagged mid-window); `TT_DRY_RUN=1` prints the
plan only. Every block below was dry-run on the 66 full nodes (10-26): all build, no node shared within a block.

| # | command |
|---|---|
| 1 | `TT_EXCLUDE=<GLM 52-node list> bash $S/tt_block.sh O:1::tt-w-smoke` (avoid rg21701 for the smoke run: add it to `TT_EXCLUDE`) |
| 2, 12 | `bash $S/tt_block.sh ALL` |
| 4 | `bash $S/tt_block.sh H:27 S:12:ihpc316u25` |
| 6, 18 | `bash $S/tt_block.sh P:27 H:3` (P27 = 12 + 8 + 7 on ihpc316 / 217 / 218; H3 on a 3-node leaf) |
| 8, 19 | `bash $S/tt_block.sh O:12 S:3:ihpc316u25` |
| 9, 20 | `bash $S/tt_block.sh H:12 S:6:ihpc218u25` |
| 14 | `bash $S/tt_block.sh S:12:ihpc316u25` |
| 16 | `bash $S/tt_block.sh H:27 S:6:ihpc316u25` |
| 23 | `bash $S/tt_block.sh H:15 S:9:ihpc316u25` |
| 25 | `bash $S/tt_block.sh P:15` (12 + 3) |
| 26 / 27 | `bash $S/tt_block.sh H:48` / `bash $S/tt_block.sh P:48` (fallback: largest common multiple of 3, e.g. `H:45` / `P:45`) |
| 28 | `bash $S/tt_block.sh H:6 S:6:ihpc316u25` |
| 29 | `bash $S/tt_block.sh O:6 S:9:ihpc316u25` |
| 30 / 31 / 32 / 33 | `H:9` / `O:9` / `O:16` / `O:3` |
| 36 | `bash $S/tt_block.sh H:12::tt-w-nodes-8k2 S:12:ihpc316u25:tt-w-nodes-8k2` |
| 37 | `bash $S/tt_block.sh ALL:::tt-w-nodes-long` (1 h of measurement; check nodemon is sampling) |

Out-of-memory fallback: same spec with bench `tt-w-nodes-lowmem` (e.g. `H:27::tt-w-nodes-lowmem`).

## Running tests in parallel

**When two runs can share the window** (without corrupting each other's measurement):

- **1-node runs** have no inter-node traffic: any idle node, any time.
- **Leaf-local runs** (all nodes on one switch) never touch the spine: they can run next to any other run, as
  long as no other run uses their leaf. Assumption: a leaf switch does not slow traffic between its own nodes
  (same assumption as the "reserved, one switch" isolation level); checked in the window (see below).
- **Spine-crossing runs** share the spine: **one at a time**.

**Idle capacity during the big runs:** GLM-5 on the 52-node set leaves 14 nodes idle, at most one per leaf
(one per 3-node leaf + one on ihpc218): only 1-node runs fit there. Qwen ALL and GLM-5 on every node leave
nothing.

**Pairings** (a spine-crossing run + a leaf-local run on a leaf it does not use, or 1-node runs on idle nodes):

| spine-crossing or full run | in parallel | where |
|---|---|---|
| GLM-5 EP8 aligned / crossed / mixed (52 nodes) | Qwen 1-node smoke; 1-node health re-checks (DCGM diagnostics, matmul) of the idle nodes | 14 idle nodes |
| Qwen H27 (9 x 3-node leaves) | Qwen **S12** | ihpc316 |
| Qwen P27 (ihpc316 + 217 + 218) | Qwen **H3** control | a free 3-node leaf |
| Qwen O12 (1 node on 12 of the 3-node leaves) | Qwen **S3** | ihpc316 |
| Qwen H12 (4 x 3-node leaves) | Qwen **S6 on ihpc218** control | ihpc218 |
| Qwen H15 (5 x 3-node leaves) | Qwen **S6** | ihpc316 |
| Qwen H6 / H9 / O6 / O9 (3-node leaves only) | Qwen **S9**, controls (S6 ihpc218, H3) | ihpc316, ihpc218, free 3-node leaves |
| any spine-crossing Qwen run not using ihpc316 | bridge S12 at 2 x 8192 | ihpc316 |
| any run not using a given leaf | single-leaf health re-check of that leaf (`--switches <leaf>`) | that leaf |

Must run alone: Qwen ALL, P15 / P48 / H48 / O16 (use ihpc316 or every leaf), the stability run, GLM-5 on every
node, and the whole-machine health checks.

**Validation of the leaf assumption (built into the queue):** run S12 next to H27 in pass 1 and alone in
pass 2. If the two differ by more than the run-to-run spread, stop pairing and run everything in sequence.
Pre-window check on the reservation (25171129, 10-25): S7 on ihpc316 gave 1176 / 1176 / 1177 tok/s/GPU
alone / next to P14 on ihpc217+218 / alone again, and P14 1154 / 1155 alone / paired: no measurable effect.

**Gain** (folded into the queue above): about 2.5 h, from ~14 h to ~11.5 h of execution; pass 1 drops from
~3 h to ~2.5 h and picks up two leaf-to-leaf controls for free.

| phase | sequential | with pairings |
|---|---|---|
| pass 1 | ~3 h | ~2.5 h |
| pass 2 | ~2.6 h | ~2.3 h |
| P2 | ~6 h | ~4.5 h |
| P3 | ~2.7 h | ~2.2 h |

## If nodes are excluded after the health check

Rorqual's 66 full GPU nodes (Slurm Gres `gpu:h100:4`; the 27 MIG-partitioned nodes on leaves ihpc125-133 are excluded): 13 leaves of 3 nodes (CPU nodes share their uplinks), ihpc218 (7), ihpc217 (8),
ihpc316 (12, no CPU nodes). Rule: a node is used whole or not at all. Consequences checked by simulation on the
real topology (`scontrol show topology`):

| excluded | Qwen consequence | GLM-5 consequence | severity |
|---|---|---|---|
| 1 node, 3-node leaf | That leaf drops out of H: **H48 impossible**, at-scale pair becomes **H45 / P45**. Everything else unchanged (12 full 3-node leaves left; O16 still possible). | none (the leaf still gives 2 nodes); every node: 64 | low |
| 1 node, ihpc218 (7) | P27 needs 4 switches instead of 3 (12+8+6+1); report the switch count. Control S6 on ihpc218 still possible. | none (ihpc218 already gave 6 of 7) | low |
| 1 node, ihpc217 (8) | P27 on 4 switches. | **52 -> 50 nodes** (217 gives 6) | medium |
| 1 node, ihpc316 (12) | **S12 impossible**: the 12-node one-switch reference is lost (no other 12-node leaf). Fig. 8 ends at **9 nodes**; the S3+S12 pass-1 plan becomes S3+S9. P27 on 4 switches. | **52 -> 50** | **high (Fig. 8, tier 1)** |
| 2 nodes, same 3-node leaf | H45 / P45 | 52 -> 50 | medium |
| 2 nodes, two 3-node leaves | **H42** max (at-scale pair H42 / P42) | none | low |
| 2 nodes, ihpc218 | Control moves to S6 on **ihpc217** | 52 -> 50 | low |
| 2 nodes, ihpc316 | S12 impossible (S9 still possible) | 52 -> 50 | high |
| 4 nodes, ihpc316 | **S9 impossible too**: one-switch line only 3-6 nodes (S6 on 316, 217 or 218); Fig. 8 has little left | **52 -> 48** | **very high** |
| whole 3-node leaf (switch down) | H45; **O16 impossible** (O15) | 52 -> 50 | medium |
| 1 node on each of 4 3-node leaves | **H36** max: at-scale pair at 36 or less; H27 still possible | none | medium |

**Hard to adapt:**

1. **S12** depends on ihpc316 alone: any exclusion there removes the 12-node one-switch reference, a tier-1
   result (Fig. 8). There is no substitute on Rorqual. Mitigation: run S12 early (pass 1) while ihpc316 is
   intact, and record its health-check margins.
2. **H48 / P48** need all 13 3-node leaves complete: likely to shrink (any exclusion in 39 nodes). Plan the
   at-scale pair as "largest common multiple of 3 that H can build" (H45, H42, ...), not 48.
3. **GLM-5 52-node set** shrinks to 50 (or 48) with exclusions on ihpc217, ihpc316, or two on one 3-node leaf.
   Aligned, mixed and crossed must all run on the same set: if the set changes **mid-comparison**, restart
   the comparison on the new set (3 runs) rather than mixing sets.
4. **Mid-window exclusions** in general: a comparison (pairs, layouts) is only valid on one node set. Runs
   made before and after an exclusion are not repetitions of each other; the queue re-runs the affected
   block on the new set.

**Preparation (before the window):**

- [ ] Picker support for these fallbacks: "largest common H/P size", "even per leaf" set size, S line
      endpoint, control leaf choice (218 -> 217). Owner:
- [ ] Precompute node lists for every single-node exclusion (66 cases) and the common two-node cases, and
      check that every block can still be built (login node, no GPU). Owner: health
- [ ] Decision table for the health owner: which blocks change for an exclusion on each leaf type (the table
      above), so the queue can be adjusted in minutes. Owner: health

## Report coverage by phase

What each figure has if the window stops after a phase (phases are cumulative), in priority order.
**none**: no window data; **usable**: supports a statement in the report, with the limit given;
**complete**: as designed. Figure numbers as in the current `report/main.pdf`.

| tier | figure | after pass 1 (~2.5 h) | after pass 2 (~4.9 h) | after P2 (~9.4 h) | after P3 (~11.5 h) |
|---|---|---|---|---|---|
| 0 | Window health check (data) | complete (block 0) | complete + re-check | complete | complete + final check |
| 1 | 11 GLM-5 expert placement | usable: aligned, crossed, mixed, every node, 1 run each; no significance test | **complete** (2 runs each) | complete | complete |
| 1 | 7 Weak scaling | usable: 12, 24, 48, 108, ~264 GPUs + ceiling | usable, with repeats | **complete**: + 36, 60, 64, 192 GPUs | complete |
| 1 | 5 At scale: balance or pack | usable: 27 nodes, 1 run; no significance test | usable: 27 nodes, tested against run-to-run spread | **complete**: 15, 27, 48 nodes, 2 runs | complete |
| 1 | 8 More nodes vs crossing switches | usable: 3 and 12 nodes (drop on one switch, crossing cost at 12) + leaf-to-leaf controls, 1 run | usable, with repeats (+ S6) | **complete**: + 6, 9, 16 nodes, 2 runs | complete |
| 2 | 4 Bandwidth by scope | **complete** (blocks 0, 9) | complete | complete | complete |
| 2 | 2 Window health check (figure) | **complete** | complete | complete | complete |
| 2 | 10 GLM-5 scaling, EP size, BF16 vs FP8 | usable: EP8 BF16 at 52 nodes and every node (+ pre-window truncated runs) | usable, with repeats | usable: + FP8 at full scale | **complete**: + EP16 |
| 2 | 12 All-to-all per EP group | **complete** if Megatron logs per-group timers (pre-window item); otherwise none | complete (2 runs) | complete | complete |
| 2 | 9 Step decomposition | usable (estimated split: 1-node ceiling vs multi-node step) | usable | usable | usable (no measured split: gradient accumulation does not provide one) |
| 3 | 6 Per-GPU batch | usable (preliminary data only) | same | same | **complete** (window bridge point) |
| 3 | 13 Startup | usable: 1, 3, 6, 12, 27 nodes, every node | usable | **complete**: 3-48 nodes | complete |
| 3 | 15 Stragglers | usable: 5-min windows of pass-1 runs (nodemon `util_mean` + SM clock + CPU clock; tensor activity on Hopper) | usable | usable | **complete** (long run) |
| 3 | 14 Stability (>= 1 h) | none | none | none | **complete** |
| 3 | 3 Per-GPU compute by cluster | **complete** (slurm-healthcheck reports per-GPU fp16 TFLOPS) | same | same | same |
| 3 | 1 Preliminary health checks | complete (no window data needed) | complete | complete | complete |
| - | Tables 2, 3 (results, ledger) | grow with every run | | | |

What is lost if the window stops early:

- **Before pass 2 ends**: no significance test for the placement results (Figs. 11, 5): differences of
  0.2-1 % can't be told from run-to-run noise with one run each. Every tier-0 and tier-1 figure already
  has data.
- **Before P2 ends**: tier-1 curves stay sparse: Fig. 7 has five scales, Fig. 5 one node count, Fig. 8 two
  node counts (with the leaf-to-leaf controls). Fig. 10 has no FP8.
- **Before P3**: no stability figure (Fig. 14), no GLM-5 EP-size comparison at full
  scale (Fig. 10), no window bridge point (Fig. 6).
- **Independent of the window**: Fig. 12 depends on a pre-window tool change (per-group all-to-all
  timers in Megatron).

## After the window

- [ ] `analyze.py collect --srygaard rorqual:<path>` (node telemetry comes with it), then `analyze.py all --no-synthetic`. Owner: health
- [ ] Check parsed numbers against the run logs; fill the report's placeholders. Owner:
