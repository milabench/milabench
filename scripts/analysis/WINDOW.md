# Window runbook: Rorqual, 10-29 17:00 -> 10-30 17:00

The order of the experiments and the command for each block. Why each block exists, the figure it feeds and the
fallbacks are in `EXPERIMENTS.md`; the GLM-5 runs are described in `GLM_RUNS.md` and run with the scripts of
PR #46 (`scripts/rorqual/`, branch `megatron`).

## How blocks run

- **One block at a time.** A block is one Slurm job, or a few jobs launched together that share no
  interconnect (a spine-crossing run plus a run on a leaf it does not use). Nothing else runs next to a GLM-5
  run, a Qwen ALL run or a whole-machine health check.
- **Each block waits for everything submitted before it**: `--dependency=afterany:<every job still pending
  or running in the reservation>`. A failed block does not stop the chain. `squeue` shows the running block
  and the ones waiting, in order.
- **Submit a few blocks ahead in the daytime**, so that a node excluded mid-window only means resubmitting
  those. **Before the night, submit to the end of the current phase.** Every job has a time limit, so a stuck
  run costs at most that.
- **The window log** (team channel): every submitter posts the block number, the job ids, then the outcome
  (done / failed / skipped and why). The health owner posts the healthy node list `H` after each health check.

Dependency on everything already in the reservation (any group, any user):

```bash
AFTER=$(squeue -h --reservation=MILABENCH2026 -t PENDING,CONFIGURING,RUNNING -o %i | paste -sd:)
```

`scripts/slurm/window.sh` (this branch) defines it as `window_after`, plus `window_dep` (the same as
`afterany:<ids>`, empty when nothing is queued: for `healthcheck.py --dependency`), `window_status` (what runs and
what waits) and `qblock` (one Qwen block, below). Check once at the start that `squeue` shows the other users'
jobs; if it does not, post job ids in the log and pass them by hand.

Time limits: health ~20 min, Qwen block 30 min (long run 90 min), GLM G block 1 h, GLM fit ladder 3 h.

## Setup (once, each group, before 17:00)

**Health**, from the slurm-healthcheck clone (template with `--reservation=MILABENCH2026`):

```bash
LEAVES=ihpc134u25,ihpc135u25,ihpc136u25,ihpc217u25,ihpc218u25,ihpc316u25,ihpc317u25,ihpc318u25,ihpc319u25,\
ihpc320u25,ihpc321u25,ihpc322u25,ihpc323u25,ihpc324u25,ihpc325u25,ihpc326u25   # the 66 full-GPU nodes, no MIG leaf
source <milabench checkout of this branch>/scripts/slurm/window.sh     # window_after, window_dep, window_status
```

**Qwen**, on a login node:

```bash
cd $SCRATCH/tt38/milabench && source scripts/slurm/window.sh
export TT_CAMPAIGN=window-1030 TT_EXCLUSIVE_CLUSTER=1 TT_CONFIG=window TT_LOCAL_VENV=1
export TT_HEALTHY=<H from the log>       # after block 1
```

**GLM**, in the `megatron` checkout, `scripts/rorqual/` (see its README):

```bash
source <milabench checkout of this branch>/scripts/slurm/window.sh     # window_after, window_status
export CAMPAIGN=window-1030 EXCLUSIVE_CLUSTER=1 NO_TAIL=1
export NODELIST=<H from the log> NODES=<number of nodes in H>     # after block 1: whole healthy machine, no MIG node
export HEALTH_DIR=<health results dir> EXCLUDE_REASONS='{"rg21701": "CPU clock drops ~30% under training load"}'
```

GLM jobs take every healthy node (`NODELIST=H`): each entry picks its own 50 or 64 nodes inside.

Blocks are numbered in the order they run, one at a time. The "exp." column is the block's number in
`EXPERIMENTS.md`, where the reasoning behind each block is.

## Block 1: health check and node lists (health, ~10 min)

```bash
uv run healthcheck.py run --job-per switch --switches $LEAVES --no-wait
uv run healthcheck.py report $SCRATCH/slurm-healthcheck/results/<job_id>
```

Then `H` = the passed nodes, minus **rg21701** (its CPU clock drops under training load and no health check
catches it reliably, journal 10-26):

```bash
H=$(sed -n 's/^#SBATCH --nodelist=//p' $SCRATCH/slurm-healthcheck/results/<job_id>/include.sbatch)
H=$(scontrol show hostnames "$H" | grep -vx rg21701 | paste -sd,)
scontrol show hostnames "$H" | wc -l     # 65 if nothing else failed
```

Post `H`, its count and the failed nodes with their reason. GLM checks its node sets before submitting
anything (`./00_plan.sh "$H"`): 50 nodes for G1-G3 / G5, 64 for G4, crossing counts G1 0, G2 all, G3 about 1
in 4. A 52-node aligned set means rg21701 is back in `H`. Exclusions change other blocks too: see "If nodes are
excluded after the health check" in `EXPERIMENTS.md`.

## Block 2: GLM-5 full-training fit ladder (GLM, ~35 min to ~2 h)

Starts after block 1, alone on the machine.

```bash
AFTER=$(window_after) ./01_check_full.sh both
```

Two jobs, one after the other: the 50-node set, then the 64-node set. Each tries as configured, then optimizer
offload 0.25, 0.5, 1.0, and stops at the first that completes; the result is saved where the G scripts read it.
4 steps per attempt; an attempt costs ~7-20 min whether it passes or fails (start-up dominates; estimate, not
measured). Best case ~35 min (both pass as configured), typical ~1 h, worst ~2 h.

Outcome, posted in the log: the rung of each set. It decides what every GLM block below runs:

| 50-node set | 64-node set | G1-G3, G5 | G4 |
|---|---|---|---|
| passes at X | passes at Y | full at X | full at Y |
| nothing fits | passes at Y | frozen | full at Y, **plus G4 frozen** (blocks 10 and 21) to compare with G1-G3 |
| nothing fits | nothing fits | frozen | frozen |

Each set runs at its own best rung, as it runs its own pipeline depth; the workload (frozen or full) must match
between runs that are compared.

## The queue

Times: Qwen ~12 min per run (measured), GLM ~20 min per G run (12 steps plus start-up, **assumed**), fit ladder
~1 h (typical). Cumulative times exclude failures and waiting; the window starts at 17:00. "‖" = runs launched
together **inside one block** (one job, no shared interconnect).

**Qwen block** (one job on exactly the nodes its specs pick, after everything before it):

```bash
qblock <block> <time limit> <tt_block.sh specs>      # e.g. qblock 06 00:30:00 H:27 S:12:ihpc316u25
```

`qblock` dry-runs `tt_block.sh` on the login node to get the nodes, then submits the block on those nodes; in
the job, `tt_block.sh` picks the same nodes from `TT_HEALTHY` and launches the runs together.

**GLM block** (every healthy node, after everything before it):

```bash
AFTER=$(window_after) ./02_g1_aligned.sh pass1       # 03_g2_crossed.sh, 04_g3_mixed.sh, 05_g4_all_nodes.sh, 06_g5_fp8.sh
```

### Pass 1: one run of every tier-0 and tier-1 comparison (~3.8 h, ends ~20:50)

| # | exp. | group | block | command | est. | cum. |
|---|---|---|---|---|---|---|
| 1 | 0 | H | Health `--job-per switch`, node lists | above | 10 min | 0:10 |
| 2 | – | G | Full-training fit ladder, 50 then 64 nodes | above | ~1 h | 1:10 |
| 3 | 1 | G | **G1 EP8 aligned** | `./02_g1_aligned.sh pass1` | 20 min | 1:30 |
| 4 | 2 | Q | **ALL** | `qblock 04 00:30:00 ALL` | 12 min | 1:42 |
| 5 | 3 | G | **G2 EP8 crossed** | `./03_g2_crossed.sh pass1` | 20 min | 2:02 |
| 6 | 4 | Q | **H27** ‖ **S12** (ihpc316) | `qblock 06 00:30:00 H:27 S:12:ihpc316u25` | 12 min | 2:14 |
| 7 | 5 | G | **G3 EP8 mixed** | `./04_g3_mixed.sh pass1` | 20 min | 2:34 |
| 8 | 6 | Q | **P27** ‖ **H3** | `qblock 08 00:30:00 P:27 H:3` | 12 min | 2:46 |
| 9 | 7 | G | **G4 every node** | `./05_g4_all_nodes.sh pass1` | 20 min | 3:06 |
| 10 | – | G | G4 frozen, **only if nothing fits on 50 nodes** (else skip) | `FULL=none ./05_g4_all_nodes.sh pass1` | +20 min | |
| 11 | 8 | Q | **O12** ‖ **S3** (ihpc316) | `qblock 11 00:30:00 O:12 S:3:ihpc316u25` | 12 min | 3:18 |
| 12 | 9 | Q | **H12** ‖ **S6** (ihpc218) | `qblock 12 00:30:00 H:12 S:6:ihpc218u25` | 12 min | 3:30 |
| 13 | 10 | H | Health `--job-per cluster`, then `--switches ihpc217u25,ihpc316u25 --job-per cluster` | below | 20 min | 3:50 |

Block 13 is two spine-crossing jobs; the second waits for the first:

```bash
uv run healthcheck.py run --job-per cluster --nodes "$H" --no-wait --dependency "$(window_dep)"
uv run healthcheck.py run --job-per cluster --switches ihpc217u25,ihpc316u25 --no-wait --dependency "$(window_dep)"
```

### Pass 2: second repetition (~2.7 h, ends ~23:30)

| # | exp. | group | block | command | est. | cum. |
|---|---|---|---|---|---|---|
| 14 | 11 | G | G1 (rep 2) | `./02_g1_aligned.sh pass2` | 20 min | 4:10 |
| 15 | 12 | Q | ALL (rep 2) | `qblock 15 00:30:00 ALL` | 12 min | 4:22 |
| 16 | 13 | G | G2 (rep 2) | `./03_g2_crossed.sh pass2` | 20 min | 4:42 |
| 17 | 14 | Q | **S12 alone** (leaf-assumption check: compare with block 6) | `qblock 17 00:30:00 S:12:ihpc316u25` | 12 min | 4:54 |
| 18 | 15 | G | G3 (rep 2) | `./04_g3_mixed.sh pass2` | 20 min | 5:14 |
| 19 | 16 | Q | H27 (rep 2) ‖ S6 (ihpc316) | `qblock 19 00:30:00 H:27 S:6:ihpc316u25` | 12 min | 5:26 |
| 20 | 17 | G | G4 (rep 2) | `./05_g4_all_nodes.sh pass2` | 20 min | 5:46 |
| 21 | – | G | G4 frozen (rep 2), only if block 10 ran | `FULL=none ./05_g4_all_nodes.sh pass2` | +20 min | |
| 22 | 18 | Q | P27 ‖ H3 (rep 2) | `qblock 22 00:30:00 P:27 H:3` | 12 min | 5:58 |
| 23 | 19 | Q | O12 ‖ S3 (rep 2) | `qblock 23 00:30:00 O:12 S:3:ihpc316u25` | 12 min | 6:10 |
| 24 | 20 | Q | H12 ‖ S6 ihpc218 (rep 2) | `qblock 24 00:30:00 H:12 S:6:ihpc218u25` | 12 min | 6:22 |
| 25 | 21 | H | Health re-check `--job-per switch` | as block 1, with `--dependency "$(window_dep)"` | 10 min | 6:32 |

If block 17 differs from block 6 by more than the run-to-run spread, stop pairing: from block 26 on, submit
each "‖" spec as its own `qblock`.

### P2: the rest of the tier-1 curves and GLM-5 FP8 (~6.2 h, overnight, ends ~05:45)

Submit the whole phase before the night.

| # | exp. | group | block | command | est. | cum. |
|---|---|---|---|---|---|---|
| 26 | 22 | G | G5 FP8 | `./06_g5_fp8.sh pass1` | 20 min | 6:52 |
| 27 | 23 | Q | H15 ‖ S9 (ihpc316) | `qblock 27 00:30:00 H:15 S:9:ihpc316u25` | 12 min | 7:04 |
| 28 | 24 | G | G5 FP8 (rep 2) | `./06_g5_fp8.sh pass2` | 20 min | 7:24 |
| 29 | 25 | Q | P15 (12 + 3) | `qblock 29 00:30:00 P:15` | 12 min | 7:36 |
| 30 | 26 | Q | H48 (or the largest H can build: H45, H42) | `qblock 30 00:30:00 H:48` | 12 min | 7:48 |
| 31 | 27 | Q | P48 (same size as block 30) | `qblock 31 00:30:00 P:48` | 12 min | 8:00 |
| 32 | 28 | Q | H6 ‖ S6 (ihpc316, rep 2) | `qblock 32 00:30:00 H:6 S:6:ihpc316u25` | 12 min | 8:12 |
| 33 | 29 | Q | O6 ‖ S9 (ihpc316, rep 2) | `qblock 33 00:30:00 O:6 S:9:ihpc316u25` | 12 min | 8:24 |
| 34 | 30 | Q | H9 | `qblock 34 00:30:00 H:9` | 12 min | 8:36 |
| 35 | 31 | Q | O9 | `qblock 35 00:30:00 O:9` | 12 min | 8:48 |
| 36 | 32 | Q | O16 | `qblock 36 00:30:00 O:16` | 12 min | 9:00 |
| 37 | 33 | Q | O3 | `qblock 37 00:30:00 O:3` | 12 min | 9:12 |
| 38 | 34 | Q | Second repetition of blocks 27, 29-31, 34-37 | the same `qblock` commands, one block each | ~2 h | 11:12 |

### P3: if time remains (~1.5 h)

| # | exp. | group | block | command | est. | cum. |
|---|---|---|---|---|---|---|
| 39 | 36 | Q | Bridge H12 ‖ S12 at 2 x 8192 | `qblock 39 00:30:00 H:12::tt-w-nodes-8k2 S:12:ihpc316u25:tt-w-nodes-8k2` | 12 min | 11:24 |
| 40 | 37 | Q | Long stability run, every healthy node, >= 1 h | `qblock 40 01:30:00 ALL:::tt-w-nodes-long` | 70 min | 12:34 |
| 41 | 38 | H | Final health check `--job-per switch` | as block 25 | 10 min | 12:44 |

EP16 (`EXPERIMENTS.md` block 35) is not in the window: it needs its own 48-node set. Out-of-memory fallback
for any Qwen block: the same spec with bench `tt-w-nodes-lowmem` (e.g. `H:27::tt-w-nodes-lowmem`).

About 12.7 h of execution in the typical case, leaving ~11 h of the 24 for failures, fixes and waiting. After
a failure, keep the order but go to the next block of the other workload; the failed block comes back after it.

## If a node is excluded mid-window

1. The health owner posts the node, the reason and the new `H`.
2. Cancel the submitted blocks that have not started (`scancel <ids>`), set the new `H` (Qwen `TT_HEALTHY`,
   GLM `NODELIST` and `NODES`) and resubmit them.
3. A comparison is only valid on one node set: if GLM-5's 50-node set changes between G1, G2 and G3, run the
   three again on the new set. Same for the Qwen pairs.

## Monitoring

- `window_status` (or `squeue --reservation=MILABENCH2026`): the running block and the queue.
- Logs: Qwen `$SCRATCH/tt38/logs/q-<block>-<jobid>.out`; GLM `$MILABENCH_BASE/slurm/<job-name>-<jobid>.out`
  (crossing counts, peak memory, PASS / FAIL per attempt); health `$SCRATCH/slurm-healthcheck/results/<job_id>/`.
- After each block, the health owner runs `analyze.py quick` and compares with the expectations.
