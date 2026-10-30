# Window runbook: Rorqual, reservation MILABENCH2026 (opens after the maintenance, ends 10-30 16:00)

The order of the experiments and the command for each block. Why each block exists, the figure it feeds and the
fallbacks are in `EXPERIMENTS.md`; the GLM-5 runs are described in `GLM_RUNS.md` and run with the scripts of
PR #46 (`scripts/rorqual/`, now on `large_scale`).

## How blocks run

- **One block at a time.** A block is one Slurm job, or a few jobs launched together that share no
  interconnect (a spine-crossing run plus a run on a leaf it does not use). Nothing else runs next to a GLM-5
  run, a Qwen ALL run or a whole-machine health check.
- **Each block waits for everything submitted before it**: `--dependency=afterany:<every job still pending
  or running in the reservation>`. A failed block does not stop the chain. `squeue` shows the running block
  and the ones waiting, in order.
- **The window opens when the electrical maintenance ends** (16:00 at the earliest, most likely during the
  night) and ends at 10-30 16:00. Jobs submitted to the reservation wait until it opens, so the **night queue
  is submitted before the night** and starts by itself; the **day queue** is submitted once people are up.
- Every job has a time limit, so a stuck run costs at most that.
- **The window log** (team channel): every submitter posts the block number, the job ids, then the outcome
  (done / failed / skipped and why). The health owner posts the healthy node list `H` after each health check.

Dependency on everything already in the reservation (any group, any user):

```bash
AFTER=$(squeue -h --reservation=MILABENCH2026 -t PENDING,CONFIGURING,RUNNING -o %i | paste -sd:)
```

`scripts/slurm/window.sh` (on `large_scale`) defines it as `window_after`, plus `window_dep` (the same as
`afterany:<ids>`, empty when nothing is queued: for `healthcheck.py --dependency`), `window_status` (what runs and
what waits) and `qblock` (one Qwen block, below). Check once at the start that `squeue` shows the other users'
jobs; if it does not, post job ids in the log and pass them by hand.

Time limits: health ~20 min, Qwen block 30 min (long run 90 min), GLM G block 1 h, GLM fit ladder 3 h.

## Setup (once, each group, before the night)

**Health**, from the slurm-healthcheck clone (template with `--reservation=MILABENCH2026`):

```bash
LEAVES=ihpc134u25,ihpc135u25,ihpc136u25,ihpc217u25,ihpc218u25,ihpc316u25,ihpc317u25,ihpc318u25,ihpc319u25,\
ihpc320u25,ihpc321u25,ihpc322u25,ihpc323u25,ihpc324u25,ihpc325u25,ihpc326u25   # the 66 full-GPU nodes, no MIG leaf
source <a large_scale checkout>/scripts/slurm/window.sh       # window_after, window_dep, window_status
```

**Qwen**, on a login node:

```bash
cd $SCRATCH/tt38/milabench && git checkout large_scale && git pull --ff-only && source scripts/slurm/window.sh
export TT_CAMPAIGN=window-1030 TT_EXCLUSIVE_CLUSTER=1 TT_CONFIG=window TT_LOCAL_VENV=1
export TT_HEALTHY=$H                     # H below
```

**GLM**, in a `large_scale` checkout, `scripts/rorqual/` (see its README):

```bash
source ../slurm/window.sh                                       # window_after, window_status
export BRANCH=large_scale    # the scripts fast-forward to origin/$BRANCH; their default, megatron, is not an ancestor
export CAMPAIGN=window-1030 EXCLUSIVE_CLUSTER=1 NO_TAIL=1
export NODELIST=$H NODES=$(scontrol show hostnames "$H" | wc -l)   # whole healthy machine (H below)
export HEALTH_DIR=<health results dir> EXCLUDE_REASONS='{"rg21701": "CPU clock drops ~30% under training load"}'
```

GLM jobs take every healthy node (`NODELIST=H`): each entry picks its own 50 or 64 nodes inside.

**`H`, the healthy nodes.** Before the window, nobody knows which nodes the health check will fail, so the night
queue leaves out only **rg21701** (its CPU clock drops under training load and no health check catches it
reliably, journal 10-26) and the reserved nodes Slurm already marks as failed or drained for another reason than
the maintenance. A job whose node list holds a node that never comes back would wait forever and block the queue
behind it. Compute `H` right before submitting the night queue:

```bash
N=$(scontrol show reservation MILABENCH2026 | sed -n 's/.* Nodes=\([^ ]*\) .*/\1/p')
BAD=$( { echo rg21701; sinfo -h -N -n "$N" -o '%n|%t|%E' | sort -u |
         awk -F'|' '$2 ~ /drain|fail|down/ && $3 != "Not responding" {print $1}'; } | sort -u | paste -sd,)
H=$(scontrol show hostnames "$N" | grep -vxF -f <(tr , '\n' <<<"$BAD") | paste -sd,)
echo "$BAD"; scontrol show hostnames "$H" | wc -l
```

On 10-29 at 15:55 this gave `BAD=rg13601,rg21701,rg32101,rg32603` (rg13601 "having problems after reboot",
rg32101 NHC timeout on /localscratch, rg32603 GPU ECC errors) and 62 nodes. Consequences of those three, all on
3-node leaves (ihpc136, ihpc321, ihpc326):

- Qwen: H48 / P48 (blocks 16, 17, 26, 27) cannot be built: use the largest H can build (`H:30` / `P:30` with 10
  complete 3-node leaves; check with `TT_DRY_RUN=1`). ALL is 62 nodes. Everything else is unchanged.
- GLM-5: the aligned set stays 50 nodes (each 3-node leaf still gives 2). The every-node set is 62 nodes: 248
  GPUs, no usable EP8 layout, so the 64-node entries (G4, its fit ladder) fail at launch. G4 needs the 60-node
  entry (`megatron-glm5-g4-ep8-all-nodes-60n`, PP15) or the nodes back.

In the morning, `H` is replaced by block 1's result (below) for the day queue.

Blocks are numbered in the order they run, one at a time. The "exp." column is the block's number in
`EXPERIMENTS.md`, where the reasoning behind each block is.

## Night queue (submitted before the night, starts when the window opens)

Blocks are numbered in the order they run, one at a time. The "exp." column is the block's number in
`EXPERIMENTS.md`, where the reasoning behind each block is. Times: Qwen ~12 min per run (measured), GLM ~20 min
per G run (12 steps plus start-up, **assumed**), fit ladder ~1 h (typical); cumulative times count from the
opening. "‖" = runs launched together **inside one block** (one job, no shared interconnect).

Submit in this order, one person after the other (each submission depends on everything already queued):
health (block 1), then GLM (block 2), then Qwen (blocks 3-31, one `qblock` each, top to bottom).

| # | exp. | group | block | command | est. | cum. |
|---|---|---|---|---|---|---|
| 1 | 0 | H | Health `--job-per switch` | below | 10 min | 0:10 |
| 2 | – | G | GLM-5 full-training fit ladder, 50 then 64 nodes | below | ~1 h | 1:10 |
| 3 | 2 | Q | **ALL** | `qblock 03 00:30:00 ALL` | 12 min | 1:22 |
| 4 | 4 | Q | **H27** ‖ **S12** (ihpc316) | `qblock 04 00:30:00 H:27 S:12:ihpc316u25` | 12 min | 1:34 |
| 5 | 6 | Q | **P27** ‖ **H3** | `qblock 05 00:30:00 P:27 H:3` | 12 min | 1:46 |
| 6 | 8 | Q | **O12** ‖ **S3** (ihpc316) | `qblock 06 00:30:00 O:12 S:3:ihpc316u25` | 12 min | 1:58 |
| 7 | 9 | Q | **H12** ‖ **S6** (ihpc218) | `qblock 07 00:30:00 H:12 S:6:ihpc218u25` | 12 min | 2:10 |
| 8 | 12 | Q | ALL (rep 2) | `qblock 08 00:30:00 ALL` | 12 min | 2:22 |
| 9 | 14 | Q | **S12 alone** (leaf-assumption check: compare with block 4) | `qblock 09 00:30:00 S:12:ihpc316u25` | 12 min | 2:34 |
| 10 | 16 | Q | H27 (rep 2) ‖ S6 (ihpc316) | `qblock 10 00:30:00 H:27 S:6:ihpc316u25` | 12 min | 2:46 |
| 11 | 18 | Q | P27 ‖ H3 (rep 2) | `qblock 11 00:30:00 P:27 H:3` | 12 min | 2:58 |
| 12 | 19 | Q | O12 ‖ S3 (rep 2) | `qblock 12 00:30:00 O:12 S:3:ihpc316u25` | 12 min | 3:10 |
| 13 | 20 | Q | H12 ‖ S6 ihpc218 (rep 2) | `qblock 13 00:30:00 H:12 S:6:ihpc218u25` | 12 min | 3:22 |
| 14 | 23 | Q | H15 ‖ S9 (ihpc316) | `qblock 14 00:30:00 H:15 S:9:ihpc316u25` | 12 min | 3:34 |
| 15 | 25 | Q | P15 (12 + 3) | `qblock 15 00:30:00 P:15` | 12 min | 3:46 |
| 16 | 26 | Q | H48 (or the largest H can build: H45, H42) | `qblock 16 00:30:00 H:48` | 12 min | 3:58 |
| 17 | 27 | Q | P48 (same size as block 16) | `qblock 17 00:30:00 P:48` | 12 min | 4:10 |
| 18 | 28 | Q | H6 ‖ S6 (ihpc316, rep 2) | `qblock 18 00:30:00 H:6 S:6:ihpc316u25` | 12 min | 4:22 |
| 19 | 29 | Q | O6 ‖ S9 (ihpc316, rep 2) | `qblock 19 00:30:00 O:6 S:9:ihpc316u25` | 12 min | 4:34 |
| 20 | 30 | Q | H9 | `qblock 20 00:30:00 H:9` | 12 min | 4:46 |
| 21 | 31 | Q | O9 | `qblock 21 00:30:00 O:9` | 12 min | 4:58 |
| 22 | 32 | Q | O16 | `qblock 22 00:30:00 O:16` | 12 min | 5:10 |
| 23 | 33 | Q | O3 | `qblock 23 00:30:00 O:3` | 12 min | 5:22 |
| 24-31 | 34 | Q | Second repetition of blocks 14-17 and 20-23 | the same `qblock` commands, numbered 24-31 | ~1.6 h | 7:00 |

If block 9 differs from block 4 by more than the run-to-run spread, the pairings after it are suspect: re-run
their "‖" specs one per `qblock` in the day queue.

### Block 1: health check

```bash
uv run healthcheck.py run --job-per switch --switches $LEAVES --no-wait
```

In the morning: `uv run healthcheck.py report $SCRATCH/slurm-healthcheck/results/<job_id>`, then the day queue's `H`
is the passed nodes, minus rg21701:

```bash
H=$(sed -n 's/^#SBATCH --nodelist=//p' $SCRATCH/slurm-healthcheck/results/<job_id>/include.sbatch)
H=$(scontrol show hostnames "$H" | grep -vx rg21701 | paste -sd,)
```

Post `H`, its count and the failed nodes with their reason. If a node failed, see "If a node is excluded
mid-window" below: the night runs that used it are re-run. GLM checks its node sets before submitting the day
queue (`./00_plan.sh "$H"`): 50 nodes for G1-G3 / G5, 64 for G4, crossing counts G1 0, G2 all, G3 about 1 in
4. A 52-node aligned set means rg21701 is back in `H`.

### Block 2: GLM-5 full-training fit ladder (~35 min to ~2 h)

```bash
AFTER=$(window_after) ./01_check_full.sh both
```

Two jobs, one after the other: the 50-node set, then the 64-node set. Each tries as configured, then optimizer
offload 0.25, 0.5, 1.0, and stops at the first that completes; the result is saved where the G scripts read it.
4 steps per attempt; an attempt costs ~7-20 min whether it passes or fails (start-up dominates; estimate, not
measured). Best case ~35 min (both pass as configured), typical ~1 h, worst ~2 h.

Its result (the rung of each set, read in the morning) decides what every G block of the day queue runs:

| 50-node set | 64-node set | G1-G3, G5 | G4 |
|---|---|---|---|
| passes at X | passes at Y | full at X | full at Y |
| nothing fits | passes at Y | frozen | full at Y, **plus G4 frozen** (blocks 36 and 41) to compare with G1-G3 |
| nothing fits | nothing fits | frozen | frozen |

Each set runs at its own best rung, as it runs its own pipeline depth; the workload (frozen or full) must match
between runs that are compared.

## Day queue (submitted in the morning, GLM first)

GLM-5's G blocks can only be submitted once the fit ladder's result exists (the G scripts read it when they
are submitted). Then:

1. Cancel the night blocks that have not started yet (`window_status`, then `scancel <ids>`), and note them.
2. GLM submits blocks 32-43 below, top to bottom.
3. Health submits block 44; Qwen resubmits the cancelled night blocks, then blocks 45-47.

**GLM block** (every healthy node, after everything before it):

```bash
AFTER=$(window_after) ./02_g1_aligned.sh pass1       # 03_g2_crossed.sh, 04_g3_mixed.sh, 05_g4_all_nodes.sh, 06_g5_fp8.sh
```

| # | exp. | group | block | command | est. |
|---|---|---|---|---|---|
| 32 | 1 | G | **G1 EP8 aligned** | `./02_g1_aligned.sh pass1` | 20 min |
| 33 | 3 | G | **G2 EP8 crossed** | `./03_g2_crossed.sh pass1` | 20 min |
| 34 | 5 | G | **G3 EP8 mixed** | `./04_g3_mixed.sh pass1` | 20 min |
| 35 | 7 | G | **G4 every node** | `./05_g4_all_nodes.sh pass1` | 20 min |
| 36 | – | G | G4 frozen, **only if nothing fits on 50 nodes** (else skip) | `FULL=none ./05_g4_all_nodes.sh pass1` | +20 min |
| 37 | 11 | G | G1 (rep 2) | `./02_g1_aligned.sh pass2` | 20 min |
| 38 | 13 | G | G2 (rep 2) | `./03_g2_crossed.sh pass2` | 20 min |
| 39 | 15 | G | G3 (rep 2) | `./04_g3_mixed.sh pass2` | 20 min |
| 40 | 17 | G | G4 (rep 2) | `./05_g4_all_nodes.sh pass2` | 20 min |
| 41 | – | G | G4 frozen (rep 2), only if block 36 ran | `FULL=none ./05_g4_all_nodes.sh pass2` | +20 min |
| 42 | 22 | G | G5 FP8 | `./06_g5_fp8.sh pass1` | 20 min |
| 43 | 24 | G | G5 FP8 (rep 2) | `./06_g5_fp8.sh pass2` | 20 min |
| 44 | 10 | H | Health `--job-per cluster`, then `--switches ihpc217u25,ihpc316u25 --job-per cluster` | below | 20 min |
| – | – | Q | Night blocks cancelled in step 1, and re-runs after an exclusion | same `qblock` commands | |
| 45 | 36 | Q | Bridge H12 ‖ S12 at 2 x 8192 | `qblock 45 00:30:00 H:12::tt-w-nodes-8k2 S:12:ihpc316u25:tt-w-nodes-8k2` | 12 min |
| 46 | 37 | Q | Long stability run, every healthy node, >= 1 h | `qblock 46 01:30:00 ALL:::tt-w-nodes-long` | 70 min |
| 47 | 38 | H | Final health check `--job-per switch` | as block 1, with `--dependency "$(window_dep)"` | 10 min |

GLM blocks 32-43: ~3.3 h (+40 min with blocks 36 and 41). Block 44 (two spine-crossing jobs; the second waits
for the first):

```bash
uv run healthcheck.py run --job-per cluster --nodes "$H" --no-wait --dependency "$(window_dep)"
uv run healthcheck.py run --job-per cluster --switches ihpc217u25,ihpc316u25 --no-wait --dependency "$(window_dep)"
```

EP16 (`EXPERIMENTS.md` block 35) is not in the window: it needs its own 48-node set. Out-of-memory fallback
for any Qwen block: the same spec with bench `tt-w-nodes-lowmem` (e.g. `H:27::tt-w-nodes-lowmem`).

About 13 h of execution in the typical case. The window ends at 10-30 16:00 whatever time it opens: if it opens
late, what does not fit is dropped from the bottom of the day queue (the queue is in priority order). After a
failure, keep the order; the failed block comes back after the next one.

## If a node is excluded mid-window

1. The health owner posts the node, the reason and the new `H`.
2. Cancel the submitted blocks that have not started (`scancel <ids>`), set the new `H` (Qwen `TT_HEALTHY`,
   GLM `NODELIST` and `NODES`) and resubmit them.
   The night queue runs on the 65 nodes assumed healthy: if block 1 failed a node, the night runs that used
   it are re-run in the day queue (Qwen: same `qblock` command, the variant gets a new repetition number).
3. A comparison is only valid on one node set: if GLM-5's 50-node set changes between G1, G2 and G3, run the
   three again on the new set. Same for the Qwen pairs.

## Monitoring

- `window_status` (or `squeue --reservation=MILABENCH2026`): the running block and the queue.
- Logs: Qwen `$SCRATCH/tt38/logs/q-<block>-<jobid>.out`; GLM `$MILABENCH_BASE/slurm/<job-name>-<jobid>.out`
  (crossing counts, peak memory, PASS / FAIL per attempt); health `$SCRATCH/slurm-healthcheck/results/<job_id>/`.
- After each block, the health owner runs `analyze.py quick` and compares with the expectations.
