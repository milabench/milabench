# GLM-5 runs on Rorqual

Scripts to run the GLM-5 expert-placement experiments (Megatron-LM, 744B, `benchmarks/megatron/dev.yaml`) on the
`MILABENCH2026` reservation. One script per step, run from a **login node** (it has internet, the compute nodes do not).
Every step submits one Slurm job and follows its log (ctrl-c stops following, not the job).

## Before you start

1. Stage everything once: `slurm/download_megatron_data.sh` (checkout, venv, Megatron-LM, data).
2. Know your healthy nodes. Keep the **same** `NODELIST` / `EXCLUDE` for every step you want to compare: G1, G2, G3
   and G5 are only comparable on the same set of nodes.

## The steps

| Step | Script | What it does |
|------|--------|--------------|
| 0 | `./00_plan.sh "<hostlist>"` | No GPU. Prints the node lists each run would use and how many expert groups cross a switch. |
| 1 | `./01_check_full.sh [g1\|g4\|both]` | Can GLM-5 train fully (nothing frozen)? Tries the memory ladder, stops at the first rung that runs. |
| 2 | `./02_g1_aligned.sh [pass1\|pass2]` | G1: 50 nodes, no expert group crosses a switch. The reference. |
| 3 | `./03_g2_crossed.sh [pass1\|pass2]` | G2: same 50 nodes, every expert group crosses. |
| 4 | `./04_g3_mixed.sh [pass1\|pass2]` | G3: same 50 nodes, about 1 group in 4 crosses. |
| 5 | `./05_g4_all_nodes.sh [pass1\|pass2]` | G4: 64 nodes (every healthy node), PP16. |
| 6 | `./06_g5_fp8.sh [pass1\|pass2]` | G5: G1 in FP8. Only if there is time. |
| – | `./all.sh [first [last]]` | Chains steps 2-6 in the window's order. `./all.sh --list` shows the order. |

Steps 2-6 each run **one** benchmark: **full training** if step 1 found that it fits on the run's node set, the
**frozen-experts** run only if it does not.

### Step 0: check the node sets

```bash
./00_plan.sh "rg[21601-21666]"
```

Check the crossing counts (G1 0, G2 all, G3 about 1 in 4) and the set sizes (50 for G1-G3 / G5, 64 for G4).
If the aligned set comes out at 52 nodes, the 50-node entries will fail at launch: exclude one node of a 2-node leaf
group with `EXCLUDE=`.

### Step 1: does full training fit?

```bash
./01_check_full.sh            # both node sets; the 64-node job waits for the 50-node one
./01_check_full.sh g1         # only the 50-node set
BALANCED=1 ./01_check_full.sh g1   # also try balanced pipeline stages without offload
```

Each attempt is 4 training steps. The ladder, from least to most memory saved:

1. as configured
2. (`BALANCED=1`, 50-node set only) balanced pipeline stages, no offload
3. optimizer state 25 % on the CPU
4. optimizer state 50 % on the CPU
5. optimizer state 100 % on the CPU

The job log ends with a summary: one line per attempt (PASS / FAIL, duration, largest peak GPU memory over the ranks)
and the rung that worked. That rung is saved to `$MILABENCH_BASE/slurm/glm5-full-rung.50n` / `.64n`, where steps 2-6
read it. If nothing fits on the 50-node set, G1, G2, G3 and G5 run with frozen experts instead.

With offload, each node keeps the optimizer state of its 4 GPUs in RAM (up to roughly 200-400 GB at 100 %): check the
nodes' memory in the log.

### Steps 2-6: the G runs

```bash
./02_g1_aligned.sh                 # pass 1
./02_g1_aligned.sh pass2           # pass 2, same nodes
./all.sh                           # everything, back to back
./all.sh 1 4                       # pass 1 of G1-G4 only
```

Full training uses the rung found in step 1 for that node set. To choose it yourself:

```bash
FULL=offload1 ./03_g2_crossed.sh   # '' | offload025 | offload05 | offload1 | balanced
FULL=none ./05_g4_all_nodes.sh     # frozen experts
```

`balanced` does not exist for G4. Any rung other than "as configured" changes what is measured: write it in the
summary's `deviations` (optimizer on the CPU: slower step; balanced layout).

The two node sets are decided separately. If full training fits on the 64-node set but not on the 50-node set, G4 runs
full and G1 frozen, and the two are no longer the same workload. To compare G4 with G1, run G4 frozen too
(`FULL=none ./05_g4_all_nodes.sh`).

`all.sh` runs the blocks back to back. In the window, other experiments sit between them: to follow the window's
queue, submit the numbered scripts one at a time instead.

## Settings

All optional, as environment variables:

| Variable | Default | Meaning |
|----------|---------|---------|
| `NODES` | 66 | Nodes of the allocation. Each run picks its own 50 or 64 nodes from it. |
| `EXCLUDE` | | Hostlist to leave out (unhealthy nodes). |
| `NODELIST` | | Hostlist the job must use. |
| `AFTER` | | Job id to wait for (`afterany`: runs even if that job failed). |
| `TIME` | 03:00:00 (step 1), 01:00:00 (steps 2-6) | Wall time per job. |
| `RESERVATION` / `ACCOUNT` | `MILABENCH2026` / `def-bengioy_gpu` | |
| `MILABENCH_BASE` | `/home/delaunay/links/scratch/shared` | Venv, data, runs and logs. |
| `WORKDIR` | `$MILABENCH_BASE/src` | The milabench checkout the job runs. |
| `BRANCH` / `UPDATE` | `megatron` / 1 | The checkout is fast-forwarded to `origin/$BRANCH` before submitting (`UPDATE=0`: fail instead). |
| `PREPARE` | 1 | 0 skips `milabench prepare` in the job. |
| `NO_TAIL` | | 1: do not follow the log. |
| `TEST_ONLY` | | 1: `sbatch --test-only`, shows where and when the job would start, submits nothing. |

Extra `sbatch` arguments go after `--`, e.g. `./02_g1_aligned.sh pass1 -- --qos=normal`.

### Metrics and wandb

On by default for every step:

| Variable | Default | Meaning |
|----------|---------|---------|
| `BENCHMATE_RECORD_FINE` | 1 | Fine-grained step timings (work / data wait per batch). |
| `BENCHMATE_RECORD_OVERHEAD` | 1 | Timing overhead of the instrumentation itself. |
| `BENCHMATE_RECORD_LAPS` | 1 | Lap timings. |
| `BENCHMATE_PUSH_EVERY` | 10 | Flush the queued step events every N steps (one GPU sync each time). |
| `BENCHMATE_WANDB` | 1 | Push the metrics to wandb. |
| `BENCHMATE_WANDB_ALL_RANKS` | 1 | One wandb run per rank (200 or 256 per benchmark), grouped by milabench run. 0: rank 0 only. |
| `BENCHMATE_WANDB_PROJECT` | `milabench-glm` | wandb project. |
| `WANDB_DIR` | `$MILABENCH_BASE/wandb` | Each job keeps its local wandb files under `$WANDB_DIR/<job-name>.<jobid>/`. |

wandb logs online. The compute nodes have no direct internet, so the job runs `module load httpproxy` and the runs go
through the cluster's proxy. The API key is not set by these scripts: it comes from your environment or `~/.netrc`
(run `wandb login` once on a login node). Submitting warns if neither has one.

To log without the network, set `WANDB_MODE=offline`. The job log then ends with the `wandb sync` command to run
from a login node.

### Run record (`run_start`)

Each run writes one `run_start` event right after start-up, in the format of `scripts/analysis/FORMAT.md`
(`run`, `placement`, `workload`), so a run that fails later still says what it was. The scripts fill it:

| Variable | Default | Goes to |
|----------|---------|---------|
| `CAMPAIGN` | `glm5-<MMDD>` | `run.campaign`: groups the runs of one session. Set the same value for the whole window. |
| `EXCLUSIVE_CLUSTER` | 0 | `run.exclusive_cluster`: set 1 in the window, when the cluster is ours. |
| `HEALTH_DIR` | | `placement.health_check`: where the health-check results are (e.g. `srygaard:/scratch/.../results/25045100`). |
| `EXCLUDE_REASONS` | | `placement.excluded_nodes`, as JSON `{"rg32101": "intra-node all-reduce 281 < 90% of 336 GB/s"}`. Without it, the `EXCLUDE` nodes are listed with "no reason given". |

Set by the scripts themselves: `run.variant` (G1-G5, or the benchmark name for step 1), `run.repetition` (pass1 = 1,
pass2 = 2), `placement.family` (EP for G1-G3 / G5, ALL for G4). The job saves `scontrol show topology` when it starts
(`$MILABENCH_BASE/slurm/topology.<jobid>.txt`): the node placement uses that snapshot, and the record gets
`placement.leaves` and `topology_snapshot_epoch` from it.

## Where things are

- Job logs: `$MILABENCH_BASE/slurm/<job-name>-<jobid>.out` (e.g. `glm5-g1-pass1-1234.out`, `glm5-check-full-g1-1234.out`)
- Run data: `$MILABENCH_BASE/runs/<job-name>.<jobid>.<entry>/`, one folder per benchmark. The other ranks' metrics
  (peak memory per pipeline stage, straggler data) are in the rankstore files next to it.

What to look at in a job log:

- `expert groups crossing a switch: X / Y`: must match the table for that run.
- no `WARNING: the hosts run in another order than the placement`: otherwise the placement is meaningless.
- `a2a_time_crossing` vs `a2a_time_aligned`, `rate`, `peak_memory_*`, `time_to_first_step`.

## Files

- `_common.sh`: defaults, login-node checks (checkout up to date, venv, data) and the `submit` function.
- `job.sbatch`: the job. Runs `milabench prepare` once, then one `milabench run` per benchmark, and prints the summary.
- `_g.sh`: shared by steps 2-6 (which benchmarks, which full-training rung).
- The benchmarks themselves are in `benchmarks/megatron/dev.yaml`.
