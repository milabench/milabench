# FLUX.1 MLPerf training on Trillium B200 (Florida's submission code)

MLPerf Training v5.1's FLUX.1 benchmark with the University of Florida's code (`mlcommons/training_results_v5.1`,
`University-of-Florida/benchmarks/flux1/implementations/{b200_nemo,hpg.dgxb200x8}`, commit `441d65f`), to compare with
their B200 results at the same node counts. Brief: `~/projects/tri-work/ops/FLUX_BRIEF.md`.

Paths: `F=$SCRATCH/trillium/new/flux`. `src/` this repository (branch `trillium-new`), `florida/` sparse checkout of
Florida's code, `impl/` their implementation directory (`b200_nemo` plus the `hpg.dgxb200x8` files, as their README
says), `images/` the base and `flux.sif`, `apptainer/` cache and build tmp, `data/` the dataset, `runs/` MLPerf logs and
GPU telemetry per job, `slurm/` job output.

## Files

- `flux.def`, `build.sh`: the image, NVIDIA's Dockerfile as an Apptainer definition (differences in its header), built
  on tri-dm1 with `--fakeroot` (no subuid here: root-mapped user namespace).
- `flux.sub`: Florida's `flux.sub` with the Trillium changes marked `Trillium:` (core dumps off and `--propagate=CORE`
  on every srun; settings as arguments and configs sourced in the job, for `.sbatch --export=NONE`; `RUNSUB_DIR` is
  `impl/`; `PYTHONPATH` unset; `nvidia-smi` telemetry on every node).
- `config_trillium.sh`: replaces their `config_hpg.sh` (their HCA and socket names do not exist here). Two Trillium
  launch differences from Florida: `SRUN_CPUS_PER_TASK=12`, each rank a node's share of the 96 cores (their srun gives
  Slurm's default, 1 CPU per task, binding unknown there; ours untested without it), and `TRITON_HOME` on `/tmp`.
- `submit.sh NAME NODES TIME [NAME=value ...] [-- sbatch args]`: one job `nw-flux-NAME` (`SCRIPT=convert.sbatch`: the
  data conversion instead of `flux.sub`).
- `data.sh download|check-download|check-conversion|cleanup`, `convert.sbatch`: the data (below).

## Setup (once, on tri-dm1)

    F=$SCRATCH/trillium/new/flux
    git clone --filter=blob:none --no-checkout --sparse https://github.com/mlcommons/training_results_v5.1.git $F/florida
    git -C $F/florida sparse-checkout set University-of-Florida/benchmarks/flux1/implementations
    git -C $F/florida checkout 441d65f4445cd59687466533fd9f55399136e7a4
    I=$F/florida/University-of-Florida/benchmarks/flux1/implementations
    mkdir -p $F/impl && cp -r $I/b200_nemo/. $I/hpg.dgxb200x8/* $F/impl/
    git clone --branch trillium-new --single-branch https://github.com/milabench/milabench.git $F/src
    APPTAINER_CACHEDIR=$F/apptainer/cache APPTAINER_TMPDIR=$F/apptainer/tmp \
        apptainer pull $F/images/pytorch-25.09-py3.sif docker://nvcr.io/nvidia/pytorch:25.09-py3
    nohup bash $F/src/scripts/trillium/flux/build.sh > $F/logs/build.log 2>&1 &
    # the image's own NCCL 2.28.3 to the host, bound and put first on LD_LIBRARY_PATH by flux.sub as Florida does:
    # the Dockerfile removes the base's libnccl2 and installs 2.28.3 in /usr/local/lib without ldconfig, so
    # `import torch` fails in the image without it
    mkdir -p $F/nccl/lib && apptainer exec -B $F/nccl/lib $F/images/flux.sif \
        cp -a /usr/local/lib/libnccl.so /usr/local/lib/libnccl.so.2 /usr/local/lib/libnccl.so.2.28.3 $F/nccl/lib/

## Florida's results, recomputed from their MLPerf logs

`results/hpg.dgxb200x8.n<N>/flux1/result_{0..9}.txt` (MLLOG lines only: no NCCL or system output), 10 runs each, all
`success`, 2025-10-03 to 05. Time to train is `run_start` to `run_stop`, the score the olympic mean (drop best and worst).
Step time: `tracked_stats.train_step_time` (each 262,144-sample block between evaluations), median over the runs.
Init (`init_start` to `run_start`, untimed: CUDA graph and warm-up steps) 182-251 s, median 186-189 s at every size.

| nodes | GBS | LR, warm-up | TTT min (olympic; min-max) | samples to 0.586 | step s | samples/s | per GPU |
|---|---|---|---|---|---|---|---|
| 2 | 512 | 2e-4, 1,600 | 173.4 (170.5-178.4) | 8.13-8.39 M | 0.620 | 825 | 51.6 |
| 4 | 1,024 | 1.85e-4, 0 | 93.5 (90.3-98.6) | 8.65-9.44 M | 0.617 | 1,661 | 51.9 |
| 8 | 2,048 | 1.85e-4, 0 | 60.1 (56.1-63.1) | 10.5-11.8 M | 0.630 | 3,249 | 50.8 |
| 9 | 2,304 | 1.85e-4, 0 | 54.5 (52.3-56.1) | 11.0-11.8 M | 0.630 | 3,656 | 50.8 |
| 16 | 4,096 | 1.85e-4, 0 | 47.7 (47.0-49.1) | 17.0-17.8 M | 0.650 | 6,302 | 49.2 |

MBS 32 per GPU everywhere, MXFP8, DDP with the distributed optimizer, full-iteration CUDA graphs; evaluations every
262,144 samples on 29,696, 4% of the time to train. Their network settings (configs, not logs): `NCCL_NVLS_ENABLE=1`,
`NCCL_GRAPH_REGISTER=0`, `NCCL_LOCAL_REGISTER=0` (NVIDIA's `config_common.sh`); `NCCL_IB_HCA` 8 HCAs,
`NCCL_SOCKET_IFNAME=bridge-1145`, UCX (`config_hpg.sh`); their own NCCL v2.28.3-1 bound from the host. System JSON:
InfiniBand, driver 570.148.08 with CUDA 13.0 forward compatibility (580.82.07), 1,000 W air-cooled B200.

## Data

MLCommons R2 (`mlc-r2-downloader.sh`: wget per file, `--continue`, md5 check), public:

| set | files | bytes | samples |
|---|---|---|---|
| `cc12m_preprocessed` | 4,764 (4,762 arrow of 500 MB) | 2.381 TB | 1,099,776 |
| `coco_preprocessed` (validation) | 131 | 64.3 GB | 29,696 |
| `empty_encodings` | 2 | 2.1 MB | |

Conversion (`to_webdataset.py`, then `energon_prepare.py`) writes 10,000-sample tars of the same bytes: 230 files and
2,450,225,588,745 bytes (`expected-mounts.csv`, which the job's mount check compares). Peak 4.9 TB with both copies;
2.45 TB once the raw sets are deleted. Probe from tri-dm1 (256 MiB, to /dev/null): 30 MB/s per stream, 92 MB/s with 4.

Steps (NVIDIA's `download.sub`, each where it can run; the converter, energon and the check import torch, so the image
runs with `nccl/lib` bound, as in `flux.sub`):

1. tri-dm1: `data.sh download` (started 2026-11-08 23:02, 4 streams, 86 MB/s), then `data.sh check-download`: per set
   the files present against the downloader's md5 list, every size against the bucket's (HEAD), and the downloader's
   own `md5sum -c` passed for the 3 sets; writes `data/download-checked`.
2. One compute node, `convert.sbatch` (`SCRIPT=convert.sbatch submit.sh NAME 1 TIME SET=...`): `SET=time` converts
   the first 4, then 16 training shards on as many workers with NVIDIA's `process_shard` (the converter has no shard
   limit) into `data/timing`, deleted after, to choose WORKERS; `SET=train WORKERS=<n>` converts the training set (started
   while the downloader still verified md5, coordinator 2026-11-09); `SET=val` converts the validation set, then
   `energon_prepare.py` (it indexes both sets and skips once `.nv-meta/dataset.yaml` exists, so it comes last), the two
   empty encodings, and `data.sh check-conversion`: every file of `expected-mounts.csv` by size (230 files,
   2,450,225,588,745 bytes) and energon's metadata read back (train 110 shards, 1,099,776 samples; val 3, 29,696); writes
   `data/conversion-checked`. Each job first removes that marker. If a job is killed during `energon_prepare.py` or the
   check, run its last lines by hand rather than resubmitting.
3. Login node: `data.sh cleanup` deletes `data/cc12m_preprocessed` and `data/coco_preprocessed` only if both marker
   files exist, with `diskusage_report` before and after.

Done 2026-11-09: download 23:02-12:06 (cc12m files by 07:30 at about 80 MB/s, then the downloader's sequential md5 at
about 18 files/min, 150 MB/s); check-download 13:05 (4,764 + 131 + 2 files, sizes as the bucket's, md5 4,897 OK).
Shard timing (job 1273949's predecessor 1273938): 4 shards on 4 workers 119.7 s, 16 on 16 127.0 s (2.73 GB/s), so
WORKERS=16; training set 12:53 (job 1273949, while the md5 step ran, which kept its rate); validation set 2:24,
`energon_prepare.py` 12:10, check-conversion passed 13:20 (job 1276286): 230 files, 2,450,225,588,745 bytes, 0
mismatches; train 110 shards, 1,099,776 samples; val 3, 29,696. `.nv-meta`: 5 files, 555 MB. Raw sets deleted 13:21:27;
`diskusage_report` /scratch 9,138 GiB and 2,005K files before, 6,861 GiB and 2,000K at 13:23 (it lagged 1 min).
## Status (step 1, 2026-11-08)

- Image: `flux.sif` 10.9 GB (base 8.15 GB), built on tri-dm1 in 1 h 10 min (20:43-21:53). Inside: NeMo `2182063e`
  (25.09-alpha.rc1), Megatron-Core `d3391901` (0.15.0rc5), TE 2.8.0+40c69e75, NCCL 2.28.3 (loaded at run time from
  `nccl/lib`), cuDNN 9.13.1.26, torch 2.9.0a0 nv25.09, transformers 4.53.3, lightning 2.4.0, energon 7.1.0, datasets 4.1.1.
- Job 1271975 (trig0145, `USE_SYNTHETIC_DATA=1 MAX_STEPS=300`, 1 node): ranks up, NCCL init; Triton could not write
  `~/.triton`, torch.compile fell back to eager, rank 5 out of memory. Fixed: `TRITON_HOME` (42a37831).
- Job 1272006 (same, after the fix; cancelled at 6:26): container start to `init_start` 30 s; compile fine; ranks 3, 5, 6
  out of memory in the first CUDA-graph warm-up step, 170 GiB allocated of 178. Mock samples have the real shapes
  (256 text + 256 image tokens). Not determined; likely 1 node x MBS 32 does not fit: the distributed optimizer splits
  the fp32 state over 8 GPUs instead of 16 (about +9 GB per GPU, estimated), and every Florida and NVIDIA config has at
  least 16 GPUs. So the smallest faithful test is 2 nodes.
- Untested: the data steps, real data, more than 2 nodes, `VERIFY_MOUNTS`, the whole MLPerf run.

## Step 2 (2026-11-08): 2 nodes, Florida's 2-node config, synthetic data

    scripts/trillium/flux/submit.sh synth2 2 0:25:00 USE_SYNTHETIC_DATA=1 VERIFY_MOUNTS=0 MAX_STEPS=300 -- --nodelist=trig0145,trig0151

Job 1272079 (trig0145, trig0151, completed, 9:57; GBS 512, LR 2e-4, 1,600 warm-up steps as logged):
- Step time 0.647 s (median of 13 ten-step intervals between `train_loss` events, 0.630-0.653) against Florida's 0.620
  at 2 nodes (+4.4%), 791 samples/s, 49.5 per GPU. The mock set holds 9,728 samples, so an epoch ends and Lightning
  validates every 19 steps (15 evaluations in 300 steps; intervals that hold one take 0.86-0.90 s, and MLPerf's
  `train_step_time` per 19-step block reads 0.71-0.74 s). Florida's real data evaluates every 512 steps.
- Memory: peak `memory.used` 170.2-171.6 GiB (174,320-175,704 MiB) of 179.1 per GPU (nvidia-smi every 5 s).
- Startup: job start to `init_start` 52 s (Florida's prolog 9 s, container and imports 31 s); `init_start` to
  `run_start` 287 s (`warmup_time` 286.8, Florida's median 186-189: +100 s, cause not determined; the Triton and
  inductor caches start empty in each job's /tmp).
- GPUs while busy (utilization >= 90%, warm-up included): 685-830 W mean per GPU, 968-985 W peak, SM clock median
  1,770-1,897 MHz. Over the training window only (`run_start` to `run_stop`, 15 evaluations in it): 824-857 W.
- Compliance checker: fails only on the quality target, as expected for 300 steps. NCCL's transport is not logged.

## Step 3 (2026-11-09): 4 nodes, Florida's 4-node setup, synthetic data

    scripts/trillium/flux/submit.sh synth4 4 0:25:00 USE_SYNTHETIC_DATA=1 VERIFY_MOUNTS=0 MAX_STEPS=300 \
        EXTRA_ARGS=+data.num_train_samples=1048576 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET \
        -- --nodelist=trig0145,trig0147,trig0148,trig0151

Job 1272764 (trig0145, 0147, 0148, 0151, one bandwidth group; completed, 10:10). As Florida's n4 logs: GBS 1,024
(MBS 32), LR 1.85e-4, no warm-up (`config_DGXB200_scale.sh`). Differs from Florida: the mock set is enlarged from NVIDIA's
10,000 samples to 1,048,576 (the mock data module's `num_train_samples`), so evaluation comes at step 256 as with their
real data (every 262,144 samples) instead of every 9 steps.
- Step time 0.652 s, MLPerf's `train_step_time` for the first 256-step block, against Florida's 0.617 (+5.6%); 1,571
  samples/s, 49.1 per GPU (Florida 51.9). Ten-step intervals between `train_loss` events: median 0.653 (28, 0.645-0.661).
- Memory: peak `memory.used` 164.6-166.6 GiB (168,544-170,566 MiB) of 179.1 per GPU.
- GPUs while busy (warm-up included): 654-840 W mean per GPU, peak 977-1,076 W (one 5 s sample above 1,000 W, trig0147),
  SM clock median 1,762-1,950 MHz. Over the training window only: 901-934 W per GPU.
- NCCL 2.28.3; inter-node through the image's net plugin ("NCCL RDMA Plugin v10", IB verbs) on `mlx5_0`, `1`, `4`, `9`
  with GPUDirect RDMA, 208 channel connections on each NIC, PXN for the GPUs without one; NVLS within the node; SHARP
  collnet plugin loaded (whether CollNet is used is not logged). The 2-node job did not log NCCL.
- Startup: job start to `init_start` 56 s (prolog and checks 26 s, container and imports 30 s); `init_start` to
  `run_start` 316 s (Florida 186-189), by rank 0's log: distributed init and model build 84 s (2 nodes: 89), data and
  samplers 44 (15; the larger mock set may account for part), training warm-up step 0 104 (97), step 1 59 (61),
  validation warm-up 21 + 3 (21 + 3). Florida's logs have no breakdown.

## Step 4 (2026-11-09): 4 nodes, real data, Florida's 4-node config unchanged

    scripts/trillium/flux/submit.sh real4 4 0:30:00 MAX_STEPS=300 -- --nodelist=trig0145,trig0147,trig0148,trig0151

Job 1276381 (completed, 11:01; GBS 1,024, LR 1.85e-4, no warm-up, seed 12291; data/energon, `VERIFY_MOUNTS=1`):
- Mount check OK on all 4 nodes: 230 files, 2,450,225,588,745 bytes.
- Step time: first 256-step block (`train_step_time`) 0.665 s against Florida's first block 0.622-0.634 (median
  0.623, +6.7%; their all-run median 0.617) and synthetic 0.652; steps 30-300 without the evaluation, ten-step intervals
  between `train_loss` events: median 0.650 s, 1,575 samples/s, 49.2 per GPU (Florida 51.9). Steps 1-20 took 0.78 and
  0.74 s (data workers starting); afterwards no data-loader stall is visible at 10-step resolution.
- Evaluation at step 256 (262,144 samples): 0.6928 in 8.9 s; Florida's n4 at the same step 0.690-0.699, 8.6-12.8 s.
  Training loss 1.80 (step 10), 0.84 (100), 0.70 (250), 0.59 (300).
- Init 325 s (Florida 186-189; synthetic 316): distributed init and model build 118 s, data module 21, training
  warm-up step 0 106, step 1 57, validation warm-up 19 + 4. Job start to `init_start` 61 s.
- Memory: peak `memory.used` 166.9-168.6 GiB of 179.1. GPUs while busy (warm-up included): 668-766 W mean, 988 W peak,
  SM clock median 1,725-1,957 MHz. Over the training window only: 897-916 W per GPU, >= 950 W in 12-32 % of samples,
  SM clock median 1,687-1,815 MHz.
- Errors: none in training. Expected failures: the cache drop (no sudo, Florida's `|| true`), the compliance checker
  (quality target not reached in 300 steps).

## Proposed rows (full runs to the quality target 0.586, NEXP=1, Florida's config for the node count, real data)

Estimates assume our step time stays 5-7% above Florida's (measured at 2 and 4 nodes). The run's own cap is Florida's
`WALLTIME_RUNANDTIME` (srun `--time`). One bandwidth group each. Startup to the first step is about 6.5 min.

| row | nodes | `--time` | Florida TTT min (min-max) | ours est. | first eval (Florida) | first-block step: Florida, ours est. | run cap |
|---|---|---|---|---|---|---|---|
| 8 | 8 | 1:45:00 | 60.1 (56.1-63.1) | 59-68 | 0.747-0.864 | 0.635-0.665, 0.67-0.71 s | 120 min |
| 16 | 16 | 1:30:00 | 47.7 (47.0-49.1) | 50-60 | 0.876-1.312 | 0.679-0.707, 0.71-0.76 s | 70 min |
| 2 | 2 | 3:45:00 | 173.4 (170.5-178.4) | 179-191 | 0.710-0.718 | 0.620-0.633, 0.65-0.68 s | 215 min (`RUN_MINUTES=215`) |
| 4 | 4 | 2:15:00 | 93.5 (90.3-98.6) | 95-106 | 0.690-0.699 | 0.622-0.634, 0.66-0.68 s | 220 min |

    scripts/trillium/flux/submit.sh run<N> <N> <time> -- --nodelist=<one bandwidth group>

Working looks like: `mountcheck OK ... 230 files` on every node; `init_start` about 1 min after the job starts,
`run_start` about 5.5 min later; `train_loss` every 10 steps; the first evaluation at 262,144 samples within Florida's
range; `run_stop` with `status: success`. Cancel: no `train_loss` 12 min after the job starts; first-block
`train_step_time` above Florida's first-block maximum plus 15% (check the nodes' power in `runs/<job>/gpu-*.csv`); a NaN
loss; no success by 1.15 times Florida's largest sample count (8: 13.5 M, 16: 20.5 M, 4: 10.9 M).

2 nodes, own rules: Florida's 2-node config caps the training step at 200 min (`WALLTIME_RUNANDTIME`), and Florida
needed 31-32 evaluation blocks (512 steps each) at 5.50-5.58 min per block, so ours takes 5.78-5.97 (est.), 179-191
min for 31-32 blocks and up to 197 for 33. `RUN_MINUTES=215` raises the cap, which `--time 3:45:00` still holds
(6.5 min to the first step, 215 min of run, the compliance checker). Cancel: first-block `train_step_time` above
0.73 s (33 blocks would then need 208 min); no success at the evaluation of block 34 (8,912,896 samples, 1.06 times
Florida's largest, 8,388,608), which comes by 203 min at the estimated speed and by 214.5 min at the 0.73 s limit,
so it can fire before the cap. Florida's 1.15 times (37 blocks, 221 min) cannot.

    scripts/trillium/flux/submit.sh run2 2 3:45:00 RUN_MINUTES=215 -- --nodelist=<two nodes of one bandwidth group>

## Differences from Florida's runs, launch and logs

- Each rank gets 12 cores (`SRUN_CPUS_PER_TASK=12`) and Triton's cache is on `/tmp` (`TRITON_HOME`), config_trillium.sh.
- `RUN_MINUTES` can raise the training step's cap (`WALLTIME_RUNANDTIME`); only the 2-node row uses it (215 min
  instead of 200), a guard, not a training parameter.
- The page-cache drop fails here (no sudo; `flux.sub` keeps going as Florida's does), yet the MLPerf log still records
  `cache_clear: true`, which `flux.sub` logs right after it.
- The compliance checker runs once per node and they all write the same file (harmless: same input, same output).
- Every job writes NeMo's logs and outputs under the shared `images/tmpmounts`, so FLUX rows run one at a time.

## Collecting a full run

Time to train moves about 12% with the number of evaluation blocks (262,144 samples each) to the target, so per run
give: samples to target and blocks; Florida's time at the same block count (below); and the steady step time against
theirs. Florida, by blocks to target (`run_stop` samples / 262,144), time to train min (runs):
- 2 nodes: 31 blocks 170.5-172.8 (6); 32: 176.0-178.4 (4)
- 4 nodes: 33: 90.3-90.4 (3); 34: 93.1-93.2 (4); 35: 95.8 (1); 36: 98.5-98.6 (2)
- 8 nodes: 40: 56.1 (1); 41: 57.4-57.5 (4); 44: 61.6 (1); 45: 63.0-63.1 (4)
- 16 nodes: 65: 47.0-47.1 (2); 66: 47.8-47.9 (7); 68: 49.1 (1)

## FLUX-01 (8 nodes, job 1276503)

Experiment coordinator's run: trig[0101-0108], 13:51:06-15:05:13, completed, code 8fbd199d; Florida's 8-node config
(GBS 2,048, LR 1.85e-4, no warm-up), seed 28470; MLPerf log `runs/nw-flux-run8.1276503/261109135114974216209_1.log`,
compliance checker SUCCESS.
- Result: `run_stop` success, time to train 67.58 min; 12,058,624 samples, 46 evaluation blocks (5,888 steps); first
  evaluation 0.7679, last 0.5851. Florida: 60.1 min (olympic mean; 56.1-63.1), 40-45 blocks. Init 282 s.
- Florida at 46 blocks: none of their runs needed 46; their time per block at 8 nodes is 1.4018 min (1.3997-1.4027 over
  the 10 runs), so 46 blocks would take 64.48 min (extrapolated). Ours, 67.58 min, is 4.8 % above that.
- Steady step: 0.6603 s (median of blocks 4-46, 0.6589-0.6699; blocks 1-3: 0.688, 0.658, 0.716) against Florida's
  0.6303 (0.6294-0.6307), +4.8 %; 3,101 samples/s, 48.5 per GPU (Florida 50.8). Validation 3.28 s median (Florida
  3.2-3.3).
- The 7.5 min above Florida's 60.08 (olympic mean; its 8 runs average 42.875 blocks), extra blocks first: 64.48 - 60.08
  = 4.4 min for the extra blocks at Florida's time per block, then 67.58 - 64.48 = 3.1 min for our slower blocks. Speed
  first, pricing the extra blocks at our 1.469 min per block: 2.9 min for the speed, 4.6 for the blocks.
- Convergence against Florida's 10 runs at the same sample counts: ours lies within their spread at most blocks; below all
  of them (better) at blocks 4, 6, 8, 9, 25 and 32; above all of them at 38 (equal to their highest; 10 runs), 40 (10),
  41 (9) and 45 (4; their runs at 42-44: 5). From
  block 36 ours stays at 0.587-0.590 (0.5897 at block 40) and reaches 0.5851 at block 46; five of Florida's runs end at
  40-41 blocks, five at 44-45. So not above from early on: late, around the target. Not the data order: energon seeds its
  shuffle from the data-parallel rank and worker index only (`megatron/energon/worker.py` `worker_seed`,
  sha1(global_worker_id, seed_offset)), and NeMo's training loader builds its `WorkerConfig` without `seed_offset`
  (`nemo/collections/multimodal/data/energon/base.py:218`; both read in the image), so at a given node count the order
  is the same whatever the seed: Florida's 10 runs saw one order, and ours too if our shards hold the same samples in the
  same order (their sizes match NVIDIA's byte for byte; the content was not compared). The seed (ours 28470, theirs
  11905-11916) sets the model-side randomness: initialization, the sampled noise and timesteps, the text dropout; their
  spread of 40-45 blocks is that randomness. What could make ours systematic:
  numerics (NCCL's reduction layout over 4 NICs with PXN against their 8, driver 595 against 570, the date-pinned pip
  set; NeMo, Megatron-Core, TE, NCCL and cuDNN are their versions); not the cache drop, which changes I/O timing only.
  One run cannot tell a systematic shift from seed spread: Florida's own 10 seeds span 40-45 blocks.
- Power and clocks, per GPU (not per node), over the training window: 912-933 W mean, >= 950 W in 15-31 % of the
  5 s samples, SM clock median 1,650-1,792 MHz (lowest 1,447). nvidia-smi's throttle reasons were not logged, so the
  share of samples at the software cap is not known. The step time does not follow power: by the same method (median of
  ten-step intervals between `train_loss` events, evaluations left out), 0.650 s at 4 nodes (897-916 W) and 0.659 at 8
  (912-933 W), +1.4 %, against Florida's +2.1 % (0.617 to 0.630) from 4 to 8 nodes.
- trig0101 and trig0106 were drained by the site's health check for `slurmstepd` high CPU during the job (HW-01, open):
  not affected. During the drains (trig0106 14:22-14:27, trig0101 14:37-14:42) the blocks ran 0.659-0.663 s; the
  slowest block, 0.670 s, is block 8 at 14:08, before them.

## Step-time gap against Florida (FLUX_BRIEF.md, last section)

Job 1277931 (`nw-flux-gap1`, trig0145, trig0148, 2026-11-09 17:01-17:31, TIMEOUT): Florida's 2-node config, real data,
telemetry every second with the clock event reasons, Nsight Systems on rank 0 (`nsys_rank.sh`, 20 s window).
- Step 0.6548 s (median of ten-step intervals, steps 30-150) against Florida's 0.620 (+5.6 %). Init 282 s.
- Throttle reasons over steps 30-150 (1 s samples, 16 GPUs): SW power cap active in 88-97 % of samples on every GPU,
  mean power 922-933 W, SM clock median 1,627-1,751 MHz (lowest 1,462; maximum 1,965); no thermal, HW slowdown or power
  brake. So the GPUs run at the 1,000 W software cap most of the time; Florida's logs do not say whether theirs did.
- Profile, rank 0, 17.5 s (about 27 steps): compute kernels 600 ms per step (91.7 % of the time), NCCL kernels 319 ms
  of which 289 under compute and 30 ms alone (4.6 %), no kernel 24 ms (3.7 %: two gaps of about 4 ms or more per step,
  14 ms, and 860 shorter ones, 10 ms). NCCL: 43 reduce-scatters and 43 all-gathers per step, 16 channels, protocol
  RING_LL (unusual for messages this large), 3-4 ms each. Top compute: MXFP8 GEMMs (nvjet), elementwise, MXFP8 casts.
- The run did not finish: ending the nsys collection (`--duration`) ended rank 0 ("The target application
  terminated"), and the other ranks waited in NCCL until the time limit (22 min of 2 nodes). `nsys_rank.sh` no longer
  ends the collection mid-run.

Next (one job, `gap.sbatch`): 200 steps per run, consecutive in one 2-node job: base (NCCL INIT, TUNING, COLL logs),
`NCCL_PROTO=Simple`, `NCCL_PXN_DISABLE=1`, `CPU_SHARE=0` (Florida's srun as it is), base again; the GPUs' power and
clock limits and each node's CPU clock every 5 s. Not varied: the power cap (needs root), the driver, the image's
pip set (one image), `TRITON_HOME` (only where compiled kernels are cached: the same kernels run).

Job 1278802 (`nw-flux-gap2`, `gap.sbatch`, trig0127, trig0128, 19:11:05-19:52:53, completed, 41:48; src-dev at
b1c84012): five runs of 200 steps, about 8.3 min each, none failed. All 16 GPUs `692-2G525-0225-001`; enforced power
limit 1,000 W on every GPU (default and maximum 1,000 W), maximum SM clock 1,965 MHz. Step: median of ten-step intervals
from step 30 (17 each); cap: share of 1 s samples with the SW power cap active over the same steps, per node.

| run | step s | vs base | power cap | power per GPU | SM clock median |
|---|---|---|---|---|---|
| base (NCCL logs) | 0.6461 | | 94-96 % | 934 W | 1,695-1,710 MHz |
| `NCCL_PROTO=Simple` | 0.6443 | -0.3 % | 92-93 % | 935 W | 1,695-1,702 MHz |
| `NCCL_PXN_DISABLE=1` | 0.6422 | -0.6 % | 90-93 % | 933-935 W | 1,702-1,710 MHz |
| `CPU_SHARE=0` | 0.6457 | -0.1 % | 92-94 % | 932-934 W | 1,698-1,717 MHz |
| base again | 0.6459 | 0.0 % | 93-95 % | 933-934 W | 1,710-1,717 MHz |

- Baseline 0.646 s against Florida's 0.620: +4.2 %. No variant moves it by more than 0.6 % (one run each; the two base
  runs are 0.03 % apart). No thermal, HW slowdown or power-brake samples.
- NCCL: each reduce-scatter and all-gather moves about 17.7 M bf16 elements per rank (35 MB); NCCL chose RING_LL for
  them in 1277931, and Simple changes the step by 0.3 %: the communication is mostly under compute (profile above).
- CPU: the cores' mean clock during training drops to a median of 1,350-2,374 MHz by run and node (3,200 idle; the known
  drop under GPU power), and the step does not follow it (trig0127: 1,400 MHz in `CPU_SHARE=0`, 2,149 in base, same step).
- Boards: job 1277931 (0.6548 s) ran on trig0145 (all 8 GPUs `-000`) and trig0148 (`-001`): 1.3 % above this baseline,
  and at the same power (928-929 W) trig0145's clock median was 1,650 MHz against trig0148's 1,725. So the `-000` node
  slowed that run by about 1.3 %; it does not explain the 4.2 %. (FLUX-01's trig0107 has one `-000` GPU; the 4-node
  real-data run 1276381 held trig0145.)
- Not determined: the remaining 4.2 %. Ruled out at this level: NCCL protocol and PXN, CPU binding and CPU clocks, the
  board type (on these nodes). The step is compute (600 of 655 ms in the profile) at clocks held by the 1,000 W software
  cap 90-96 % of the time; Florida's logs carry no clock or power data, so whether their GPUs ran faster at the same limit
  (air-cooled, driver 570 with 580 compatibility) cannot be told from here. Not testable without root (clock or power
  settings), another driver, or another build of the image.
