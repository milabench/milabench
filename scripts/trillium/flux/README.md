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
- Memory: peak `memory.used` 166.9-168.6 GiB of 179.1. GPUs while busy: 668-766 W mean, 988 W peak, SM clock median
  1,725-1,957 MHz.
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
