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
- `config_trillium.sh`: replaces their `config_hpg.sh` (their HCA and socket names do not exist here).
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
2. One compute node: `SCRIPT=convert.sbatch submit.sh convert-val 1 0:30:00 SET=val` converts the validation set
   (3 shards, one per worker) and times a 21.7 GB shard; the training set is 110 such shards, so its time is about
   ceil(110 / WORKERS) shards plus `energon_prepare.py`. Then `SET=train WORKERS=<from the subset>` in one 1-hour job if
   that fits, else a request row. It ends with `data.sh check-conversion`: every file of `expected-mounts.csv` by size
   (230 files, 2,450,225,588,745 bytes) and energon's metadata read back (train 110 shards, 1,099,776 samples; val 3,
   29,696); writes `data/conversion-checked`.
3. Login node: `data.sh cleanup` deletes `data/cc12m_preprocessed` and `data/coco_preprocessed` only if both marker
   files exist, with `diskusage_report` before and after.

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
- GPUs while busy (utilization >= 90%): 685-830 W mean per GPU, 968-985 W peak, SM clock median 1,770-1,897 MHz.
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
- GPUs while busy: 654-840 W mean per GPU, peak 977-1,076 W (one 5 s sample above 1,000 W, trig0147), SM clock median
  1,762-1,950 MHz.
- NCCL 2.28.3; inter-node through the image's net plugin ("NCCL RDMA Plugin v10", IB verbs) on `mlx5_0`, `1`, `4`, `9`
  with GPUDirect RDMA, 208 channel connections on each NIC, PXN for the GPUs without one; NVLS within the node; SHARP
  collnet plugin loaded (whether CollNet is used is not logged). The 2-node job did not log NCCL.
- Startup: job start to `init_start` 56 s (prolog and checks 26 s, container and imports 30 s); `init_start` to
  `run_start` 316 s (Florida 186-189), by rank 0's log: distributed init and model build 84 s (2 nodes: 89), data and
  samplers 44 (15; the larger mock set may account for part), training warm-up step 0 104 (97), step 1 59 (61),
  validation warm-up 21 + 3 (21 + 3). Florida's logs have no breakdown.
