# OF3: OpenFold3 training on Trillium B200

OpenFold3's own training code (`aqlaboratory/openfold-3`, release `0.5.1`, commit `795e45b0`, 2026-10-28, Apache-2.0;
weights and data under permissive licenses, 2026-03 release) running its published first training stage
(`examples/training_yamls/initial_training.yml`). Brief: `~/projects/tri-work/ops/OF3_BRIEF.md`.

Paths: `O=$SCRATCH/trillium/new/of3`. `openfold-3/` the checkout at `0.5.1`, `src/` this repository (branch
`trillium-new`), `venv/` (own `uv-cache/`), `data/` the subset (`s3/` in the bucket's layout, `subset/` the subset's
dataset caches and key list, `caches/` the downloaded caches), `runs/of3.<job>.<variant>/`, `slurm/`, `cache/` (the
DeepSpeed kernel build, OpenFold3's cache directory).

## Files

- `job.sbatch STEPS VARIANT...`: one run per variant on the job's nodes (the submit line is at its top); `VARIANT+tf32`
  adds `run_openfold train --use_tf32 True`. 8 ranks per node, 12 cores each, Lightning's SLURM environment; benchmate's
  node monitor per run.
- `initial_training.yml`: the published first stage on the subset; its header lists the differences.
- `variants.yml`: the kernel switches (`settings.memory.train`), the profiler (`prof`), fine-tuning stage 1 (`ft1`), the
  step-2 subset (`uniform`), merged by `merge.py` (comma-separated: `uniform,cueq+tf32`).
- `train.py`: `run_openfold train` plus one JSON line per step and rank (`steps-r<rank>.jsonl`: step time, the gap
  before it, loss, crop, peak memory). OpenFold3 itself logs only to wandb.
- `summary.py RUN [SKIP [REF]]`: the numbers below from one run (REF: step-by-step period ratio and loss difference to a
  run of the same samples).
- `prof.py TRACE`, `prof.sbatch RUN...`: kernel classes, model parts, NCCL kernels and GPU idle gaps from a profiler
  trace (variant `prof`; job.sbatch runs it after each such run).
- `data.sh` (on tri-dm2), `subset.py`, `fetch.py`: the data subsets; `mix.py`: a subset's PDB mix against the full set.

## Install (login node, once)

    O=$SCRATCH/trillium/new/of3; export UV_CACHE_DIR=$O/uv-cache UV_LINK_MODE=copy
    git clone https://github.com/aqlaboratory/openfold-3.git $O/openfold-3 && git -C $O/openfold-3 checkout 0.5.1
    git clone --branch trillium-new https://github.com/milabench/milabench.git $O/src
    uv venv -p 3.12 $O/venv && echo torch==2.15.0 > $O/constraints.txt
    uv pip install -p $O/venv/bin/python torch==2.15.0
    uv pip install -p $O/venv/bin/python -c $O/constraints.txt -e $O/openfold-3 "pytorch-lightning<=2.6.1" \
        cuequivariance==0.13.0 cuequivariance-torch==0.13.0 cuequivariance-ops-torch-cu13==0.13.0 nvidia-ml-py \
        -e $O/src/benchmate setuptools wheel ninja nvidia-cutlass
    CUDA_HOME=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Core/cudacore/13.2.0 \
        uv pip install -p $O/venv/bin/python -c $O/constraints.txt --no-build-isolation deepspeed==0.19.9

torch 2.15.0 from PyPI (CUDA 13.2, NCCL 2.30.7; the campaign runs the 2.15.0 nightly of 2026-10-05 with CUDA 13.0),
Lightning 2.6.1 (OpenFold3 pins `<=2.6.1`). OpenFold3's own environments are pixi; the `cuequivariance` extra pins
the CUDA 12 ops, so the CUDA 13 ops are installed by name. DeepSpeed's Evoformer attention is built at first use from
CUTLASS headers (`nvidia-cutlass`) with the CUDA 13.2 toolkit (`CUDA_HOME`, set by `job.sbatch`): 7.4 min on a
compute node for `sm_100`, then cached in `cache/torch_extensions`.

## Data

The training data is in the public bucket `s3://openfold3-data` (no sign-in), as individual files (no archives).
Sizes and counts from a listing of the bucket on 2026-11-09:

| part (first stage weight) | files | size |
|---|---|---|
| PDB: MSA arrays `alignment_arrays` | 208,676 | 893.8 GB |
| PDB: structures `structure_files` (`.npz` read; `.cif`, `.fasta` not) | 244,272 (732,816) | 51.8 GB (348.4 GB) |
| PDB: template structures `template_structure_arrays` | 1,199,204 | 37.5 GB |
| PDB: template hits `train_template_cache` | 123,282 | 20.1 GB |
| PDB: reference molecules, dataset caches | 68,146 + 3 | 0.2 + 1.9 GB |
| disordered PDB (0.02), `.npz` | 61,227 | 6.3 GB |
| long distillation monomers (0.495): about 16M monomers, 3 files each | about 48M (est.) | about 38 TB (est., 2.4 MB per monomer measured on 800) |
| short distillation monomers (0.005) | 1,291,206 | 253.8 GB |
| RNA distillation (0.05) | 253,562 | 55.3 GB |

PDB weighted (0.5) is about 1.0 TB in 1.64M files; with the long monomers the whole first stage cannot fit under the
new workloads' 10 TB.

Step-1 subset (`data.sh`, 2026-11-09 05:32): 106,485 files, 9.08 GB. 150 PDB entries drawn without replacement by their
summed datapoint weight (AF3 SI Algorithm 1, OpenFold3's defaults): they hold 32,098 chains and interfaces, the
datapoints OpenFold3's sampler draws from, of which 25,918 are in the 17 entries with 100 or more chains (ribosomes,
capsids; 5tga alone 1,771 chains, most of them ligands); median 7 chains and 9.5 interfaces per entry. The first 800
long monomers of the distillation cache (the dataset reads them in order), 400 RNA monomers drawn uniformly, all
reference molecules. Short monomers and disordered PDB are left out (2.3 % of samples).

What one sample reads and computes, as published: a crop of 384 tokens (contiguous, spatial or spatial-interface), an
MSA of up to 16,384 rows (median 10,241 in step 1), up to 4 templates (a random number), RDKit conformers for every
residue and ligand, 48 diffusion samples per trunk pass, 0-3 recycles drawn per step (the same on all ranks), a
20-step rollout without gradient for the confidence heads. Every sample is featurized from scratch: OpenFold3 keeps no
feature, crop or MSA cache (the only caches are an SDF parse cache inside one sample and a residue lookup; workers are
not persistent). What does repeat is the files: the subset's 9 GB stays in a node's page cache after the first read.

## Step 1: one node (2026-11-09, jobs 1273594, 1273611, 1273674)

`job.sbatch 100 published` (trig0145), `job.sbatch 40 cueq ds cueq-ds triton` (trig0151) and
`job.sbatch 40 published+tf32 cueq+tf32 cueq-ds+tf32` (trig0152); each run takes about 4 min before its first step
(imports, dataset caches, 128 workers; plus 7.4 min the first time DeepSpeed's kernel is built), then 4-6 s per step.
All 8 GPUs, batch 1 per GPU,
bf16-mixed, 16 workers per GPU. Same seed, so every run trains on the same sample at each step: the variants compare
step by step. Throughput by wall clock: (t of the last step - t of step 6) / steps between, from rank 0's step ends
(`summary.py`); the median step period beside it; ratio: the variant's wall clock over published's over the same steps
6-40 (published: 5.314 s there). Peak memory over all ranks.
Power, clocks and use: benchmate's node monitor over the same steps.

| run (job) | steps | s per step, wall clock (median, p10-p90) | samples/s per GPU | ratio | peak alloc / reserved | loss, mean of ranks |
|---|---|---|---|---|---|---|
| published (1273594) | 100 | 5.252 s (5.207, 4.43-6.06) | 0.1904 | 1 | 31.5 / 33.5 GiB (28.9 by step 40) | steps 1-10 12.32, 91-100 8.54 |
| `+tf32` (1273674) | 40 | 5.008 s (5.010) | 0.1997 | 0.942 | 28.9 / 30.6 | 31-40 10.38 |
| `triton` (1273611) | 40 | 5.223 s (5.113) | 0.1915 | 0.983 | 28.9 / 30.6 | 31-40 10.39 |
| `ds` (1273611) | 40 | 4.673 s (4.698) | 0.2140 | 0.879 | 28.9 / 30.5 | 31-40 10.38 |
| `cueq+tf32` (1273674) | 40 | 4.254 s (4.262, 3.62-4.79) | 0.2351 | 0.801 | 28.9 / 30.6 | 31-40 10.39 |
| `cueq-ds+tf32` (1273674) | 40 | 4.229 s (4.199) | 0.2364 | 0.796 | 28.9 / 30.6 | 31-40 10.39 |
| `cueq`, `cueq-ds` (1273611) | 0 | fail in the first backward | | | | |

- Kernels on B200 (`sm_100`): all run. The published config uses none of them (plain PyTorch triangle attention and
  multiplication, FP32 matmuls without TF32). cuEquivariance's triangle attention refuses an FP32 backward unless TF32
  is on ("Full precision FP32 backward pass for triangle attention is not implemented yet"), so it needs
  `--use_tf32 True`, itself 6 % faster; cuEquivariance then takes another 15 % (0.801 against FP32). DeepSpeed's
  Evoformer attention: 12 %; with cuEquivariance it adds little (0.796) (cuEquivariance takes the calls first). OpenFold3's
  Triton kernels: 2 %, and they warn that their tuning is AMD's. cuEquivariance falls back to PyTorch for crops of
  100 tokens or fewer and for head widths it does not handle (which calls fall back: not measured, the step-2 profile).
  Losses agree to 0.01 at step 40; peak memory does not change.
- Data loader: not starved at the published 16 workers. The gap before each step (waiting for the batch, plus the EMA
  update and Lightning's hooks) is 0.115 s median, 3.8 % of the time (mean over ranks; 4.4 % worst rank); Lightning's
  `train_dataloader_next` 0.255 s mean over 100 calls, the first included. CPU use 17 % of the 96 cores, 6-7 busy
  cores per socket. 800 samples in 100 steps, 485 distinct entries, read cold the first time on each node.
- GPUs: 620 W mean, 888 W max, SM clock median 1,965 MHz, utilization 82 % (published); 485 W and 65 % with
  cuEquivariance: the GPUs wait more as the kernels speed up (on what: the step-2 profile). No power cap.
- CPU clocks: busy cores 3,200 MHz median, 3,133 minimum: no drop at this GPU power (about 5 kW per node).
- Steps vary with the sample (39-384 tokens, up to 7,653 atoms, 0-16,384 MSA rows) and with the recycles drawn
  (0-3, the same on all ranks); every rank waits for the slowest at the gradient all-reduce.

## How training is distributed

Lightning DDP (`DDPStrategy(find_unused_parameters=False)`, no DeepSpeed config in the published stage), one sample
per GPU. The training config clips gradients per sample (`per_sample_clipping`, default), so the step is manual: the
backward runs under `no_sync`, then `grad_manager.sync_and_average_grads` copies the gradients out of its accumulator,
all-reduces a sample count (then `.item()`), the per-parameter sample counts (one int64 per parameter tensor) and all
gradients as one flat FP32 buffer (368,293,788 parameters, 1.473 GB), then divides each parameter's gradient in a
Python loop. Grad-norm metrics sync through torchmetrics. Nothing of it overlaps the backward. Overlap is a
configuration option the published setup leaves off: `settings.gradient_clipping.per_sample_clipping: false` switches
to Lightning's automatic optimization, where DDP's bucketed all-reduce runs during the backward, and clips the
summed gradient instead of each sample's (not run: it changes the training).

## Step 2: 1, 2 and 4 nodes (2026-11-09)

### The data mix

The step-1 subset drew its 150 PDB entries by summed weight, without replacement. `mix.py` weights every chain and
interface as OpenFold3's sampler does (AF3 SI Algorithm 1) and compares the PDB part of a subset with the full training
cache (both caches read from the bucket into memory, 1.42 and 1.68 GB, nothing written). Weighted shares and
p10/p50/p90:

| | step-1 subset (150, by weight) | full (180,975 entries) | step-2 subset (700, uniform) |
|---|---|---|---|
| entry tokens before cropping | 4,420 / 22,800 / 30,300 | 216 / 806 / 15,800 | 195 / 840 / 13,900 |
| crop full (entry of 384 tokens or more) | 97.1 % | 74.6 % | 74.2 % |
| chains per entry | 24 / 103 / 578 | 2 / 8 / 133 | 2 / 7 / 71 |
| entries of 100 chains or more | 56.9 % | 11.8 % | 5.4 % |
| datapoint with a nucleic-acid chain / a ligand | 62.8 / 10.2 % | 27.1 / 45.4 % | 29.5 / 43.5 % |
| template hits (up to 4 per chain) | 0 / 4 / 8 | 0 / 4 / 8 | 0 / 4 / 8 |
| MSA arrays of the datapoint's chains (MB) | 0.2 / 6.0 / 25.2 | 0 / 3.8 / 19.6 | 0 / 4.3 / 22 |

Drawing by weight without replacement keeps every heavy entry once and then gives it its full weight, so ribosomes
and capsids took 57 % of the samples. A uniform draw lets the sampler reproduce the full mix in expectation; with 700
entries the largest assemblies are under-represented (5.4 % against 11.8 %). Step 2 runs on that draw
(`data.sh subset-u uniform 700 subset`, 07:13: 103,370 files, 9.32 GB in `s3/` after deleting the 19,059 files only
the step-1 subset read; variant `uniform`). Same 800 monomers, another 400 RNA entries.

What the mix changes:
- GPU step: next to nothing. Every sample is padded to the crop (384 tokens) and the model keeps at most 1,024 MSA
  rows (`subsample_all_msa`), so a sample's cost moves with its atoms and the recycles drawn: per rank, forward and
  backward = 3.85 s + 0.14 s per 1,000 atoms + 0.47 s per recycle (R2 0.92, 712 rank-steps of the 1-node profile run,
  profiler overhead included). Large assemblies and other PDB entries cost the same (5.17 and 5.18 s median), monomers
  5 % less. 1-node step by wall clock: 5.252 s on the step-1 subset (100 steps), 5.244 s on the uniform one (40).
- Data loader: the giant assemblies are what the loader finds hard. On the step-1 subset the 4-node job (1273810,
  trig[0101-0104]) had no first step after 9 min: 30 ranks waited in NCCL (GPUs at 100 % use, 255 W) for ranks 16 and
  23 (trig0103), whose workers were in OpenFold3's MSA processing of PDB samples (`create_main`, `calculate_profile`,
  `vstack_pad_msa_arrays`, py-spy; that output was not saved) for over 7 min; cancelled. The DataLoader hands batches out in order, so one slow
  sample holds its rank, and DDP holds every rank. On the uniform subset, per sample (`samples/`, 2,303 at 4 nodes):
  median 2.5 s, p99 18 s, max 23 s (2 nodes: p99 50 s, max 59 s); the workers are busy 3-4 % of the time. The full
  mix has twice the uniform subset's share of 100-chain entries; how often its slowest samples outlast the
  prefetch (16 workers x 2 samples, about 160 s at 5 s per step) is not measured.

### Scaling, published and cuEquivariance + TF32 (uniform subset, 40 steps, no profiler)

Wall clock, (t40 - t6) / 34 of rank 0's step ends, the median step period in brackets; per-GPU throughput against
1 node in brackets; ratio of the wall clocks.

| nodes (job, nodes) | published s per step | samples/s per GPU | cueq+tf32 s per step | samples/s per GPU | ratio |
|---|---|---|---|---|---|
| 1 (1273887, trig0145) | 5.244 (5.195) | 0.1907 | 4.284 (4.269) | 0.2334 | 0.817 |
| 2 (1273885, trig[0105-0106]) | 5.467 (5.524) | 0.1829 (95.9 %) | 4.367 (4.402) | 0.2290 (98.1 %) | 0.799 |
| 4 (1273884, trig[0101-0104]) | 5.618 (5.708) | 0.1780 (93.3 %) | 4.593 (4.536) | 0.2177 (93.3 %) | 0.818 |

`ds` (DeepSpeed's attention), 1 node: 4.617 s (4.594), 0.2166 samples/s per GPU, ratio 0.880. Losses of each variant against published on the same
samples: within 0.01 over 40 steps. The data loader's gap: 3.5-4.8 % at every size.

InfiniBand bytes sent per node and step (nodemon, timing runs), against one 1.473 GB all-reduce: 2 nodes 1.489 GB on
each node (+1.1 %); 4 nodes 1.493, 2.986, 2.986, 1.493 GB, 6.08 x 1.473 GB in all. That is NCCL's double binary tree
(each node sends the buffer once per tree it is inside of, 2 (N - 1) x 1.473 GB in all, 6 at 4 nodes; a ring would send
1.94 x 1.473 GB from every node): at 13 nodes about 3 GB from the busiest nodes per step (est.), 15 ms at 200 GB/s
per node.

### Where a step goes (profiles)

`variant prof`: PyTorch's profiler on each node's local rank 0, warm-up over logged steps 26-27 (tracing, results
discarded), traces of ProfilerStep#27-28 = logged steps 28-29, `prof.py` on rank 0's trace (4 nodes:
1273884; 2 nodes: 1273763, step-1 subset; 1 node: 1273761, 1273762). Traced steps take 7.0 s against 5.7 s (the
profiler's per-module hooks); kernel times are the GPU's own. Published, 4 nodes, per traced step:
- About 190,000 kernels, median 2.9 us, 67 % under 10 us; 1.06 s of CPU time in launch calls.
- Kernel time 4.2-4.6 s: elementwise 37 %, copies, casts, cat and indexing 18 %, layer norm 13 %, FP32 SIMT GEMMs 13 %,
  BF16 GEMMs 8 %, softmax 6 %, reductions 2-3 %, NCCL 0.5-2 %. Small, mostly memory-bound kernels rather than GEMMs;
  that this is why the GPUs draw 600 W at 1,965 MHz, unlike the LLM workloads, is inferred, not measured.
- By module (kernels inside a module's annotated range, forward and the recomputed forward of the backward;
  approximate): triangle attention 33-38 %, diffusion module 23-27 %, the rest of the pairformer 18 %, triangle
  multiplication 10-11 %, MSA module 3 %, template module 2 %.
- The all-reduce: the 1.473 GB buffer takes 17.9 and 18.2 ms (82 GB/s algorithm bandwidth), 0.3 % of the step; the
  count all-reduces 0.1-1.5 ms, the first of them also absorbing the wait for the slowest rank. The whole gradient sync
  on the last rank to arrive (train.py's timing: copies, all-reduces, per-parameter averaging) is 0.26, 0.21 and
  0.22 s at 1, 2 and 4 nodes: InfiniBand adds nothing visible over NVLink. The per-GPU loss with nodes (4.1 and 6.7 % for
  published, by wall clock) is the wait for the slowest of more samples (est.: the period grows while the sync stays flat).
- The GPU's idle time (16 % by nodemon in the timing run): in the traced steps, 0.34 s of gaps under 10 us between
  kernels, 1.8-1.9 s of 10-100 us gaps, 0.1 s of 0.1-1 ms, 0.14-0.43 s over 1 ms (`prof.py` bins gaps by length
  only). The launch calls (`trace_gaps.py`, the same traced steps) are short: median 3.9 us, 0.87-0.89 s in calls up
  to 20 us and 0.15-0.17 s in calls over 100 us per traced step. 31-35 % of the launches meet over 1 ms of GPU work
  already queued: GPU-bound stretches; between them the GPU idles in the 10-100 us gaps while the CPU issues one small
  kernel after another: launch-bound stretches. Inferred: the gaps over 1 ms are the gradient sync's copies and Python
  loop and the `.item()` syncs; measured: the all-reduce (18 ms) and the data loader's gap (0.11 s) are minor. With cuEquivariance (2 nodes) kernel time falls
  from 5.07 to 2.98 s per traced step, and GPU use from 83 to 65-68 %: the step gains only 18-20 % (wall clock).
- DeepSpeed's attention (1 node): its kernels take 0.79 s, elementwise falls from 1.73 to 0.76 s.

### Precision

The published stage trains in Lightning's `bf16-mixed`: BF16 autocast for most matmuls, FP32 weights, Adam state and
gradient all-reduce. OpenFold3 keeps some parts in FP32 under autocast: layer norms, the input embedder's atom
attention encoder (autocast to float32), the confidence heads' pairformer (`pairformer_dtype` float32), losses.
`run_openfold train` leaves TF32 off (`--use_tf32` False), so those FP32 matmuls run as SIMT SGEMM, without tensor
cores (`cutlass3x_sm100_simt_sgemm_f32`): 0.59-0.60 s per profiled step, 12-14 % of the GPU's kernel time.
`--use_tf32 True` (`torch.set_float32_matmul_precision("high")`) puts them on TF32 tensor cores (0.14 s in the
cuEquivariance profile) and sets OpenFold3's Triton dot precision to TF32: 0.942 of the published step alone.

cuEquivariance's triangle attention refuses an FP32 backward ("Full precision FP32 backward pass for triangle attention
is not implemented yet! Please set torch.backends.cuda.matmul.allow_tf32=True"): the FP32 inputs come from the
confidence heads' pairformer. This is a documented limitation, not new: cuEquivariance's docs (CUDA 12.9.1 archive)
say "Full FP32 is not supported for backward pass. Please set torch.backends.cuda.matmul.allow_tf32=True"; the
current page no longer says it, and 0.13.0 (the latest on PyPI) still raises. OpenFold3 documents cuEquivariance for
inference and does not mention TF32. So cuEquivariance in training needs TF32: a compromise against the published
numerics.

Loss over 100 steps, 1 node, same samples (step-1 subset; mean over ranks, per 10 steps):

| against published (1273594) | mean absolute difference | per 10 steps |
|---|---|---|
| published again (1273761, with the profiler) | 0.044 | 0.000, 0.000, 0.002, 0.002, 0.003, 0.014, 0.041, 0.091, 0.124, -0.035 |
| cuEquivariance + TF32 (1273761, with the profiler) | 0.034 | 0.001, 0.001, 0.004, 0.005, 0.006, 0.009, 0.006, 0.012, -0.028, -0.177 |

Two published runs drift apart as much as cuEquivariance + TF32 does from published: the difference is inside the
run-to-run noise over 100 steps (one run each).

DeepSpeed's attention needs no TF32 and agrees with published to 0.01 over 40 steps, for 0.873-0.880 of the
published step by wall clock (1 and 13 nodes); cuEquivariance + TF32 gives 0.80-0.82. For runs at scale: DeepSpeed as the fast variant without a
compromise, cuEquivariance + TF32 as the fastest, named as one.

### Fine-tuning stage 1 (crop 640, 32 diffusion samples), 1 node (1273762, trig0151, step-1 subset)

Published 12.578 s per step by wall clock (median 12.93; with train.py's gradient-sync timing, a few % slow), 0.0795
samples/s per GPU, peak
47.2 GiB allocated, GPUs 721 W and 93 % used; cuEquivariance + TF32 7.001 s (median 6.95; 0.557 of the step: triangle attention grows
with the cube of the crop), 46.1 GiB, 606 W, 85 %. Loss within 0.02 over 40 steps.

### Kernel messages

`NVRM: knvlinkSendInbandData_IMPL: Failed to send inband data: 0` in the Slurm logs' kernel messages, at the times
runs start and end: trig0145 (1273590 at 05:37; 1273594 at 05:40, 05:42, 05:51), trig0151 (1273611: 05:45, 05:48,
05:49, 05:59, 06:02, 06:03, 06:06, 06:07, 06:10, 06:14), trig0152 (1273674: 06:04, 06:06, 06:10, 06:11, 06:13,
06:16, 06:18, 06:20, 06:23). The same line is in 51 other new-workload jobs (flux, glm-fp4, glm-rl, k3-inf, k3-xn,
k3-xn2) on 32 nodes, trig0145 (1262531, 1262691, 1263799, 1269990, 1270053, 1271416, 1271975, 1272006, 1272079),
trig0151 (1264182, 1272079) and trig0152 (1264182) included. Training ran normally.

### Monomers for OF3-01 (13 nodes, 40 steps)

The long-monomer set is read in order (`sample_in_order: true`), and OpenFold3's ordered slice
(`_get_ordered_subset` in `stochastic_sampler_dataset.py`) wraps back to entry 0 once only: past 2 N monomer draws an
index runs off the set and `monomer.py` raises IndexError (an upstream limit). With 800 monomers, all four OF3-01
runs would have failed at step 32 (review of 2026-11-09). `monomers.py RANKS STEPS N` reproduces the sampler's global
list (seed 42) and counts the draws a run reaches, the 32 samples each rank prefetches included; it predicts every
monomer the 1-, 2- and 4-node runs logged (381, 306 and 611, none different). At 104 ranks, 40 steps reach 3,605
draws; with a 20 % margin the OF3-01 subset has 4,330 monomers (`data.sh subset-u2 uniform 700 4330`: 44,805 more
files, 9.98 GB). At 13 nodes it has no repeated monomer up to 54 steps and no IndexError up to 142. Variant
`uniform` now reads `subset-u2`, whose first 800 monomers, PDB and RNA entries are those of `subset-u`, so the step-2
runs read the same samples from it). Runs that reach more than 2 N monomer draws fail; longer runs need more
monomers, or `sample_in_order: false` (a compromise).

## OF3-01 (13 nodes, job 1274059, trig[0101-0113], 2026-11-09 08:22-08:59, commit 0d9484c0)

`job.sbatch 40 uniform uniform,ds uniform,cueq+tf32 uniform,prof`, 104 ranks, `subset-u2` (4,330 monomers), all four
PASS. `summary.py`: throughput by wall clock, (t40 - t6) / 34 of rank 0's step ends; in brackets, seconds per step
by wall clock, then the median step period (steps 7-40):

| samples/s per GPU | 1 node | 2 nodes | 4 nodes | 13 nodes | 13 against 1 |
|---|---|---|---|---|---|
| published | 0.1907 (5.244; 5.195) | 0.1829 (5.467; 5.524) | 0.1780 (5.618; 5.708) | 0.1687 (5.927; 5.967) | 88.5 % |
| ds | 0.2166 (4.617; 4.594) | | | 0.1934 (5.172; 5.264) | 89.3 % |
| cueq+tf32 | 0.2334 (4.284; 4.269) | 0.2290 (4.367; 4.402) | 0.2177 (4.593; 4.536) | 0.2094 (4.775; 4.695) | 89.7 % |

The three scale alike: 93.3 % at 4 nodes for published and cueq+tf32, 88.5-89.7 % at 13. In all: 17.55, 20.11 and
21.78 samples/s. `ds`'s first step took 118 s (its extension loaded by 104 ranks). Peak 33.6 GiB
allocated, 35.7 reserved; no NaN; the three timing runs' losses agree to 0.01 over the last 10 steps.

- Data loader: the gap before a step 4.3-5.3 % of the time (mean over ranks; worst rank 5.2-6.5 %). Featurization per
  sample (7,487 per run): median 2.5-2.6 s, p99 19-34 s, max 32-57 s; the workers busy 3-4 % of the time. Each run
  skipped the same 2 samples (OpenFold3's debug mode replaces a failing sample with a random other one, in the
  worker): monomer MGYP003513374722, whose `alignment.npz` is not a valid zip in the bucket itself (1,835,008 bytes
  there and here, same md5: cut short upstream; `check_npz.py` over all 80,004 `.npz` of the subset finds only this
  one), and PDB 6x0v chain 1 ("No resolved atoms"). Cost: one failed attempt
  plus one more featurization in a worker, inside the prefetch.
- Profile runs' step statistics leave out steps 26-30 (summary.py): 26-27 the profiler's warm-up (tracing on,
  results discarded; per-step period 1.05 and 1.19 x the timing run's at 13 nodes), 28-29 traced, and 30 the wait for
  the trace export (the profiling ranks' step 29 takes 26 s; the other ranks wait up to 20.3 s before step 30). Check
  at 13 nodes, mean over the kept steps: rank 0's period 6.217 s against the slowest rank's compute and loader wait
  plus the fastest rank's sync 6.218 s (6.217 against 6.219 leaving out 28-30 only; 6.201 against 6.813 with step 30
  kept). Every profile run uses this exclusion; the check also holds at 4 nodes (5.861 against 5.861 s) and on the
  H100 (7.127 against 7.132 s).
- Gradient sync (profile run, last rank to arrive): 0.221 s, as at 4 nodes (0.219). The 1.473 GB all-reduce kernel:
  18.6 and 18.4 ms (17.9 and 18.2 at 4 nodes); the first count all-reduce holds rank 0's wait for the slowest rank
  (0.50 and 0.19 s). The per-GPU loss from 1 to 13 nodes (11.5 % for
  published by wall clock, 10.3-10.7 % for the others) is the wait for the slowest of 104 samples (est.), not the
  network.
- InfiniBand sent per node and step (timing runs): 1.493 GB on one node, 2.239-2.240 GB on two, 2.986 GB on ten;
  35.83 GB in all, 24.3 x 1.473 GB against the double binary tree's 2 (N - 1) = 24 (+1.3 %). The profile run
  +1.3-2.7 % over them with exact step boundaries: about one 1.4 GB trace per node written to VAST over 34 steps
  (est.).
- GPUs: 573 W mean (published), 532 (ds), 454 (cueq+tf32); SM clock median 1,965 MHz in all; use 85, 83, 69 %.
  CPU: busy cores at 3,200 MHz median, 3,084-3,134 minimum: no drop.
- Profile at 13 nodes, rank 0, published, ProfilerStep#27 / #28 (logged steps 28 / 29): elementwise 1.73 / 1.45 s, copies 0.81 / 0.72, layer
  norm 0.62 / 0.53, FP32 SIMT GEMM 0.60 / 0.58, BF16 GEMM 0.38 / 0.33, softmax 0.29 / 0.27; triangle attention 36-38 %
  of compute kernels, diffusion 23-25 %; 193,989 / 179,342 kernels; 1.89 / 1.86 s of 10-100 us gaps. ProfilerStep#27
  matches 4 nodes within 0.01 s in every compute class. It is the logged step 28 (a ProfilerStep#N lasts as logged
  step N + 1 with the gap before it: 7.050 s against 7.050 at 4 nodes, 7.581 against 7.582 at 13; #28 is step 29,
  whose trace export follows), so the two traced near-identical samples: monomers of 384 tokens with 3 recycles,
  MGYP002639174100 (2,890 atoms) here and MGYP002718610066 (2,933) at 4 nodes. The match therefore no longer shows
  that the sample does not matter. Unchanged by this mapping: the class and module figures, NCCL's 0.522 against
  0.071 s (rank 0's wait for the slowest rank), the 18 ms all-reduce, the 0.22 s sync, and "7.0 against 5.7 s"
  (7.050 against 5.761 s on the same sample).
- Nodemon near each run's end (about 210 W, 0 % use): after the run's last step, outside the step window the numbers
  above use (1.2-2.2 % of in-window samples, all nodes, have 0 % instantaneous use): Lightning's teardown and exit, then prof.py (profile run)
  or the next run's 4-min start. Each run's sampler was also left running until the job's end (job.sbatch killed the
  function's subshell, not its srun; fixed after this job), writing into its own run directory: no effect on the
  per-run windows.

## Step 4: one H100 node (Nibi)

Nibi's H100 nodes: 8 H100 80 GB (NVLink), 112 cores, 2 TB. Same layout as Trillium, `O=$SCRATCH/trillium/new/of3` on
Nibi's scratch: the same commit of this repository, OpenFold3 0.5.1, a venv from the same install lines (uv 0.10.12
installed in `$O/bin`; `uv pip freeze` identical to Trillium's venv but for py-spy), the step-2 subset fetched on the
login node (`subset-u/keys.txt`, same md5, plus the reference molecules: 103,370 files, 9.32 GB) and the subset caches
copied from Trillium (`subset-u`, `subset-u2`). Same seed, so the same samples as the 1-node B200 runs (1273887).
12 cores per rank as on Trillium (Nibi has 14). Submit from the login node, one job at a time:

    sbatch --job-name=nw-of3-h100-a --account=rrg-bengioy-ad_gpu --nodes=1 --ntasks-per-node=8 --cpus-per-task=12 \
      --exclusive --mem=0 --gpus-per-node=h100:8 --time=00:35:00 --output=$O/slurm/%x-%j.out \
      $O/src/scripts/trillium/of3/job.sbatch 40 uniform uniform,cueq+tf32
    # then job.sbatch 40 uniform,ds uniform,prof (--time=00:50:00: DeepSpeed's kernel is built for sm_90 first)

### First job: published and cueq+tf32 (26594554, g26, 2026-11-09 13:08-13:35, commit 0185adf4)

Setup, from job.sbatch's start print and the run logs: 8 NVIDIA H100 80GB HBM3 (SXM, NVLink), driver 580.196.07
(reports CUDA 13.0, `cudaDriverVersion 13000`; torch 2.15.0 brings CUDA 13.2 and runs on it), enforced power limit
700 W (B200: 1,000 W); NCCL 2.30.7, one node. Differences from the Trillium runs (1273887) besides the GPU:
- commit 0185adf4 against 23d64566: between them, of what the runs execute, `uniform` points at subset-u2 and
  job.sbatch prints a start time and the GPUs; train.py, merge.py and initial_training.yml are unchanged (the rest:
  README, data.sh, monomers.py, prof.py, summary.py, and other workloads' files);
- `uniform` read subset-u2's caches on Nibi against subset-u's on Trillium (same PDB, RNA and first 800 monomers;
  the samples were checked equal on all 8 ranks at all 40 steps);
- the B200 driver reports CUDA 13.2 (`cudaDriverVersion 13020`), Nibi's 13.0;
- CPUs: 2 x 56 Sapphire Rapids cores on g26 (Slurm feature `sapphire`; model not recorded) against 2 x 48 Xeon
  Platinum 8558P (lscpu in `$SCRATCH/trillium/hw/`). Clocks of the cores at least 50 % busy (nodemon, from
  /proc/cpuinfo), median over steps 6-40: 4,000 MHz on Nibi (minimum 3,590 and 3,461 in the two runs) against
  3,200 MHz on Trillium, which never goes above 3,200 though its scaling_max_freq is 4,000;
- NUMA: on g26, 5 of the 8 ranks have no core in their GPU's NUMA domain and 2 only partly; on trig0145, 4 of 8 have
  none (effect on step time not measured);
- filesystem: Nibi's `/scratch` for code, venv and data, NFS v3 over TCP (Trillium: VAST, NFS v3 over RDMA).
Same package versions, subset, seed and 12 cores per rank. The first run took 1,066 s, 13 min of it before training
started (cause not measured; 23 s from Lightning's start to step 1); the second run, warm, reached step 1 in 5.2 min,
against 3.3-3.5 min on B200.

Wall clock, (t40 - t6) / 34 of rank 0's step ends, as OF3-01; the median step period beside it; B200 = 1273887
(trig0145), the same samples at every step on every rank:

| | H100 s per step (median) | H100 samples/s per GPU | B200 samples/s per GPU | B200 / H100 |
|---|---|---|---|---|
| published | 6.926 (6.939) | 0.1444 | 0.1907 | 1.321 |
| cueq+tf32 | 4.418 (4.393) | 0.2264 | 0.2334 | 1.031 |
| ds | job 2 | | 0.2166 | |

- cuEquivariance + TF32 against published on the H100: 0.638 of the step (B200: 0.817). The B200's lead shrinks
  from 32 % to 3 % with the fast kernels. Estimates from GPU use x step period: with cueq+tf32 the GPUs are busy
  3.73 s per step on the H100 against 2.76 s on the B200 (1.35x; published 1.48x), and idle 0.69 s against 1.52 s.
  So the lead shrinks because the B200 waits while the H100 stays 84 % busy; the faster CPU clock is not needed to
  explain it. Inferred, not measured, until job 2's H100 profile: the B200's waiting is the host's kernel launches
  (inferred from the B200 profile's gaps).
- Loss tracks B200's: 0.012 mean absolute over 40 steps (largest 0.086, step 21; loss 6.5-15.3), ten times a same-seed
  B200 pair (1273594 `published` against 1273761 `published,prof`, step-1 subset, profiler on in the second; loss
  averaged over ranks per step, steps 1-40: 0.0012, max 0.0043) or cueq+tf32 on either GPU: the GPU moves the loss
  more than the kernels do. Over 100 steps the same pair differs by 0.044 (max 0.32, recomputed from both runs'
  logs), so the comparison holds over these 40 steps only. Peak 31.5 GiB
  allocated, 33.3-33.4 reserved, as on B200.
- Data loader: gap 2.4 % (published) and 3.3 % (cueq+tf32) of the time, mean over ranks (B200: 3.5, 4.8 %);
  featurization median 2.4-2.6 s per sample, max 20 s, workers busy 3-4 %.
- GPUs: 439 W mean, 590 max (published), 378 and 548 W (cueq+tf32), of 700 W; SM clock 1,980 MHz, the H100's
  maximum, in every sample, and the clock-event reasons, which nodemon logs, are 0x0 in every sample: no power or
  thermal cap, on both the reasons and the clocks; 55 C at most; use 93 and 84 %. B200: 615 and 471 W, SM clock
  median 1,965 MHz; its cueq+tf32 run had 18 of 1,168 samples at the software power cap (reason 0x4), down to
  1,897 MHz; use 83 and 65 %.
- Limits: one run per variant and side, 34 timed steps, one node; H100 run-to-run spread not measured.

### Second job: ds and the profile (26618631, g5, 2026-11-13 05:57-06:24, commit e3b83808)

g5 instead of g26. The start print is the same (8 H100 80GB HBM3, driver 580.196.07, 700 W), and so is the NUMA
placement (NCCL's affinity lines: GPU 0's rank inside its domain, GPUs 5 and 6 partly, the other five outside). Busy
cores at 3,968 MHz median (minimum 3,509 and 3,402) against 4,000 on g26. Job 2 has no `uniform` or `cueq+tf32` entry,
so the node's effect is not measured. Samples equal on all 8 ranks at all 40 steps to B200's `ds`, to job 1's two
runs and to the profile run.

| | H100 s per step (median) | H100 samples/s per GPU | B200 samples/s per GPU | B200 / H100 |
|---|---|---|---|---|
| ds (g5) | 5.674 (5.671) | 0.1762 | 0.2166 | 1.229 |

- On the H100, ds is 0.819 of job 1's published step and 1.284 x its cueq+tf32 step (other node); on the B200, 0.880
  and 1.078. DeepSpeed's attention gains more on the H100 (18 % against 12 %, across nodes g5 and g26).
- Loss against B200's ds: 0.012 mean absolute over 40 steps (largest 0.086), as job 1; against job 1's published on
  the H100: 0.002 (largest 0.006). Peak 31.5 GiB allocated, 33.3 reserved. Loader: gap 2.8 % (worst rank 3.4 %);
  featurization median 2.5 s, max 31 s, workers busy 3 %.
- GPUs: 420 W mean, 588 max (ds), 409 and 591 W (profile run, whose window includes the 17 s trace export, when
  GPU 0 sits at 0 % and 120 W), of 700 W; SM clock 1,980 MHz in every sample, clock-event reasons 0x0 in every sample
  (no cap, on both), 57 C at most; use 91 and 92 % (the profile run's with the export).
- Like for like (same samples, one node, no profiler; job 1's published against B200's 1273887): H100 / B200 is 1.40
  at step 28 and 1.28 at step 29.

Profile (`uniform,prof`, published with the profiler, rank 0, ProfilerStep#27 / #28 (logged steps 28 / 29) with
prof.py, as the B200 profile in "Where a step goes": 4 nodes, 1273884). #27 is 6olg (PDB, 2,795 atoms, 3 recycles);
4-node B200 #27: MGYP002718610066 (monomer, 2,933 atoms). The 1-node B200 profile run 1273761 (step-1 subset) traced
6tdz at #27 (PDB, 2,896 atoms, 384 tokens, 3 recycles), closer to the H100's and without inter-node waits: the better
comparison for #27. Neither B200 trace matches the H100's #28 (1ppr, 894 atoms; 4-node B200: 2a3v, 3,778 atoms).
H100 / B200 per traced step, #27 then #28:

| | H100 | B200 (4 nodes) | B200 (1 node, 1273761) |
|---|---|---|---|
| traced span | 8.93 / 7.48 s | 7.04 / 7.01 s | 7.81 / 6.91 s |
| GPU busy (share of span) | 6.81 / 5.69 s (76 / 76 %) | 4.63 / 4.23 s (66 / 60 %) | 4.97 / 4.40 s (64 / 64 %), NCCL 0.42 / 0.21 s of it |
| idle: gaps < 10 us, 10-100 us, 0.1-1 ms, > 1 ms | 0.58, 1.30, 0.05, 0.20 / 0.46, 1.15, 0.05, 0.13 s | 0.34, 1.82, 0.10, 0.14 / 0.35, 1.89, 0.12, 0.43 s | 0.35, 1.90, 0.09, 0.49 / 0.35, 1.88, 0.09, 0.19 s |
| kernels, median | 210,295, 2.5 us / 177,137, 2.9 us | 193,989, 2.9 us / 185,895, 2.8 us | 196,220, 2.9 us / 186,598, 2.9 us |
| launch calls, CPU time | 3.23 / 2.63 s | 1.06 / 1.06 s | 1.07 / 1.05 s |
| launch calls: median; up to 20 us; over 100 us | 3.2 us; 0.84 / 0.70 s; 2.19 / 1.75 s | 3.9 us; 0.89 / 0.87 s; 0.15 / 0.17 s | 3.9 us; 0.92 / 0.88 s; 0.14 / 0.16 s |
| launches meeting over 1 ms of queued GPU work | 44 / 45 % | 35 / 31 % | 35 / 31 % |
| elementwise; copies; layer norm; softmax | 2.58; 1.42; 0.82; 0.34 / 2.08; 1.20; 0.65; 0.29 s | 1.73; 0.82; 0.62; 0.29 / 1.55; 0.78; 0.55; 0.24 s | 1.73; 0.83; 0.62; 0.29 / 1.54; 0.78; 0.55; 0.24 s |
| GEMMs | 1.50 / 1.36 s ("other": prof.py's FP32 SIMT pattern matches 0.003 s on the H100, so its FP32 GEMMs are in here, not separated) | FP32 SIMT 0.60 / 0.61 s, BF16 0.38 / 0.33 s | FP32 SIMT 0.59 / 0.60 s, BF16 0.38 / 0.33 s |
| the 1.473 GB all-reduce | 5.7 / 5.6 ms (NVLink, one node) | 17.9 / 18.2 ms (InfiniBand, 4 nodes) | not listed in that analysis |

Against job 1's reading. Published, #27 against the 1-node B200 (similar samples): the H100's kernels take 1.37 x
longer (6.81 against 4.97 s) and its GPU idles less (2.12 against 2.84 s of the traced span), mostly in the 10-100 us
gaps between small kernels (1.30 against 1.90 s): the B200 waits more. On the H100 the launch calls take 3.0 / 2.5 x
the B200's CPU time: 2.19 / 1.75 s of the 3.23 / 2.63 s are calls over 100 us, 99-100 % of them made while the GPU
already had more than 1 ms of work queued (median 53 ms). The CPU is blocked on a full launch queue, waiting for the
GPU, not paying a profiler or driver cost: its median call takes 3.2 us (B200 3.9), and its calls up to 20 us total
0.84 / 0.70 s (B200 0.88-0.92). 44-45 % of the H100's launches meet over 1 ms of queued work, against 31-35 % on the
B200, whose blocked calls total only 0.14-0.17 s: the B200's queue drains between them. So the H100's published step
is GPU-bound for long stretches, the B200's for shorter ones.

With cueq+tf32, measured on the B200 only (1273761, 1 node, step-1 subset; 1273763, 2 nodes): the GPU is busy 37-42 %
of the traced span (2.40-2.98 s), idle 2.78-2.97 s in 10-100 us gaps and 0.64-0.72 s in 0.1-1 ms gaps; launch calls
median 3.9 us, 0.02-0.03 s in calls over 100 us, and only 4-6 % of the launches meet over 1 ms of queued work: the
B200 waits on the host's launches. That half of job 1's reading is now measured; the H100's side with cueq+tf32 is
not profiled. The profiler also adds more idle time on the B200 (below), so the idle comparisons partly reflect it.
`ds` on the B200 (1273762) sits between: busy 53-57 %, 16-17 % of launches meeting over 1 ms of queued work.

Limits: one run per entry. The profiler's own cost, untraced steps against a run without it: B200 1-node (1273761
against 1273594) 1.146 over steps 7-100, 1.084 over steps 7-40, the H100's window; H100 1.032 (job 1's published,
other node; median of the per-step period ratio over steps 7-40 without 26-30; 1.034 leaving out only 28-30; ratio
of the summed periods 1.029, or 1.033 without only 28-30); both with the sync timing on. Traced steps against the same sample without the profiler: H100 1.164,
B200 1.224 (4 nodes) and 1.314 (1 node).

B200 traces checked on 2026-11-13 (`trace_gaps.py`: rank 0 of 1273761 published and cueq+tf32, 1273762 ds,
1273763 published and cueq+tf32, 1273884 uniform; copied with one rsync, parsed locally): on all six, ProfilerStep#27
starts within 0.017 s of logged step 27's end and ends within 0.017 s of logged step 28's end, and #28 starts at step
28's end: ProfilerStep#N is logged step N + 1 by timestamp, as on the H100. The H100 trace, scanned the same way,
gives prof.py's figures to the millisecond.

## Not tested yet

More than 4 nodes; the step-1 subset's stall at scale with the full mix; the later stages beyond one 1-node point;
more or fewer workers; short monomers and disordered PDB; checkpoint writes and validation; `per_sample_clipping:
false` (overlapped all-reduce).
