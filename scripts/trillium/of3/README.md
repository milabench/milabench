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
- `variants.yml`: the kernel switches (`settings.memory.train`), merged by `merge.py`.
- `train.py`: `run_openfold train` plus one JSON line per step and rank (`steps-r<rank>.jsonl`: step time, the gap
  before it, loss, crop, peak memory). OpenFold3 itself logs only to wandb.
- `summary.py RUN [SKIP [REF]]`: the numbers below from one run (REF: step-by-step ratio to a run of the same samples).
- `data.sh` (on tri-dm2), `subset.py`, `fetch.py`: the data subset.

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
step by step. Step period: time between successive step ends on rank 0 (`summary.py`, steps 7 to the end); ratio: the
median over those steps of the variant's period over the published run's at the same step. Peak memory over all ranks.
Power, clocks and use: benchmate's node monitor over the same steps.

| run (job) | steps | period median (p10-p90) | samples/s per GPU | ratio | peak alloc / reserved | loss, mean of ranks |
|---|---|---|---|---|---|---|
| published (1273594) | 100 | 5.207 s (4.43-6.06) | 0.192 | 1 | 31.5 / 33.5 GiB (28.9 by step 40) | steps 1-10 12.32, 91-100 8.54 |
| `+tf32` (1273674) | 40 | 5.010 s | 0.200 | 0.937 | 28.9 / 30.6 | 31-40 10.38 |
| `triton` (1273611) | 40 | 5.113 s | 0.196 | 0.979 | 28.9 / 30.6 | 31-40 10.39 |
| `ds` (1273611) | 40 | 4.698 s | 0.213 | 0.875 | 28.9 / 30.5 | 31-40 10.38 |
| `cueq+tf32` (1273674) | 40 | 4.262 s (3.62-4.79) | 0.235 | 0.788 | 28.9 / 30.6 | 31-40 10.39 |
| `cueq-ds+tf32` (1273674) | 40 | 4.199 s | 0.238 | 0.788 | 28.9 / 30.6 | 31-40 10.39 |
| `cueq`, `cueq-ds` (1273611) | 0 | fail in the first backward | | | | |

- Kernels on B200 (`sm_100`): all run. The published config uses none of them (plain PyTorch triangle attention and
  multiplication, FP32 matmuls without TF32). cuEquivariance's triangle attention refuses an FP32 backward unless TF32
  is on ("Full precision FP32 backward pass for triangle attention is not implemented yet"), so it needs
  `--use_tf32 True`, itself 6 % faster; cuEquivariance then takes another 15 % (0.788 against FP32). DeepSpeed's
  Evoformer attention: 12.5 %; with cuEquivariance it adds nothing (cuEquivariance takes the calls first). OpenFold3's
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
backward runs under `no_sync`, then `grad_manager.sync_and_average_grads` all-reduces a sample count (then `.item()`),
the per-parameter sample counts (one int64 per parameter tensor) and all gradients as one flat FP32 buffer: 368,293,788
parameters, 1.47 GB per step, after the backward, not overlapped with it. Grad-norm metrics sync through torchmetrics.
At 13 nodes that is one 1.47 GB all-reduce per 4-5 s step.

## Not tested yet

More than one node (NCCL over InfiniBand with the pip NCCL 2.30.7), the all-reduce's share, a profile; the later
stages (larger crops); more or fewer workers; short monomers and disordered PDB; checkpoint writes and validation.

Kernel messages during the jobs on trig0145 (05:37, 05:40) and trig0151 (05:45-05:49), at process starts:
`NVRM: knvlinkSendInbandData_IMPL: Failed to send inband data: 0`. Training ran normally; cause not determined, not
compared with other workloads' jobs.
