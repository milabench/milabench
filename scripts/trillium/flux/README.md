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
- `submit.sh NAME NODES TIME [NAME=value ...] [-- sbatch args]`: one job `nw-flux-NAME`.

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

## Status

Step 1 (feasibility) in progress. Untested: the image build, every job.
