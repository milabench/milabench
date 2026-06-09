## Overview

## Entry point

`src/scripts/train.py` (from the cloned `operator_learning` repo, milabench branch).
The benchfile sets `main_script = "src/scripts/train.py"`.

## Source code location

Not vendored. Cloned at install time from:
`https://github.com/Delaunay/operator_learning` (branch: `milabench`)
into `<code_dir>/src/`.

## Data pipeline

- Dataset: HDF5 files from `chelseajohn/FNOBenchmark` on HuggingFace (git-lfs)
- Cloned into `<data_dir>/FNOBenchmark/` during `prepare()`
- Data loaded via `operator_learning.data.getDataLoaders` which returns
  (train_loader, val_loader, test_loader)
- Batch size derived from config YAML files in `src/config/`

## voirfile.py instrumentation

Uses ptera probes to intercept two functions:
1. `getDataLoaders() as loader` -- wraps the train dataloader with
   `BenchObserver.loader()` for throughput measurement
2. `get_loss_fn() as criterion` -- wraps loss function with
   `observer.criterion` for loss tracking

The observer uses `earlystop = stop + skip` (default 165 batches total).
`batch_size_fn` extracts `x[0].shape[0]` from the batch tuple.

After the run (or early stop), `cantilever.core.timer.show_timings(force=True)`
prints accumulated timing data.

## benchfile.py

`Fno_benchmark(Package)`:
- `install()` clones the operator_learning repo if not present
- `prepare()` clones the FNOBenchmark dataset from HuggingFace if not present
- Uses `install_variant: unpinned` (no locked requirements)
- `prepare.py` is a no-op placeholder (actual prep is in `benchfile.prepare()`)

## Config structure

Each variant passes:
- `--configf`: path to a YAML file in `src/config/` defining model architecture,
  training hyperparameters, and data config
- `--dataFile`: path to the HDF5 data file
- `--benchmark true`: enables benchmarking mode
- `--use_amp 1`: mixed precision training

## Computational profile

- FNO layers perform spectral convolutions via FFT -> pointwise multiply -> iFFT
- Memory-bound on large grids (256x64 for 2D, 64x64x32 for 3D)
- `finufft` (non-uniform FFT) is a key dependency for some operator variants
- `opt_einsum` used for tensor contractions in the operator layers
- 3D variants (rbc3d) are significantly more compute-intensive

## Gotchas

- Requires git-lfs for dataset download; will silently get pointer files without it
- `clone_subtree` is a milabench utility, not standard git
- Heavy dependency list (33 packages) including MPI -- can conflict in containers
- The `prepare.py` script does nothing useful; all prep logic is in benchfile
- `mpi4py` is listed but single-GPU only in current config
- Config files live inside the cloned repo, referenced by `{benchmark_folder}` template variable
