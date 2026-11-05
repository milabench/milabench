# GLM-5 runs on Trillium (B200)

Megatron-LM GLM-5 (744B, `benchmarks/megatron/dev.yaml`, entries `megatron-glm5-tri-*`) for the experiments of
`scripts/trillium/DESIGN.md`: A-G (layouts), B-G (copies), C (precision), F1 (checkpoints). One script per step, run from
a **login node** of `trillium-gpu`; each submits one Slurm job per node count and follows its log (ctrl-c stops
following, not the job). `DRY_RUN=1` prints what would be submitted.

| Step | Script | What it does |
|------|--------|--------------|
| 0 | `./00_plan.sh` | No GPU. Every layout of the campaign against Megatron's rules: copies, data parallel, micro-batches per pipeline. `./00_plan.sh pp13 8 13 256` shows one layout's rank groups. |
| A-G | `./10_ag.sh 13\|26\|52\|12\|48\|10\|20\|30\|40\|25\|50` | Layouts pp13-ep8 / ep16 / ep32, pp26-ep8, pp6-ep16, pp10-ep8, pp25-ep8; selective recompute, 64 per copy |
| B-G | `./20_bg.sh 13\|26\|39\|52` | pp13-ep8 with 1-4 copies: 256 per copy (weak) and 768 in total (strong) |
| C | `./30_c.sh speed 13\|52`, `./30_c.sh loss [recipes]` | bf16 / fp8-tensorwise / fp8-blockwise / mxfp8; 500-step loss runs from seed 1234 |
| F1 | `./40_f1.sh 13\|52 [all\|weights\|full]`, `./40_f1.sh 26` | weights then full: save sync, load, save async, load; the checkpoint is deleted after each load. 26: full at pp13-ep16, the fallback |

Global batch: with PP stages and m micro-batches per data-parallel rank, the 1F1B pipeline bubble is
(PP - 1) / (m + PP - 1), so the batch is chosen in micro-batches. Weak scaling, and the layout comparison at a node count
(A-G), use 32 micro-batches per data-parallel rank of pp13-ep8 at micro-batch 1: 256 / 512 / 768 / 1024 sequences at
13 / 26 / 39 / 52 nodes, bubble 12 / 44 = 27 % throughout (the other layouts take their node count's batch; pp6-ep16:
256 at 12 nodes, 1024 at 48). Strong scaling keeps 768 in total: 96 / 48 / 32 / 24 micro-batches, bubble 11 % to 33 %,
the cost of spreading a fixed batch. 64 per copy (8 micro-batches, 60 %) is the small-batch point. Micro-batch stays at
1: the GLM-5 plugin's attention takes THD input only (`--micro-batch-size 2` fails in its first forward, Rorqual job
25534799) and the launcher refuses it. `./00_plan.sh` prints
every entry's micro-batches and bubble.

Every entry runs on exactly its `num_machines` nodes (the job is sized from the node count), and the layout is checked
before launch (`placement.check_layout`): a wrong node count or batch fails at once, not after the model build.

Environment: see `_common.sh` (`MILABENCH_BASE` = `$SCRATCH/trillium/glm`, account `rrg-bengioy-ad`, partition
`compute_b200_full_node`, `--gpus-per-node=b200:8`, reservation `b200-friendly-users` and QOS `bump` (`RESERVATION`, `QOS`), `CAMPAIGN` =
`trillium-MMDD`, `REPETITION`). Node telemetry under `$SCRATCH/trillium/nodemon/<run>`, checkpoints under
`$SCRATCH/trillium/checkpoints/<job>`, run data in `$SCRATCH/trillium/runs` (`$MILABENCH_BASE/runs` links there).

## If only 47 nodes are available

Five nodes may stay reserved for SciNet. Nothing below runs by default; each is selected by its node count, with the same `run.experiment`:
`./30_c.sh speed 39` (C-speed), `./40_f1.sh 39` (F1), `./10_ag.sh 36` (pp6-ep16, 3 copies, 768, the substitute for n48: 288 GPUs, 48 per stage). 52-node A-G
and B-G runs have no substitute beyond pp13-ep8 at 39 nodes (`./20_bg.sh 39`).

## C-loss noise

`./30_c.sh loss` runs 6 jobs: bf16, bf16 again with the same seed (`bf16-rep2`), bf16 with another seed (`bf16-seed2`), and the three FP8
recipes. The two extra runs give the size of the differences between BF16 runs, against which the FP8 vs BF16 gap can be read. Budget: 2 more
runs of 13 nodes x 500 steps, +50 % on C-loss (limit 8 h 30 each: 48.61 s per BF16 step at 13 nodes on the speed entry's data, job 1242058, so about 7 h with startup, an estimate). To drop them: `./30_c.sh loss "bf16 fp8-tensorwise ..."`.

## Runs that serve two experiments

`run.experiment` holds one value, the tag the launcher gives. Four entries are not run a second time, and serve a second
experiment under the tag they already have:

| entry | `run.experiment` | also serves |
|---|---|---|
| `megatron-glm5-tri-pp13-ep8-n13-gbs256` | `A-G` (`10_ag.sh 13`) | `B-G-weak`, 13 nodes |
| `megatron-glm5-tri-pp13-ep8-n52-gbs1024` | `A-G` (`10_ag.sh 52`) | `B-G-weak`, 52 nodes |
| `megatron-glm5-tri-pp13-ep8-n39-gbs768` | `B-G-weak` (`20_bg.sh 39`) | `B-G-strong`, 39 nodes |
| `megatron-glm5-tri-pp13-ep8-n26-gbs512` | `B-G-weak` (`20_bg.sh 26`) | `A-G`, the pp13-ep8 reference at 26 nodes (`10_ag.sh 26` has no pp13-ep8 run) |

Everything else has one experiment: n13 / n26 / n52 at 768 are `B-G-strong`. C-speed runs its own BF16 baselines (`30_c.sh speed`, same entries as the first two rows, tagged `C-speed`).

## MFU

`mfu_inputs.flops_per_token` is Megatron's `num_floating_point_operations` with the attention term per packed slot (8 slots of
1024 tokens; treating a sequence as 8192 long is too high, by about 22 % according to a review's estimate, not measured here). GLM-5's DSA sparsity is not modelled: attention counts as
dense, so the FLOPs, and the MFU, are still overestimates.

## Data to stage on Trillium

Compute nodes have no internet: FineWeb is staged from a login node, once. C-loss reads `$SCRATCH/trillium/data`, shared with Qwen's
C-loss (`MILABENCH_DIRS_DATA`, set by `./30_c.sh loss`). Staged on 2026-11-05: 4.2 M documents of `sample-10BT`, 2.91e9 tokens (GPT-2
count), 21 parquet files, 5.5 GiB; an Arrow cache of 13 GiB per `datasets` version (GLM-5's 5.1.0, Qwen's 4.7.0), both built; the
two tokenizers, copied from the two bases. C-loss reads 500 x 256 x 8 = 1,024,000 slots, 1.05e9 tokens, at document indices up to 4.096 M (stride 4).

How (`stage-fineweb.sh` and `warm-cache.sh` in that directory): `prepare_fineweb(dest, 4_200_000, warm_cache=False, workers=1)` under `nice`, 8.2 GB read in 8.5 min; then
`load_fineweb(dest)` from each venv with `HF_HOME=$SCRATCH/trillium/data`, 57 s and 26 s of one core. Keep the parquet files where they are: the Arrow cache's
key holds their paths and mtimes. `prepare_fineweb` returns early when the documents are there, without building a cache. A `milabench prepare` of a loss entry without `MILABENCH_DIRS_DATA` would fetch a second copy into `$MILABENCH_BASE/data`.

The summary's `data_loading` records `docs_available`, `doc_stride`, `slots_read`, `docs_read`,
`slots_over_stride` (slots that needed more documents than the stride: those overlap the next slot; 8.6 % *(est.: 6,000 documents through `slot()`, 2.8 documents per slot)*), `max_doc_index` and
`wrapped`, read loaders' prefetching included. The speed entries read 200,000 documents at most from `$MILABENCH_BASE/data` and need no stride: they measure
speed, and C-loss needs the distinct text.

## What the runs record (run data, `scripts/analysis/FORMAT.md` + `scripts/trillium/FORMAT.md`)

- `run_start` / `summary`: `run.experiment` (A-G, B-G-weak, B-G-strong, C-speed, C-loss, F1), `parallelism.layout`
  (`pp13-ep8`), `pp_schedule`, `expert_dp`, `copies`, `pp_split` (layers per stage), `microbatches_per_pipeline`;
  `workload.precision` (`bf16`, `fp8-tensorwise`, `fp8-blockwise`, `mxfp8`), `precision_detail`,
  `grad_reduce_dtype`, `seed`; `mfu_inputs.peak_flops_per_gpu` (dense BF16 spec).
- `loss` events with `step` and `grad_norm`, from the last pipeline stage, emitted by rank 0 every 10 steps and at the end
  (a run that times out loses at most 10 steps; the same values are in that rank's step rows). `step` counts from 1: the first
  step's loss has `step` 1, the first step's `rate` event has `batch_id` 0.
- F1: one `checkpoint` event per save / load, as in `scripts/trillium/FORMAT.md`. Before a load, every node evicts the
  checkpoint from its page cache (`posix_fadvise(DONTNEED)`), so the load time is the filesystem's.

## When a rank fails

An uncaught Python exception stops the benchmark on every node (`kill_everywhere` in `main.py`, the first failing rank of each node
runs it, it also stops the checkpoint worker), its traceback is in the rank's metrics file as `kind: exception`. A segfault or an
external kill does not go through it: the other ranks stay blocked until Megatron's `--distributed-timeout-minutes` where it applies, or
the entry's `max_duration`, or the job's limit. `job.sbatch` then kills what is left on the nodes before the next entry.

## Checkpoint memory (F1)

A full load holds the loaded and the current optimizer state on the GPU at once. Measured on a truncated GLM-5 (20.7B parameters,
PP2 x EP8), on B200 (job 1239618), peak GiB, largest rank: training 25.85, full save 32.66 sync and 36.9 async, full load 50.28
(1.95 x training). Fir gave the same save and load peaks (job 2304131). At PP2 x EP4 a load ran out of the H100's 79 GiB (job 2302534).
Whether full GLM-5 loads fit on a B200 (179 GiB) has not been measured: read `peak_allocated_gib` in the load's `checkpoint` event at
the first 13-node full load, and treat a full load as needing about twice the training peak (from the truncated model, an estimate)
before starting the 52-node one. Weights-only loads have no optimizer state to hold.

Size of a full GLM-5 checkpoint: ~8.9 TB, 12.0 B/param measured on the truncated model x 744B parameters, an estimate; weights ~1.5 TB.

A full save raises GPU memory too (above, and 38.4 to 56.5-60.5 GiB at PP2 x EP4 on Fir, job 2302534). At `full 13` this may not fit:
~190 GiB against 179 (a review's estimate, not measured). So `40_f1.sh` runs weights first, and full afterwards: an out-of-memory run costs little.
The fallback if it happens is `./40_f1.sh 26`: full, pp13-ep16, one copy, with the expert state over twice the GPUs.

## Install on Trillium

Done once on 2026-11-04 (jobs 1239118, 1239128, 1239150). From a login node, which has internet and an H100 but must stay light:

```bash
unset PYTHONPATH PIP_CONFIG_FILE; export PATH=$HOME/.local/bin:$PATH UV_CACHE_DIR=$SCRATCH/uv-cache
export MILABENCH_BASE=$SCRATCH/trillium/glm MILABENCH_GPU_ARCH=cuda
mkdir -p $MILABENCH_BASE $SCRATCH/trillium/runs; cd $MILABENCH_BASE
ln -s $SCRATCH/trillium/runs runs   # first: `milabench install` would create a real directory, and the launchers would not link it
git clone -b trillium https://github.com/milabench/milabench.git src
uv venv --python-preference only-managed --python 3.12 --seed src/.venv
VIRTUAL_ENV=$PWD/src/.venv uv pip install -e ./src pybind11 nvidia-ml-py
export PATH=$PWD/src/.venv/bin:$PATH E=megatron-glm5-tri-test-n2-gbs16 C=src/benchmarks/megatron/dev.yaml
SLIME_DOWNLOAD_ONLY=1 milabench install --config $C --select $E
milabench prepare --config $C --select $E   # the small data; for the runs, "Data to stage"
uv pip install --python venv/torch/bin/python onnxscript onnx nvdlfw-inspect nvidia-cudnn-frontend importlib-metadata   # TE's dependencies
mkdir wheelhouse   # transformer_engine, transformer_engine_cu13 (x86_64) and transformer_engine_torch 2.18.0, from PyPI's file URLs
```

Then, on one B200 node, a job (15 min is the partition's minimum; it took 3.5 min) running the same `milabench install` with
`UV_OFFLINE=1 UV_NO_INDEX=1 UV_FIND_LINKS=$MILABENCH_BASE/wheelhouse NVTE_PYTORCH_FORCE_BUILD=TRUE TMPDIR=/dev/shm/$USER/tmp`. Compute nodes have
no internet: the force-build flag skips TE's search for a prebuilt wheel. The install returns 1 because `torch_memory_saver`, optional, cannot be
fetched offline; `.slime_installed` then reads `ok, missing: torch_memory_saver`.

Notes from the bring-up:

- **Driver**: 595.91.07, the torch cu130 wheels work (they need >= 580).
- **CUDA module**: the job loads none: the venv brings its own toolkit (torch / nvcc wheels). `cuda/12.9` loads only after
  `StdEnv/2023 gcc/12.3`, and its headers on `CPATH` then break TileLang's kernel build with the wheels' nvcc (`cuda/std/utility`,
  "missing binary operator before token", job 1239249).
- **Caches**: `$HOME` is read-only from the compute nodes: `job.sbatch` points `XDG_CACHE_HOME` at `$MILABENCH_BASE/cache` (scratch). Install with `UV_CACHE_DIR` on scratch too.
- **Failed builds**: `milabench install` only logs a failed `pip install`. After the builds the pack imports
  `transformer_engine.pytorch` in the venv and stops if it can't (the launchers also refuse a venv without the marker); `torch_memory_saver` is optional for this pack (a `WARNING`
  in the install log, and `.slime_installed` then reads `ok, missing: torch_memory_saver`). Check that file after installing.
  On Fir it fails at link time (`cannot find -lcuda`, no `libcuda.so` on the linker path; Fir job 2302534): the GLM runs there
  go without it. `TMS_CUDA_MAJOR` defaults to 13 like the toolkit wheels, so `--set cuda=130` is not needed for it (not run).
- **Megatron's dataset helpers**: rank 0 runs `make` in `Megatron-LM/megatron/core/datasets` at every start, with
  the `python3` on `PATH`, which under milabench's launcher is the milabench CLI's (`MILABENCH_VENV`): install
  `pybind11` there (`uv pip install pybind11`). The checkout must be writable; a failed build stops the run.
- **`TMPDIR`**: `job.sbatch` puts `TMPDIR` and the Triton / TileLang caches in `/dev/shm/$USER` (RAM), logs the free space at the start, removes it at the end. Kernels are therefore compiled again per job.
- **No local disk** (`/tmp` and `/dev/shm` are RAM): unlike Rorqual's `job.sbatch`, nothing is copied to a node-local base; data and
  venv are read from scratch (VAST).
- **sm_100 kernels**: on 2 nodes of the truncated model (jobs 1239307, 1239405), TileLang's kernels (`-arch=sm_100a`) and SGLang's RoPE (`sm100a`) built
  and trained, TE's grouped GEMM too, in BF16, FP8 tensorwise / blockwise and MXFP8. Which FP8 kernels ran was not read directly: peak memory
  (21.6 to 32-34 GiB) and the losses (BF16 repeats to 3 decimals, the recipes do not) changed.
- **NCCL**: the GLM entries set `NCCL_NVLS_ENABLE=0` and `NCCL_CUMEM_ENABLE=0` (Rorqual workarounds). On B200 NVLink
  SHARP (NVLS) may help the in-node collectives: try a 13-node run without them before A-G.
- **FP8 blockwise**: on Hopper it needs `NVTE_FP8_BLOCK_SCALING_FP32_SCALES=1`; on B200 it must not be set (TE runs
  it with MXFP8-style scales). The Trillium entries do not set it.
- **wandb**: no `httpproxy` module on Trillium so far: the job falls back to `WANDB_MODE=offline`; `wandb sync` from a
  login node afterwards (the job prints the command).

## Known broken entry

`megatron-glm5-g1-ep8-aligned-pp5` (a Rorqual fallback, left as it is) fails the layout check before launch: global batch 256
is not divisible by data parallel 40.
