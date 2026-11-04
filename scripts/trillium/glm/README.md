# GLM-5 runs on Trillium (B200)

Megatron-LM GLM-5 (744B, `benchmarks/megatron/dev.yaml`, entries `megatron-glm5-tri-*`) for the experiments of
`scripts/trillium/DESIGN.md`: A-G (layouts), B-G (copies), C (precision), F1 (checkpoints). One script per step, run from
a **login node** of `trillium-gpu`; each submits one Slurm job per node count and follows its log (ctrl-c stops
following, not the job). `DRY_RUN=1` prints what would be submitted.

| Step | Script | What it does |
|------|--------|--------------|
| 0 | `./00_plan.sh` | No GPU. Every layout of the campaign against Megatron's rules: copies, data parallel, micro-batches per pipeline. `./00_plan.sh pp13 8 13 256` shows one layout's rank groups. |
| A-G | `./10_ag.sh 13\|26\|52\|12\|48` | Layouts pp13-ep8 / ep16 / ep32, pp26-ep8, pp6-ep16; selective recompute, 64 per copy |
| B-G | `./20_bg.sh 13\|26\|39\|52` | pp13-ep8 with 1-4 copies: 256 per copy (weak) and 768 in total (strong) |
| C | `./30_c.sh speed 13\|52`, `./30_c.sh loss [recipes]` | bf16 / fp8-tensorwise / fp8-blockwise / mxfp8; 500-step loss runs from seed 1234 |
| F1 | `./40_f1.sh weights\|full 13\|52` | save sync, load, save async, load; the checkpoint is deleted after each load (timed) |

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
`compute_b200_full_node`, `--gpus-per-node=b200:8`, no reservation unless `RESERVATION` is set, `CAMPAIGN` =
`trillium-MMDD`, `REPETITION`). Node telemetry under `$SCRATCH/trillium/nodemon/<run>`, checkpoints under
`$SCRATCH/trillium/checkpoints/<job>`, run data in `$SCRATCH/trillium/runs` (`$MILABENCH_BASE/runs` links there), checkpoint deletions in `$MILABENCH_BASE/slurm/ckpt-rm.<job>.jsonl`.

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

## What the runs record (run data, `scripts/analysis/FORMAT.md` + `scripts/trillium/FORMAT.md`)

- `run_start` / `summary`: `run.experiment` (A-G, B-G-weak, B-G-strong, C-speed, C-loss, F1), `parallelism.layout`
  (`pp13-ep8`), `pp_schedule`, `expert_dp`, `copies`, `pp_split` (layers per stage), `microbatches_per_pipeline`;
  `workload.precision` (`bf16`, `fp8-tensorwise`, `fp8-blockwise`, `mxfp8`), `precision_detail`,
  `grad_reduce_dtype`, `seed`; `mfu_inputs.peak_flops_per_gpu` (dense BF16 spec).
- `loss` events with `step` and `grad_norm`, from the last pipeline stage (emitted by rank 0 at the end of the run;
  the same values are in that rank's per-step rows as they happen).
- F1: one `checkpoint` event per save / load: `op`, `step`, `repetition`, `mode`, `content`, `format`, `bytes`,
  `files`, `blocked_s` (training stopped), `total_s` (until the write is complete; async: until Megatron finalizes
  it), `path`, `filesystem`, `start_epoch`, `end_epoch`; a load adds `checksum_match` (weights checksum at save vs
  after load; a mismatch fails the run) and `saved_by` (mode and run of the save).

## What `checksum_match` proves (F1)

For every rank and every named model parameter (full: also the Adam moments and master weights of every parameter of the
optimizer), three float64 numbers: sum, sum of squares, sum weighted by position. They are written beside the checkpoint at
save time and must be exactly equal after the load; a difference, or a missing file, fails the run. This shows that each rank
got back the tensors it had, in the same order, on the same layout. It is not a cryptographic hash: a change that keeps all
three numbers would pass. It does not check the scheduler, the RNG (`--no-load-rng`), a load on another layout, or the data
loader position, and the save-side numbers are taken from memory before the write, so what it proves is the round trip.

## Checkpoint memory (F1)

A full load holds the loaded and the current optimizer state on the GPU at once. Measured on a truncated GLM-5 (20.7B
parameters, Fir): at PP2 x EP8 the peak was 50.3 GiB against 32.7 GiB in training with the same layout (1.54 x, job 2304131); at
PP2 x EP4 (38.4 GiB in training) it ran out of the H100's 79 GiB (job 2302534). Whether full GLM-5 loads fit on a B200 (180 GB) has
not been measured: read `peak_allocated_gib` in the load's `checkpoint` event at the first 13-node full load, and treat a full
load as needing about twice the training peak (from the two points above, an estimate) before starting the 52-node one.
Weights-only loads have no optimizer state to hold.

## Install on Trillium (not done yet: notes for the bring-up)

Same pack as Rorqual (`milabench install --config benchmarks/megatron/dev.yaml --select megatron-glm5-tri-pp13-ep8-n13-gbs256`),
with these differences, to check on the first B200 node:

- **Driver**: the torch cu130 wheels need driver >= 580 (`nvidia-smi`; the job prints it). Older driver:
  `milabench install ... --set cuda=129` (cu129 wheels and the matching nvcc wheels).
- **TransformerEngine build**: `NVTE_CUDA_ARCHS="100"` (sm_100 only, much shorter build); the build runs on a GPU node
  (it needs libcuda), from a job.
- **CUDA module**: the job loads `cuda/12.9` (SciNet's recommendation), but the venv's own toolkit (torch / nvcc wheels)
  must win: the module's `nvcc` and headers must not come first on `PATH` / `CPATH` when TE or TileLang compile.
  `benchmarks/slime/benchfile.py` puts the wheels' `nvcc` first; check `which nvcc` inside a run if a build fails.
- **Caches**: `$HOME` is read-only from the compute nodes: `job.sbatch` points `XDG_CACHE_HOME`, `TRITON_CACHE_DIR`,
  `TILELANG_CACHE_DIR` at `$MILABENCH_BASE/cache` (scratch). Install with `UV_CACHE_DIR` on scratch too.
- **Failed builds**: `milabench install` only logs a failed `pip install`. After the builds the pack imports
  `transformer_engine` in the venv and stops if it can't; `torch_memory_saver` is optional for this pack (a `WARNING`
  in the install log, and `.slime_installed` then reads `ok, missing: torch_memory_saver`). Check that file after installing.
  On Fir it fails at link time (`cannot find -lcuda`, no `libcuda.so` on the linker path; Fir job 2302534): the GLM runs there
  go without it. `TMS_CUDA_MAJOR` defaults to 13 like the toolkit wheels, so `--set cuda=130` is not needed for it (not run).
- **Megatron's dataset helpers**: rank 0 runs `make` in `Megatron-LM/megatron/core/datasets` at every start, with
  the `python3` on `PATH`, which under milabench's launcher is the milabench CLI's (`MILABENCH_VENV`): install
  `pybind11` there (`uv pip install pybind11`). The checkout must be writable; a failed build stops the run.
- **No local disk** (`TmpDisk` 2 GB): unlike Rorqual's `job.sbatch`, nothing is copied to a node-local base; data and
  venv are read from scratch (VAST).
- **sm_100 kernels to verify** in the 1-node smoke: TileLang DSA indexer and sparse MLA (GLM-5 plugin), SGLang RoPE
  (`MEGATRON_USE_SGLANG_ROPE=1`), TE grouped GEMM in BF16, FP8 tensorwise / blockwise and MXFP8.
- **NCCL**: the GLM entries set `NCCL_NVLS_ENABLE=0` and `NCCL_CUMEM_ENABLE=0` (Rorqual workarounds). On B200 NVLink
  SHARP (NVLS) may help the in-node collectives: try a 13-node run without them before A-G.
- **FP8 blockwise**: on Hopper it needs `NVTE_FP8_BLOCK_SCALING_FP32_SCALES=1`; on B200 it must not be set (TE runs
  it with MXFP8-style scales). The Trillium entries do not set it.
- **wandb**: no `httpproxy` module on Trillium so far: the job falls back to `WANDB_MODE=offline`; `wandb sync` from a
  login node afterwards (the job prints the command).

## Known broken entry

`megatron-glm5-g1-ep8-aligned-pp5` (a Rorqual fallback, left as it is) fails the layout check before launch: global batch 256
is not divisible by data parallel 40.
