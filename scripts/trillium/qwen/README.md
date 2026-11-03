# Qwen3.8-27B (TorchTitan) on Trillium B200

Launcher for the Qwen experiments of `scripts/trillium/DESIGN.md` (A-Q, B-Q, C, F1, R).
Entries: `config/torchtitan_native_trillium.yaml`. Each run is one Slurm job running
`scripts/slurm/tt_native.sh` (the Rorqual launcher) with `TT_CONFIG=trillium`.

```bash
cd $SCRATCH/trillium/qwen/milabench && git pull --ff-only
scripts/trillium/qwen/tt_trillium.sh --dry-run tri-smoke             # print the sbatch command
scripts/trillium/qwen/tt_trillium.sh --nodes 1 --time 00:20:00 tri-smoke
scripts/trillium/qwen/tt_trillium.sh --nodes 13 --time 00:30:00 TT_CAMPAIGN=window-1 tri-aq-hsdp8-24k
```

Defaults: partition `compute_b200_full_node`, `--gpus-per-node=b200:8`, account
`rrg-bengioy-ad` (`TRI_ACCOUNT`, `TRI_PARTITION`, `TRI_RESERVATION` change them),
`module load cuda/12.9` in the job, `CC_CLUSTER=trillium`, wandb offline (no proxy on
Trillium: `wandb sync` from a login node).

## Layout on `$SCRATCH`

| path | content |
|---|---|
| `$SCRATCH/trillium/qwen/env` | venv with milabench (`uv pip install -e milabench`) |
| `$SCRATCH/trillium/qwen/milabench` | this repository, branch with the Trillium entries |
| `$SCRATCH/trillium/qwen/output` | `MILABENCH_BASE`: bench venv, FineWeb data |
| `$SCRATCH/trillium/qwen/logs` | Slurm output, `<bench>-<job>.out` |
| `$SCRATCH/trillium/qwen/compile-cache` | torch.compile cache archives (per torch / Triton / GPU model) |
| `$SCRATCH/trillium/runs/<run>` | milabench run data (`MILABENCH_DIRS_RUNS`) |
| `$SCRATCH/trillium/nodemon/<run>` | node telemetry (benchmate.nodemon) |
| `$SCRATCH/trillium/checkpoints/<run>` | `tri-ckpt-*` checkpoints, deleted once the load is timed |

Layout and event names: `scripts/trillium/FORMAT.md`. `run.experiment` comes from
`TT_EXPERIMENT`, derived from the entry name (pass `TT_EXPERIMENT=B-Q-weak` for the A-Q
entries run at other node counts, and `TT_NODE_SET` / `TT_CONDITION` for E1).

## Install (login node: it downloads)

```bash
unset PYTHONPATH PIP_CONFIG_FILE; export PATH=$HOME/.local/bin:$PATH UV_CACHE_DIR=$SCRATCH/uv-cache
mkdir -p $SCRATCH/trillium/qwen && cd $SCRATCH/trillium/qwen
git clone -b <branch> https://github.com/milabench/milabench.git
uv venv --python-preference only-managed --python 3.12 --seed env
VIRTUAL_ENV=$PWD/env uv pip install -e ./milabench
export PATH=$PWD/env/bin:$PATH MILABENCH_BASE=$PWD/output MILABENCH_GPU_ARCH=cuda
milabench install --config milabench/config/torchtitan_native_trillium.yaml --select tri-smoke
milabench prepare --config milabench/config/torchtitan_native_trillium.yaml --select tri-smoke
```

The install pins the PyTorch nightly `2.15.0.dev20261005+cu130` and the torchao nightly of
the same day (MXFP8 kernels), both from `download.pytorch.org/whl/nightly/cu130`.

## B200 risks to check in the first job

- **Driver**: the cu130 wheels need an NVIDIA driver >= 580. `tt_native.sh` prints
  `[tt-native] gpu <name>, <driver>` at the top of the log: check it before anything else.
  With an older driver, torch fails at the first CUDA call (`CUDA driver version is
  insufficient`); the fallback would be a cu129 / cu128 nightly of the same day.
- **sm_100**: the cu130 nightly ships sm_100 kernels; Triton / Inductor compile for it on the
  first run (cold compile cache: the archive is keyed by GPU model, so nothing from Rorqual
  is reused). Expect a slow first step.
- **MXFP8** (`tri-smoke-mxfp8`, `tri-c-mxfp8`): needs sm_100 and torchao's 32x32 cast
  kernels. On another GPU the run fails right after `run_start` with
  `precision=mxfp8 unavailable (GPU sm_9.0): ...`. Never run on B200 before the window.
- **Gated DeltaNet kernels** (attn-gym / CuTe) and FlexAttention were only run on H100.
- `/tmp` is in RAM on the diskless nodes (2 TB): the compile cache, the local datasets copy
  and (for `async_with_pinned_mem`) checkpoint staging all live in host memory.

Checkpoint entries (`tri-ckpt-*`, F1) save and load twice per run (`repetition` 1 and 2 in
the events). For `async_with_pinned_mem`, the first save's `blocked_s` also includes
allocating the pinned host buffers: it is an upper bound; use repetition 2 for the steady
state. Loads are timed after the step directory's pages are dropped on every node
(`cache_drop: posix_fadvise_dontneed`).

Smoke order for the bring-up: `tri-smoke` (1 node, 16 layers), `tri-smoke-mxfp8` and
`tri-smoke-mxfp8-tt`, then `tri-ref` (4 nodes).
