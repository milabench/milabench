#!/bin/bash


# Test of the per-rank metric store (benchmate/rankstore.py) + wandb + fine grained timers on the Megatron pack:
# 1 node x 2 A100 with a tiny stand-in model (megatron-tiny-moe-smoke-1n: 2-layer MoE GPT, EP2, plain attention): same
# Megatron pack, not GLM-5. Rank 0 does not write to the rank store, so 2 ranks are the minimum to get a file (rank 1).
# GLM-5 on one A100 (EP1, no rank-store file): MILABENCH_SELECT=megatron-glm5-frozen-smoke-1gpu. Multi-node variant: megatron-glm5-frozen-smoke-2n-l40s, 2 nodes x 2 L40S, ranks 2-3 on the other node:
#   sbatch -N 2 --ntasks-per-node=1 --gres=gpu:l40s:2 --cpus-per-task=16 --mem=128G -t 1:30:00 scripts/slurm/run_rankstore.sh
# Default (2 A100):
#   sbatch -N 1 --ntasks-per-node=1 --gres=gpu:a100:2 --cpus-per-task=16 --mem=128G -t 1:30:00 scripts/slurm/run_rankstore.sh
# The branch must contain rankstore.py and the megatron pack (push them first).
export MILABENCH_BRANCH=metrics_mega
export PYTHON_VERSION=3.12
export MILABENCH_GPU_ARCH=cuda
export HF_TOKEN="{{ secrets.HF_TOKEN }}"
export MILABENCH_REPO=https://github.com/milabench/milabench.git
export CUDA_VERSION="130"
# TE / sglang / slime's GLM-5 plugin are pinned to torch 2.13 (same as the slime pack)
export PYTORCH_VERSION="2.13.0"

# the Megatron pack lives in its own config; random weights, only GLM-5.3's config.json is downloaded
export CONFIG_PATH=benchmarks/megatron/dev.yaml
export MILABENCH_ARGS="--select ${MILABENCH_SELECT:-megatron-tiny-moe-smoke-1n}"

# --- what is under test ---
export BENCHMATE_RECORD_FINE=1
export BENCHMATE_RECORD_OVERHEAD=1
export BENCHMATE_RECORD_LAPS=1
# 1: sync the GPU and log each step's rate right when it ends (events in step order, but a sync per step);
# 0: keep the events and push them in batches (no per-step sync, `work` lands one step late)
export BENCHMATE_EAGER_SYNC=${BENCHMATE_EAGER_SYNC:-1}
export BENCHMATE_WANDB=1
export BENCHMATE_WANDB_PROJECT=milabench-rankstore-test
export WANDB_API_KEY="{{ secrets.WANDB_API_KEY }}"
# no key (or no network on the node): keep the run, sync later with `wandb sync`
# every rank writes <run dir>/metrics.<bench>.<host>.<rank>.jsonl (rankstore); override the base name here if wanted
# export MILABENCH_LOCAL_METRICS=$HOME/scratch/rankstore/metrics.jsonl

set -ex

# ===
OUTPUT_DIRECTORY=$(scontrol show job "$SLURM_JOB_ID" --json | jq -r '.jobs[0].standard_output' | xargs dirname)
export OUTPUT_DIRECTORY
mkdir -p $OUTPUT_DIRECTORY/meta
scontrol show job --json $SLURM_JOB_ID | jq '.jobs[0]' > $OUTPUT_DIRECTORY/meta/info.json
# ===

# module load cuda/12.6.0

# Node-local /tmp on every node, same path everywhere: each node builds its own identical copy (SETUP below, one srun
# task per node), so the worker launched by `srun -w <node>` finds the same venv / code / data as the main node.
export MILABENCH_WORDIR="/tmp/$SLURM_JOB_ID/$MILABENCH_GPU_ARCH"
export MILABENCH_BASE="$MILABENCH_WORDIR/results"
export MILABENCH_ENV="$MILABENCH_WORDIR/.env/$PYTHON_VERSION/"
export BENCHMARK_VENV="$MILABENCH_WORDIR/results/venv/torch"
export MILABENCH_SIZER_SAVE="$MILABENCH_WORDIR/scaling.yaml"
export MILABENCH_HF_TOKEN="$HF_TOKEN"
export HF_TOKEN
export MILABENCH_SHARED="$HOME/scratch/shared"
export MILABENCH_SOURCE="$MILABENCH_WORDIR/milabench"
export MILABENCH_CONFIG="$MILABENCH_SOURCE/$CONFIG_PATH"
export UV=$HOME/.local/bin/uv
export MILABENCH_USE_TOML_DEPS=1
# the TE / sglang builds need the CUDA toolkit and a few cores
export MAX_JOBS=${SLURM_CPUS_PER_TASK:-8}

# --- on EVERY node: clone, venv, install, prepare (no --system: each node only prepares itself) ---
read -r -d '' SETUP <<'SETUP_END' || true
set -ex
mkdir -p $MILABENCH_WORDIR/results/runs
cd $MILABENCH_WORDIR
git clone $MILABENCH_REPO -b $MILABENCH_BRANCH
$UV venv --python=$PYTHON_VERSION $MILABENCH_ENV
. $MILABENCH_ENV/bin/activate
$UV pip install -e $MILABENCH_SOURCE[$MILABENCH_GPU_ARCH]

# dev.yaml pins WANDB_MODE=offline in the pack env, which wins over ours: follow the mode chosen above
sed -i "s/WANDB_MODE: .*/WANDB_MODE: \"${WANDB_MODE:-online}\"/" $MILABENCH_CONFIG

# a failing stage must not stop the later ones: we want every issue from one job
milabench install --config $MILABENCH_CONFIG --set cuda=$CUDA_VERSION torch=$PYTORCH_VERSION $MILABENCH_ARGS || echo "STAGE FAILED: install on $(hostname)"
milabench prepare --config $MILABENCH_CONFIG $MILABENCH_ARGS || echo "STAGE FAILED: prepare on $(hostname)"
SETUP_END
srun --nodes=$SLURM_NNODES --ntasks-per-node=1 bash -c "$SETUP"

# --- main node: describe the allocation, run (the workers of the other nodes are started by srun) ---
. $MILABENCH_ENV/bin/activate
milabench slurm system > $MILABENCH_WORDIR/system.yaml
cat $MILABENCH_WORDIR/system.yaml

milabench run --config $MILABENCH_CONFIG --system $MILABENCH_WORDIR/system.yaml $MILABENCH_ARGS || echo "STAGE FAILED: run"

# --- every node: copy its runs (rank store files, wandb offline runs) next to the job output ---
read -r -d '' GATHER <<'GATHER_END' || true
dest=$OUTPUT_DIRECTORY/runs/$(hostname)
mkdir -p $dest
rsync -az $MILABENCH_WORDIR/results/runs/ $dest/
for d in $(find $MILABENCH_WORDIR -maxdepth 6 -type d -name wandb 2>/dev/null); do
    rsync -az $d $dest/
done
GATHER_END
srun --nodes=$SLURM_NNODES --ntasks-per-node=1 bash -c "$GATHER"

# --- what did the test produce? ---
set +x
R=$OUTPUT_DIRECTORY/runs
echo "== rank stores (expect one metrics.<bench>.<host>.<rank>.jsonl per GPU) =="
find $R -name 'metrics.*.jsonl' -exec sh -c 'for f; do echo "$f: $(wc -l < "$f") records"; done' sh {} +
echo "== metric names seen in the rank stores =="
cat $(find $R -name 'metrics.*.jsonl') 2>/dev/null \
    | python3 -c 'import sys,json,collections; c=collections.Counter(k for l in sys.stdin for k in json.loads(l) if k not in ("rank","time","task","units","batch_id")); print(dict(c))'
echo "== rank 0 events (smuggled stdout, .data) =="
cat $(find $R -name 'megatron-*.data') 2>/dev/null | grep -o '"\(rate\|loss\|a2a_time\|next\|work\|step\|overhead\|params_trainable_fraction\)": ' | sort | uniq -c
echo "== wandb =="
find $R -maxdepth 4 -path '*wandb/*run-*' 2>/dev/null | head -3
echo "WANDB_MODE=${WANDB_MODE:-online}  (offline: wandb sync <dir>)"
set -x

# ===
scontrol show job --json $SLURM_JOB_ID | jq '.jobs[0]' > $OUTPUT_DIRECTORY/meta/info.json
# ===
