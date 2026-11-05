#!/usr/bin/env bash
# Shared by the GLM-5 step scripts on Trillium's B200 partition (Megatron-LM, 8 GPUs per node). Source it, then call
# `submit NAME TIME NODES ENTRY...`. Every entry of benchmarks/megatron/dev.yaml runs on exactly its num_machines nodes:
# one job holds entries of one node count only.
#
# env (all optional):
#   MILABENCH_BASE   venv, data, runs, slurm logs                   (default $SCRATCH/trillium/glm; $HOME is read-only
#                                                                    from the compute nodes: everything lives on scratch)
#   WORKDIR          the milabench checkout the job runs            (default $MILABENCH_BASE/src)
#   MILABENCH_VENV   venv with the milabench CLI                    (default $WORKDIR/.venv)
#   CONFIG           (default $WORKDIR/benchmarks/megatron/dev.yaml)
#   ACCOUNT / PARTITION / GPUS / RESERVATION / QOS                  (default rrg-bengioy-ad / compute_b200_full_node /
#                                                                    b200:8 / b200-friendly-users / bump)
#   CUDA_MODULE      module loaded by the job, after StdEnv/2023 gcc/12.3 (default cuda/12.9; empty: none)
#   EXCLUDE / NODELIST  hostlists -> sbatch --exclude / --nodelist (EXCLUDE default trig0122; empty: none)
#   AFTER            job id to wait for (afterany)
#   BRANCH / UPDATE  the job runs $WORKDIR as it is: it must be origin/$BRANCH's latest (default trillium); UPDATE=1
#                    fast-forwards it (default 1), UPDATE=0 fails instead
#   PREPARE=0        skip `milabench prepare` in the job (data already staged)
#   REPETITION       run.repetition (default 1); CAMPAIGN (default trillium-MMDD); EXCLUSIVE_CLUSTER (1 in the window)
#   CKPT_ROOT        where F1 writes its checkpoints (default $SCRATCH/trillium/checkpoints)
#   RUNS_DIR         run data, $MILABENCH_BASE/runs links there (default $SCRATCH/trillium/runs, scripts/trillium/FORMAT.md)
#   metrics: BENCHMATE_RECORD_FINE / _OVERHEAD / _LAPS (default 1), BENCHMATE_PUSH_EVERY (10),
#            BENCHMATE_WANDB / _ALL_RANKS (1), BENCHMATE_WANDB_PROJECT (milabench-glm), WANDB_DIR ($MILABENCH_BASE/wandb)
#   nodemon: always on, $SCRATCH/trillium/nodemon/<run name>
#   NO_TAIL=1        do not follow the log      TEST_ONLY=1   sbatch --test-only, nothing submitted
#   DRY_RUN=1        print the sbatch command and the entries, submit nothing (no Slurm, no git needed)
set -euo pipefail

GLM=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")
export MILABENCH_BASE=${MILABENCH_BASE:-${SCRATCH:?}/trillium/glm}
export WORKDIR=${WORKDIR:-$MILABENCH_BASE/src}
export MILABENCH_VENV=${MILABENCH_VENV:-$WORKDIR/.venv}
export CONFIG=${CONFIG:-$WORKDIR/benchmarks/megatron/dev.yaml}
export CUDA_MODULE=${CUDA_MODULE-cuda/12.9}
export CKPT_ROOT=${CKPT_ROOT:-$SCRATCH/trillium/checkpoints}
RUNS_DIR=${RUNS_DIR:-$SCRATCH/trillium/runs}
export NODEMON_ROOT=${NODEMON_ROOT:-$SCRATCH/trillium/nodemon}
ACCOUNT=${ACCOUNT:-rrg-bengioy-ad}
PARTITION=${PARTITION-compute_b200_full_node}
GPUS=${GPUS:-b200:8}
RESERVATION=${RESERVATION-b200-friendly-users}
EXCLUDE=${EXCLUDE-trig0122}  # mlx5_1 firmware, 14% slower pairs: until the admins update it and a health check passes
QOS=${QOS-bump}  # normal stops at 100 GPUs per user, 12 nodes
BRANCH=${BRANCH:-trillium}
UPDATE=${UPDATE:-1}

# metrics, handed to the job (--export=ALL) and from there to every rank
export BENCHMATE_RECORD_FINE=${BENCHMATE_RECORD_FINE:-1}
export BENCHMATE_RECORD_OVERHEAD=${BENCHMATE_RECORD_OVERHEAD:-1}
export BENCHMATE_RECORD_LAPS=${BENCHMATE_RECORD_LAPS:-1}
export BENCHMATE_PUSH_EVERY=${BENCHMATE_PUSH_EVERY:-10}
export BENCHMATE_WANDB=${BENCHMATE_WANDB:-1}
export BENCHMATE_WANDB_ALL_RANKS=${BENCHMATE_WANDB_ALL_RANKS:-1}
export BENCHMATE_WANDB_PROJECT=${BENCHMATE_WANDB_PROJECT:-milabench-glm}
export WANDB_ENTITY=${WANDB_ENTITY:-brdg-mila}
export WANDB_DIR=${WANDB_DIR:-$MILABENCH_BASE/wandb}

# run identity for run_start (scripts/analysis/FORMAT.md, scripts/trillium/FORMAT.md). The step scripts set
# TT_EXPERIMENT (A-G, B-G-weak, C-speed, C-loss, F1, ...; an entry given as ENTRY@EXPERIMENT overrides it for that
# run); job.sbatch sets TT_VARIANT (the entry) per run
export TT_CAMPAIGN=${CAMPAIGN:-${TT_CAMPAIGN:-trillium-$(date +%m%d)}}
export TT_EXPERIMENT=${TT_EXPERIMENT:-}
export TT_REPETITION=${REPETITION:-${TT_REPETITION:-1}}
export TT_HEALTH_DIR=${HEALTH_DIR:-${TT_HEALTH_DIR:-}}
export TT_EXCLUSIVE=${EXCLUSIVE_CLUSTER:-${TT_EXCLUSIVE:-0}}
export TT_FAMILY=${TT_FAMILY:-slurm}

# login node: the checkout must be the branch's latest; the venvs, data and Megatron-LM must be there
preflight() {
    if timeout 120 git -C "$WORKDIR" fetch --quiet origin "$BRANCH"; then  # a hung fetch (cvmfs, network) must not block
        local behind
        behind=$(git -C "$WORKDIR" rev-list --count "HEAD..origin/$BRANCH")
        if [ "$behind" != 0 ]; then
            [ "$UPDATE" = 1 ] || { echo "$WORKDIR is $behind commit(s) behind origin/$BRANCH: rerun with UPDATE=1" >&2; exit 1; }
            git -C "$WORKDIR" merge --ff-only "origin/$BRANCH"
        fi
        echo "milabench $(git -C "$WORKDIR" rev-parse --short HEAD) ($BRANCH, up to date)"
    else
        echo "WARNING: could not fetch origin/$BRANCH, running $(git -C "$WORKDIR" rev-parse --short HEAD) unchecked" >&2
    fi
    [ -x "$MILABENCH_VENV/bin/milabench" ] || { echo "no milabench in $MILABENCH_VENV (README: install)" >&2; exit 1; }
    [ -d "$MILABENCH_BASE/venv/torch" ] || { echo "no $MILABENCH_BASE/venv/torch (README: install)" >&2; exit 1; }
    # written by the install once the venv imports TransformerEngine (a build that failed leaves no marker)
    [ -f "$MILABENCH_BASE/venv/torch/.slime_installed" ] || { echo "no $MILABENCH_BASE/venv/torch/.slime_installed: the install did not finish, see its log" >&2; exit 1; }
    echo "install: $(cat "$MILABENCH_BASE/venv/torch/.slime_installed")"
    [ -d "$MILABENCH_BASE/data/megatron" ] || { echo "no $MILABENCH_BASE/data/megatron (README: install)" >&2; exit 1; }
    [ -d "$WORKDIR/benchmarks/megatron/Megatron-LM" ] || { echo "no Megatron-LM in $WORKDIR/benchmarks/megatron" >&2; exit 1; }
    # the run data where the analysis copies it from (milabench writes to $MILABENCH_BASE/runs)
    if [ ! -e "$MILABENCH_BASE/runs" ]; then
        mkdir -p "$RUNS_DIR" && ln -s "$RUNS_DIR" "$MILABENCH_BASE/runs"
    fi
}

# refuse a checkpoint job while CKPT_ROOT holds something (job.sbatch checks again when it starts)
ckpt_root_empty() {
    [ -n "${DRY_RUN:-}" ] && return 0
    if [ -n "$(ls -A "$CKPT_ROOT" 2>/dev/null)" ]; then
        echo "$CKPT_ROOT is not empty: delete what is there or wait for its job" >&2
        ls -la "$CKPT_ROOT" >&2
        exit 3
    fi
}

# job id from submit's output ("submitted job N")
job_id() {
    printf '%s\n' "$1" | sed -n 's/^submitted job \([0-9]*\).*/\1/p' | tail -1
}

# submit NAME TIME NODES ENTRY[@EXPERIMENT] [...] [-- extra sbatch args]
#   one job on NODES nodes; job.sbatch runs the entries one after the other
submit() {
    local name=$1 time=$2 nodes=$3
    shift 3
    local entries=()
    while [ $# -gt 0 ] && [ "$1" != -- ]; do entries+=("$1"); shift; done
    [ $# -gt 0 ] && shift
    [ ${#entries[@]} -gt 0 ] || { echo "submit: no entry" >&2; exit 2; }

    # JOB_NAME + SINGLETON=1: at most one job of that name runs at a time (F1: one checkpoint on disk)
    local dep=${AFTER:+afterany:$AFTER}
    [ -n "${SINGLETON:-}" ] && dep=${dep:+$dep,}singleton
    local args=(--job-name="${JOB_NAME:-$name}" --account="$ACCOUNT" --nodes="$nodes" --ntasks-per-node=1 --exclusive
                --gpus-per-node="$GPUS" --time="$time" --output="$MILABENCH_BASE/slurm/$name-%j.out" --export=ALL)
    [ -n "$PARTITION" ]        && args+=(--partition="$PARTITION")
    [ -n "$RESERVATION" ]      && args+=(--reservation="$RESERVATION")
    [ -n "$QOS" ]              && args+=(--qos="$QOS")
    [ -n "${EXCLUDE:-}" ]      && args+=(--exclude="$EXCLUDE")
    [ -n "${NODELIST:-}" ]     && args+=(--nodelist="$NODELIST")
    [ -n "$dep" ]              && args+=(--dependency="$dep")
    [ -n "${TEST_ONLY:-}" ]    && args+=(--test-only)

    export ENTRIES="${entries[*]}" RUN_PREFIX=$name PREPARE=${PREPARE:-1}
    echo "[$name] nodes=$nodes time=$time experiment=${TT_EXPERIMENT:-} rep=$TT_REPETITION ${EXCLUDE:+exclude=$EXCLUDE }${NODELIST:+nodelist=$NODELIST }${AFTER:+after=$AFTER }"
    printf '  %s\n' "${entries[@]}"
    if [ -n "${DRY_RUN:-}" ]; then
        echo "  sbatch ${args[*]} $* $GLM/job.sbatch"
        return 0
    fi
    preflight
    mkdir -p "$MILABENCH_BASE/slurm"
    local jobid
    jobid=$(sbatch --parsable "${args[@]}" "$@" "$GLM/job.sbatch")
    [ -n "${TEST_ONLY:-}" ] && return 0
    echo "submitted job $jobid"
    [ -n "${NO_TAIL:-}" ] && return 0
    local log=$MILABENCH_BASE/slurm/$name-${jobid%%;*}.out
    touch "$log"
    tail -f "$log"   # ctrl-c stops following, not the job
}
