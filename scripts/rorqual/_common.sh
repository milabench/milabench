#!/usr/bin/env bash
# Shared by the numbered GLM-5 step scripts (Megatron-LM, Rorqual, 4 GPUs per node). Source it, then call `submit`.
#
# env (all optional):
#   MILABENCH_BASE   shared base: venv, data, runs, slurm logs     (default /home/delaunay/links/scratch/shared)
#   WORKDIR          the milabench checkout the job runs            (default $MILABENCH_BASE/src)
#   CONFIG           (default $WORKDIR/benchmarks/megatron/dev.yaml)
#   RESERVATION / ACCOUNT / GPUS                                    (default MILABENCH2026 / def-bengioy_gpu / 4)
#   NODES            nodes of the allocation (default 66: every healthy full-GPU node; each entry picks and orders its
#                    own set from it, see benchmarks/megatron/placement.py)
#   EXCLUDE          hostlist to leave out (unhealthy nodes)        -> sbatch --exclude
#   NODELIST         hostlist the job must use                      -> sbatch --nodelist. Keep the SAME one for every step
#                    that is compared (G1, G2, G3, G5): the comparison is only valid on one set of nodes
#   AFTER            job id to wait for (afterany)
#   BRANCH / UPDATE  the job runs $WORKDIR as it is: it must be origin/$BRANCH's latest (default large_scale); UPDATE=1
#                    fast-forwards it (default 1), UPDATE=0 fails instead
#   PREPARE=0        skip `milabench prepare` in the job (data already staged)
#   metrics: BENCHMATE_RECORD_FINE / _OVERHEAD / _LAPS (default 1), BENCHMATE_PUSH_EVERY (10),
#            BENCHMATE_WANDB / _ALL_RANKS (1), BENCHMATE_WANDB_PROJECT (milabench-glm), WANDB_DIR ($MILABENCH_BASE/wandb)
#   nodemon: always on, under $SCRATCH/nodemon/<run name> (FORMAT.md §3). job.sbatch sets it,
#            one directory per entry.
#   run record: CAMPAIGN (default glm5-MMDD), HEALTH_DIR (health-check results), EXCLUSIVE_CLUSTER (1 in the window),
#            EXCLUDE_REASONS (JSON {host: reason} for the EXCLUDE nodes)
#   NO_TAIL=1        do not follow the log      TEST_ONLY=1   sbatch --test-only, nothing submitted
set -euo pipefail

RORQUAL=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")
export MILABENCH_BASE=${MILABENCH_BASE:-/home/delaunay/links/scratch/shared}
export WORKDIR=${WORKDIR:-$MILABENCH_BASE/src}
export CONFIG=${CONFIG:-$WORKDIR/benchmarks/megatron/dev.yaml}
RESERVATION=${RESERVATION:-MILABENCH2026}
ACCOUNT=${ACCOUNT:-def-bengioy_gpu}
GPUS=${GPUS:-h100:4}   # typed: the MIG nodes of the reservation would match a plain count
# the reservation's 66 full-GPU nodes minus the EXCLUDE ones (asking for more nodes than remain never starts)
excluded=0
[ -n "${EXCLUDE:-}" ] && excluded=$(scontrol show hostnames "$EXCLUDE" | wc -l)
NODES=${NODES:-$((66 - excluded))}
BRANCH=${BRANCH:-large_scale}
UPDATE=${UPDATE:-1}

# metrics, handed to the job (--export=ALL) and from there to every rank
#   fine-grained step timings (work / next per batch, overhead, laps), flushed every BENCHMATE_PUSH_EVERY steps (one GPU sync)
export BENCHMATE_RECORD_FINE=${BENCHMATE_RECORD_FINE:-1}
export BENCHMATE_RECORD_OVERHEAD=${BENCHMATE_RECORD_OVERHEAD:-1}
export BENCHMATE_RECORD_LAPS=${BENCHMATE_RECORD_LAPS:-1}
export BENCHMATE_PUSH_EVERY=${BENCHMATE_PUSH_EVERY:-10}
#   wandb: one run per rank (same group = the milabench run), online through the cluster's proxy (job.sbatch loads the
#   httpproxy module). The API key is not set here: it comes from your environment / ~/.netrc (`wandb login`).
#   Each job keeps its local wandb files under $WANDB_DIR/<job-name>.<jobid> (job.sbatch).
export BENCHMATE_WANDB=${BENCHMATE_WANDB:-1}
export BENCHMATE_WANDB_ALL_RANKS=${BENCHMATE_WANDB_ALL_RANKS:-1}
export BENCHMATE_WANDB_PROJECT=${BENCHMATE_WANDB_PROJECT:-milabench-glm}
export WANDB_DIR=${WANDB_DIR:-$MILABENCH_BASE/wandb}

# run identity for the run_start record (scripts/analysis/FORMAT.md, "What the launcher sets"). job.sbatch adds
# TT_TOPOLOGY (topology snapshot), TT_EXCLUDED (from EXCLUDE) and TT_VARIANT per entry; _g.sh sets VARIANT / REPETITION / FAMILY
export TT_CAMPAIGN=${CAMPAIGN:-${TT_CAMPAIGN:-glm5-$(date +%m%d)}}             # groups the runs of one session
export TT_HEALTH_DIR=${HEALTH_DIR:-${TT_HEALTH_DIR:-}}                          # e.g. srygaard:/scratch/.../results/25045100
export TT_EXCLUSIVE=${EXCLUSIVE_CLUSTER:-${TT_EXCLUSIVE:-0}}                    # 1 in the window, when the cluster is ours

# login node (internet): the checkout must be the branch's latest, the venv, data and Megatron-LM must be there
preflight() {
    if git -C "$WORKDIR" fetch --quiet origin "$BRANCH"; then
        local behind
        behind=$(git -C "$WORKDIR" rev-list --count "HEAD..origin/$BRANCH")
        if [ "$behind" != 0 ]; then
            [ "$UPDATE" = 1 ] || { echo "$WORKDIR is $behind commit(s) behind origin/$BRANCH: rerun with UPDATE=1" >&2; exit 1; }
            git -C "$WORKDIR" merge --ff-only "origin/$BRANCH"
        fi
        [ -z "$(git -C "$WORKDIR" status --porcelain --untracked-files=no)" ] \
            || echo "WARNING: $WORKDIR has local modifications, the job runs them" >&2
        echo "milabench $(git -C "$WORKDIR" rev-parse --short HEAD) ($BRANCH, up to date)"
    else
        echo "WARNING: could not fetch origin/$BRANCH, running $(git -C "$WORKDIR" rev-parse --short HEAD) unchecked" >&2
    fi
    [ -x "$WORKDIR/.venv/bin/milabench" ] || { echo "no milabench in $WORKDIR/.venv: run slurm/download_megatron_data.sh" >&2; exit 1; }
    [ -d "$MILABENCH_BASE/data/megatron" ] || { echo "no $MILABENCH_BASE/data/megatron: run slurm/download_megatron_data.sh" >&2; exit 1; }
    [ -d "$WORKDIR/benchmarks/megatron/Megatron-LM" ] || { echo "no Megatron-LM in $WORKDIR/benchmarks/megatron: run slurm/download_megatron_data.sh" >&2; exit 1; }
    if [ "$BENCHMATE_WANDB" = 1 ] && [ -z "${WANDB_API_KEY:-}" ] && ! grep -qs api.wandb.ai ~/.netrc; then
        echo "WARNING: no WANDB_API_KEY and no api.wandb.ai in ~/.netrc: the wandb runs may not log in (wandb login)" >&2
    fi
}

# job id from submit's output ("submitted job N")
job_id() {
    printf '%s\n' "$1" | sed -n 's/^submitted job \([0-9]*\).*/\1/p' | tail -1
}

# where 01_check_full.sh's job writes the rung that fits for a node set (50 | 64)
rung_file() {
    echo "$MILABENCH_BASE/slurm/glm5-full-rung.${1}n"
}

# full_rung 50|64: the suffix of the full entries for that set, from $FULL if set, else from 01_check_full.sh's result:
#   "" (as configured) | -offload025 | -offload05 | -offload1 | -balanced (50 only) | none (full training does not fit)
full_rung() {
    local rung
    if [ -n "${FULL+x}" ]; then
        rung=$FULL
    else
        local file
        file=$(rung_file "$1")
        [ -s "$file" ] || { echo "no $file: run 01_check_full.sh first, or set FULL (''|offload025|offload05|offload1|balanced|none)" >&2; return 1; }
        rung=$(awk '{print $1}' "$file")
        case $rung in
            NONE) rung=none ;;
            *) rung=${rung#megatron-glm5-check-fit-full}; rung=${rung%-${1}n}; rung=${rung#-} ;;
        esac
    fi
    case $rung in
        none) echo none ;;
        "") echo "" ;;
        offload025|offload05|offload1|balanced) echo "-$rung" ;;
        *) echo "unknown full rung '$rung'" >&2; return 1 ;;
    esac
}

# submit NAME TIME ENTRY [ENTRY ...] [-- extra sbatch args]
#   one job on $NODES nodes; job.sbatch runs the entries one after the other (STOP_AT_FIRST=1: up to the first that completes)
submit() {
    local name=$1 time=$2
    shift 2
    local entries=()
    while [ $# -gt 0 ] && [ "$1" != -- ]; do entries+=("$1"); shift; done
    [ $# -gt 0 ] && shift
    [ ${#entries[@]} -gt 0 ] || { echo "submit: no entry" >&2; exit 2; }

    preflight
    mkdir -p "$MILABENCH_BASE/slurm"
    local args=(--job-name="$name" --reservation="$RESERVATION" --account="$ACCOUNT"
                --nodes="$NODES" --ntasks-per-node=1 --exclusive --mem=0 --gpus-per-node="$GPUS" --time="$time"
                --output="$MILABENCH_BASE/slurm/$name-%j.out" --export=ALL)
    [ -n "${EXCLUDE:-}" ]  && args+=(--exclude="$EXCLUDE")
    [ -n "${NODELIST:-}" ] && args+=(--nodelist="$NODELIST")
    [ -n "${AFTER:-}" ]    && args+=(--dependency="afterany:$AFTER")
    [ -n "${TEST_ONLY:-}" ] && args+=(--test-only)

    # --export=ALL hands these to the job (ENTRIES is space separated)
    export ENTRIES="${entries[*]}" RUN_PREFIX=$name STOP_AT_FIRST=${STOP_AT_FIRST:-0} PREPARE=${PREPARE:-1}
    export RESULT_FILE=${RESULT_FILE:-}
    echo "[$name] nodes=$NODES time=$time ${EXCLUDE:+exclude=$EXCLUDE }${NODELIST:+nodelist=$NODELIST }${AFTER:+after=$AFTER }"
    printf '  %s\n' "${entries[@]}"
    scontrol show reservation "$RESERVATION" 2>/dev/null | head -2 || true

    local jobid
    jobid=$(sbatch --parsable "${args[@]}" "$@" "$RORQUAL/job.sbatch")
    [ -n "${TEST_ONLY:-}" ] && return 0
    echo "submitted job $jobid"
    [ -n "${NO_TAIL:-}" ] && return 0
    local log=$MILABENCH_BASE/slurm/$name-${jobid%%;*}.out
    touch "$log"
    tail -f "$log"   # ctrl-c stops following, not the job
}
