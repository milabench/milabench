# Window helpers (Rorqual, MILABENCH2026): source this file, then submit blocks one after the other.
#
#   source scripts/slurm/window.sh
#   window_after                     job ids of every job of the reservation still pending or running (a:b:c)
#   window_dep                       the same as a dependency (afterany:a:b:c), empty when nothing is queued
#   qblock NAME TIME SPEC [SPEC ...] one Qwen block (tt_block.sh specs) as one job, on exactly the nodes its specs
#                                    pick, after every job already in the reservation
#   window_status                    what runs and what waits in the reservation
#
# A block waits for everything submitted before it (afterany: a failed block does not stop the chain), so only one
# block runs at a time. Within a block, runs share no interconnect (scripts/analysis/WINDOW.md).
#
# env: TT_HEALTHY (hostlist of the healthy nodes, from the health check: required by qblock), TT_EXCLUDE (more nodes
# to leave out), WINDOW_RES (default MILABENCH2026), WINDOW_ACCOUNT (default def-bengioy_gpu), TT38 (default
# $SCRATCH/tt38).

WINDOW_RES=${WINDOW_RES:-MILABENCH2026}
WINDOW_ACCOUNT=${WINDOW_ACCOUNT:-def-bengioy_gpu}
TT38=${TT38:-$SCRATCH/tt38}

window_after() {
    squeue -h --reservation="$WINDOW_RES" -t PENDING,CONFIGURING,RUNNING -o %i | paste -sd:
}

window_dep() {
    local after
    after=$(window_after)
    echo "${after:+afterany:$after}"
}

window_status() {
    squeue --reservation="$WINDOW_RES" -o "%.10i %.9u %.24j %.2t %.10M %.10l %.5D %R" --sort=i
}

qblock() {
    local name=$1 time=$2
    shift 2
    [ $# -gt 0 ] || { echo "usage: qblock NAME TIME SPEC [SPEC ...]" >&2; return 2; }
    [ -n "${TT_HEALTHY:-}" ] || { echo "qblock: set TT_HEALTHY to the health check's healthy hostlist" >&2; return 2; }
    local plan nodes n after
    # the same picks as in the job: tt_block.sh picks from TT_HEALTHY minus TT_EXCLUDE, deterministically
    plan=$(TT_DRY_RUN=1 bash "$TT38/milabench/scripts/slurm/tt_block.sh" "$@") || { echo "$plan" >&2; return 1; }
    echo "$plan"
    nodes=$(sed -n 's/^\[tt-block\] spec=.* nodes=\([^ ]*\)$/\1/p' <<<"$plan" | paste -sd,)
    [ -n "$nodes" ] || { echo "qblock: no node picked" >&2; return 1; }
    n=$(tr , '\n' <<<"$nodes" | grep -c .)
    after=$(window_after)
    sbatch --parsable --job-name="q-$name" --reservation="$WINDOW_RES" --account="$WINDOW_ACCOUNT" \
        --nodes="$n" --nodelist="$nodes" --gpus-per-node=h100:4 --ntasks-per-node=1 --exclusive --mem=0 \
        --time="$time" --output="$TT38/logs/%x-%j.out" ${after:+--dependency=afterany:$after} --export=ALL \
        --wrap "bash $TT38/milabench/scripts/slurm/tt_block.sh $*"
}
