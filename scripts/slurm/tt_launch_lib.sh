# Shared by tt_variants.sh and tt_parallel.sh (sourced): run identity recorded
# in each run's summary (scripts/analysis/FORMAT.md, "What the launcher sets").
#
#   TT_CAMPAIGN           groups the runs of one session (default: adhoc)
#   TT_EXCLUSIVE_CLUSTER  1 in a window where the cluster is ours
#   TT_TOPOLOGY           `scontrol show topology` saved once per job
#   TT_HEALTH_DIR         health check of this job, if any
#   TT_VARIANT, TT_REPETITION, TT_FAMILY   per run, set by the launcher

export TT_CAMPAIGN=${TT_CAMPAIGN:-adhoc}
export TT_EXCLUSIVE_CLUSTER=${TT_EXCLUSIVE_CLUSTER:-0}

tt_snapshot_topology() {
    local dir=$TT38/health/$SLURM_JOB_ID
    mkdir -p "$dir"
    [ -s "$dir/topology.txt" ] || scontrol show topology > "$dir/topology.txt"
    export TT_TOPOLOGY=$dir/topology.txt
    if [ -z "${TT_HEALTH_DIR:-}" ] && [ -f "$dir/healthy.txt" ]; then
        export TT_HEALTH_DIR=$dir
    fi
}

# Next repetition index of VARIANT BENCH in the campaign (1, 2, ...), shared by
# every job of the campaign.
tt_next_repetition() {
    local variant=$1 bench=$2
    local file=$TT38/campaigns/$TT_CAMPAIGN/repetitions.tsv
    mkdir -p "$(dirname "$file")"
    touch "$file"
    (
        flock 9
        local n
        n=$(awk -F'\t' -v v="$variant" -v b="$bench" '$1 == v && $2 == b {n++} END {print n + 1}' "$file")
        printf '%s\t%s\t%s\t%s\n' "$variant" "$bench" "$n" "$SLURM_JOB_ID" >> "$file"
        echo "$n"
    ) 9>"$file.lock"
}
