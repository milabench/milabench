#!/usr/bin/env bash
# Submit slime entries (benchmarks/slime/dev.yaml) as one bring-up job, nw-glm-rl-NAME, on Trillium's B200 partition;
# job.sbatch installs (offline), prepares and runs them one after the other on NODES nodes (README.md):
#   scripts/trillium/slime/submit.sh NAME TIME NODES ENTRY... [-- sbatch args, e.g. --nodelist=trig0145]
# The job runs $SCRATCH/trillium/new/glm-rl/src as it is: pull origin/trillium-new there first.
set -euo pipefail
B=${SCRATCH:?}/trillium/new/glm-rl
name=nw-glm-rl-$1 time=$2 nodes=$3
shift 3
entries=()
while [ $# -gt 0 ] && [ "$1" != -- ]; do entries+=("$1"); shift; done
[ $# -gt 0 ] && shift
[ ${#entries[@]} -gt 0 ] || { echo "usage: submit.sh NAME TIME NODES ENTRY... [-- sbatch args]" >&2; exit 2; }
mkdir -p "$B/slurm"
# the real sbatch: the site's filter refuses above 25 nodes, the admins asked for this one (2026-11-05)
/opt/slurm/bin/.sbatch --export=NONE --get-user-env --parsable --job-name="$name" --account=rrg-bengioy-ad \
    --partition=compute_b200_full_node --gpus-per-node=b200:8 --reservation=b200-friendly-users --qos=bump \
    --nodes="$nodes" --ntasks-per-node=1 --exclusive --mem=0 --time="$time" --output="$B/slurm/%x-%j.out" \
    "$@" "$B/src/scripts/trillium/slime/job.sbatch" "${entries[@]}"
