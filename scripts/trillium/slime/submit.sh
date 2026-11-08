#!/usr/bin/env bash
# Submit slime entries (benchmarks/slime/dev.yaml) as one bring-up job, nw-glm-rl-NAME, on Trillium's B200 partition;
# job.sbatch installs (offline), prepares and runs them one after the other on NODES nodes (README.md):
#   scripts/trillium/slime/submit.sh NAME TIME NODES [NAME=value...] ENTRY... [-- sbatch args, e.g. --nodelist=trig0145]
# (NAME=value: exported in the job, e.g. SLIME_CONVERT=1 for a conversion job, README.md)
# The job runs $SCRATCH/trillium/new/glm-rl/src as it is: pull origin/trillium-new there first.
set -euo pipefail
B=${SCRATCH:?}/trillium/new/glm-rl
name=nw-glm-rl-$1 time=$2 nodes=$3
shift 3
entries=()
while [ $# -gt 0 ] && [ "$1" != -- ]; do entries+=("$1"); shift; done
[ $# -gt 0 ] && shift
[ ${#entries[@]} -gt 0 ] || { echo "usage: submit.sh NAME TIME NODES [NAME=value...] ENTRY... [-- sbatch args]" >&2; exit 2; }
mkdir -p "$B/slurm"
# the real sbatch: the site's filter refuses above 25 nodes, the admins asked for this one (2026-11-05).
# trig0125 and trig0132 never (one GPU without NVLink each, 2026-11-07): reserved, still allocatable
/opt/slurm/bin/.sbatch --export=NONE --get-user-env --parsable --job-name="$name" --account=rrg-bengioy-ad \
    --partition=compute_b200_full_node --gpus-per-node=b200:8 --reservation=b200-friendly-users --qos=bump \
    --nodes="$nodes" --ntasks-per-node=1 --exclusive --mem=0 --time="$time" --output="$B/slurm/%x-%j.out" \
    --exclude=trig0125,trig0132 --signal=B:USR1@300 \
    "$@" "$B/src/scripts/trillium/slime/job.sbatch" "${entries[@]}"
