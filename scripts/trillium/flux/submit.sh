#!/usr/bin/env bash
# Submit Florida's FLUX.1 MLPerf training (flux.sub, Trillium-adapted) as one job, nw-flux-NAME, on Trillium's B200
# partition; the node count picks Florida's config as their flux_training_scale.sh does (README.md):
#   scripts/trillium/flux/submit.sh NAME NODES TIME [NAME=value ...] [-- sbatch args, e.g. --nodelist=trig0145]
# NAME=value: exported in the job before the config, e.g. MAX_STEPS=300, USE_SYNTHETIC_DATA=1, NEXP=1 (the default).
# SCRIPT=convert.sbatch submits the data conversion instead of flux.sub. SRC=<checkout> runs another checkout than
# $F/src (the rows run from $F/src; tests from e.g. $F/src-dev).
# The job runs $SCRATCH/trillium/new/flux/src as it is: pull origin/trillium-new there first.
set -euo pipefail
F=${SCRATCH:?}/trillium/new/flux
S=${SRC:-$F/src}
name=nw-flux-$1 nodes=$2 time=$3
shift 3
args=()
while [ $# -gt 0 ] && [ "$1" != -- ]; do args+=("$1"); shift; done
[ $# -gt 0 ] && shift
mkdir -p "$F/slurm"
# the real sbatch: the site's filter refuses above 25 nodes, the admins asked for this one (2026-11-05).
# trig0114 never (ops/BRIEF.md); trig0118 out (GPU 2 lost at 21:56 on 2026-11-09, HW-33) until our recheck passes
/opt/slurm/bin/.sbatch --export=NONE --get-user-env --parsable --job-name="$name" --account=rrg-bengioy-ad \
    --partition=compute_b200_full_node --gpus-per-node=b200:8 --reservation=b200-friendly-users --qos=bump \
    --nodes="$nodes" --time="$time" --output="$F/slurm/%x-%j.out" --exclude=trig0114,trig0118 \
    "$@" "$S/scripts/trillium/flux/${SCRIPT:-flux.sub}" "SRC=$S" "${args[@]}"
