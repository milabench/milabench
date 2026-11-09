#!/usr/bin/env bash
# Submit one Kimi-K3 serving entry of config/kimi_study.yaml on Trillium's B200 nodes: job.sh runs `milabench run` of
# that entry (the vLLM server on the entry's num_machines nodes, then its client passes against it). README.md.
#
#   scripts/trillium/kimi/serve.sh ENTRY NODES TIME [NAME=value ...]
#   NODELIST=trig[0147-0148] scripts/trillium/kimi/serve.sh kimi-vllm-2n-tp8pp2-bringup 2 1:00:00
#
# NAME=value pairs are exported to the job, e.g. MILABENCH_DIRS_DATA=<dir> (a truncated model, truncate_k3.py).
# JOB_PREFIX: the job name's prefix, nw-k3inf by default (K3-XN: nw-k3-xn).
# Paths: $B = MILABENCH_BASE (default $SCRATCH/trillium/new/k3-inf): src/ (this repository, its .venv the milabench CLI),
# venv/torch (vLLM), data/hub (the checkpoint), runs/<entry>.<job id> (run data), slurm/<job name>.<job id>.out.
set -euo pipefail
ENTRY=${1:?usage: serve.sh ENTRY NODES TIME [NAME=value ...]} NODES=${2:?} TIME=${3:?}
shift 3
B=${MILABENCH_BASE:-${SCRATCH:?}/trillium/new/k3-inf}
mkdir -p "$B/slurm"
for kv in "$@"; do export "${kv?}"; done
export MILABENCH_BASE=$B ENTRY
# the real sbatch: the site filter refuses above 25 nodes, the admins asked us to use this (2026-11-05).
# never: trig0125, trig0132 (2026-11-07), trig0150 (2026-11-08 15:10), one GPU without NVLink each; trig0114 (PSU),
# trig0138 (ECC), trig0146 (ECC, 2026-11-08 20:40)
/opt/slurm/bin/.sbatch --export=NONE --get-user-env --parsable \
    --job-name="${JOB_PREFIX:-nw-k3inf}-$ENTRY" --account=rrg-bengioy-ad --partition=compute_b200_full_node \
    --reservation=b200-friendly-users --qos=bump --gpus-per-node=b200:8 --ntasks-per-node=1 --exclusive --mem=0 \
    --nodes="$NODES" --time="$TIME" --signal=B:USR1@300 ${NODELIST:+--nodelist="$NODELIST"} \
    --exclude=trig0114,trig0125,trig0132,trig0138,trig0146,trig0150 \
    --output="$B/slurm/%x.%j.out" --export=ALL "$B/src/scripts/trillium/kimi/job.sh"
