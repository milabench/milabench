#!/bin/bash
# Step-time gap against Florida (FLUX_BRIEF.md, last section): Florida's 2-node config on real data, STEPS steps per run
# (default 200), one run per variant, one after another in one 2-node job (IMPL_BRIEF.md, "No job churn"). Each run is
# flux.sub with its own LOGDIR, dev/runs/gap.<job>/<variant>; the image through dev/flux.sif (its own tmpmounts). Once
# per job: the GPUs' power and clock limits, and each node's mean CPU clock every 5 s (cpu-<node>.csv).
#   SRC=$F/src-dev SCRIPT=gap.sbatch $F/src-dev/scripts/trillium/flux/submit.sh gap2 2 0:55:00 -- --nodelist=...
#SBATCH --exclusive
#SBATCH --mem=0

set -uxo pipefail  # not -e: one failed variant does not stop the next
ulimit -c 0  # no core dumps (2026-11-07); flux.sub's srun steps get --propagate=CORE
for kv in "$@"; do export "$kv"; done
F=${SCRATCH}/trillium/new/flux
B=$F/dev/runs/gap.${SLURM_JOB_ID}
mkdir -p "$B"
srun --propagate=CORE --ntasks-per-node=1 nvidia-smi --query-gpu=index,power.limit,enforced.power.limit,power.default_limit,power.max_limit,clocks.max.sm,clocks.applications.graphics --format=csv > "$B/gpu-limits.csv"
srun --propagate=CORE --overlap --ntasks-per-node=1 bash -c 'while :; do echo "$(date +%s) $(awk "/MHz/ {s += \$4; n++} END {printf \"%.0f\", s / n}" /proc/cpuinfo)"; sleep 5; done > "$0/cpu-$(hostname -s).csv"' "$B" &

run() {
    local name=$1
    shift
    mkdir -p "$B/$name"
    echo "GAP_RUN_START $name $(date +%s)"
    bash "$SRC/scripts/trillium/flux/flux.sub" SRC="$SRC" CONT="$F/dev/flux.sif" LOGDIR="$B/$name" MAX_STEPS="${STEPS:-200}" "$@"
    echo "GAP_RUN_STOP $name $(date +%s)"
}
run base NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,TUNING,COLL  # NCCL's algorithm and protocol choices, message sizes
run proto-simple NCCL_PROTO=Simple  # the reduce-scatters and all-gathers ran RING_LL in job 1277931
run pxn-off NCCL_PXN_DISABLE=1  # GPUs without a NIC send through their own PCIe path instead of a peer's NIC
run cpu-default CPU_SHARE=0  # Florida's srun as it is: Slurm's default CPUs per task
run base2  # the baseline again: run-to-run spread
