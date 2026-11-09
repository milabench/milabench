#!/bin/bash
# NSYSCMD for NVIDIA's run_and_time.sh (its profiler hook, run inside the image): rank NSYS_RANK (default 0) runs under
# Nsight Systems for NSYS_DURATION s (default 20) from NSYS_DELAY s after it starts (default 360: past init and the first
# steps), with the kernels inside CUDA graphs traced; the other ranks run as they are. --kill=none: when the window ends
# nsys would otherwise send SIGTERM to the run. Copy it into the job's LOGDIR (bound at /results), set
# NSYSCMD=/results/nsys_rank.sh; the report lands in LOGDIR.
if [ "${SLURM_PROCID}" = "${NSYS_RANK:-0}" ]; then
    exec nsys profile -t cuda,nvtx -s none --cpuctxsw=none --cuda-graph-trace=node --kill=none \
        --delay="${NSYS_DELAY:-360}" --duration="${NSYS_DURATION:-20}" -o "/results/nsys-rank${SLURM_PROCID}" \
        --force-overwrite=true "$@"
fi
exec "$@"
