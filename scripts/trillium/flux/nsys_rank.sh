#!/bin/bash
# NSYSCMD for NVIDIA's run_and_time.sh (its profiler hook, run inside the image): rank NSYS_RANK (default 0) runs under
# Nsight Systems from NSYS_DELAY s after it starts (default 360: past init and the first steps) until it exits, with the
# kernels inside CUDA graphs traced; the other ranks run as they are. Bound the trace with MAX_STEPS, not a duration:
# ending the collection mid-run ended rank 0 and hung the others (job 1277931). Copy it into the job's LOGDIR (bound at
# /results), set NSYSCMD=/results/nsys_rank.sh; the report lands in LOGDIR.
if [ "${SLURM_PROCID}" = "${NSYS_RANK:-0}" ]; then
    exec nsys profile -t cuda,nvtx -s none --cpuctxsw=none --cuda-graph-trace=node \
        --delay="${NSYS_DELAY:-360}" -o "/results/nsys-rank${SLURM_PROCID}" \
        --force-overwrite=true "$@"
fi
exec "$@"
