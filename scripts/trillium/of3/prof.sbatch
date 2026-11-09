#!/bin/bash
# prof.py on rank 0's trace of each RUN directory (runs of job.sbatch with variant prof), into RUN/prof.txt, on one
# node (a 1.4 GB trace takes about 10 GB of memory to read). Rank 0's process is found in the run's NCCL logs. Submit as
# job.sbatch, with --nodes=1 --time=00:30:00 and a job name nw-of3-prof:
#   /opt/slurm/bin/.sbatch ... prof.sbatch RUN...
ulimit -c 0
O=$SCRATCH/trillium/new/of3
for run in "$@"; do
    log=$(grep -l " rank 0 nRanks" "$run"/nccl/*.log | head -1)
    host_pid=$(basename "$log" .log)
    trace=$(ls "$run"/prof/${host_pid/./_}.*.pt.trace.json)
    echo "[of3] $run: $trace"
    srun --export=ALL --propagate=CORE --ntasks=1 $O/venv/bin/python $O/src/scripts/trillium/of3/prof.py "$trace" > "$run/prof.txt" 2>&1
    cat "$run/prof.txt"
done
