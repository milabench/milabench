#!/bin/bash
# prof.py on the first trace of each RUN directory (runs of job.sbatch with variant prof), into RUN/prof.txt, on one
# node (a trace of 1.4 GB takes about 10 GB of memory to read). Submit as job.sbatch, --nodes=1 --time=00:20:00:
#   /opt/slurm/bin/.sbatch ... --job-name=nw-of3-prof prof.sbatch RUN...
ulimit -c 0
O=$SCRATCH/trillium/new/of3
for run in "$@"; do
    trace=$(ls "$run"/prof/*.pt.trace.json | head -1)
    echo "[of3] $run: $trace"
    srun --export=ALL --propagate=CORE --ntasks=1 $O/venv/bin/python $O/src/scripts/trillium/of3/prof.py "$trace" > "$run/prof.txt" 2>&1
    cat "$run/prof.txt"
done
