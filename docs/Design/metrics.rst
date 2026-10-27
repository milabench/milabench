

Level 1: wandb experiment metrics - loss, step time, etc

Purpose: nice graphs
Rank: global rank 0 - through httpproxy to wandb
Instrument: the model code

Level 2: per-rank training metrics - dataloader latency, fwd/bwd pass, etc

Purpose: report generation
Rank: on all ranks - to tsv on /tmp, or $SCRATCH
Instrument: the model code

Level 3: system logging - gpu clock, cpu clock, ib tx/rx, temp, etc

Purpose: straggler monitoring - we can tail all node health
Rank: on local rank 0 - to tsvs on $SCRATCH
Instrument: nothing, starts as a process with milabench