# Report status

One row per block of `~/projects/tri-work/ops/ANALYSIS_BRIEF.md`. What ran against what's planned: `cd ../analysis && $PY analyze.py progress`.

| Block | State | Runs used | Checked against | Missing |
|---|---|---|---|---|
| Machine | partial | health 1238238-40, 1238571, 1245968, 1246233, 1246278; hw-probe, post-pstate; hw/post-pstate tables | clocks = `healthcheck_cpu_clocks.tsv` on 43 rows; step means = `before_after.tsv`; +8.5% from raw steps; 26.3 s from rank 0 of 1239932 | admin answers, `trig0122`, `trig0146` |
| Interference | partial | E1 12 runs; R 8 runs; 1241064, 1241200 | agent's +0.40..-0.51% reproduced from rank 0 metrics with its window (batch 11 on); report uses measured steps only, +0.17% | more `tri-ref`; "+0.37% beside GLM" not rechecked |
| Health and network | done for now | as Machine; nccl 1239634, 1239766, 1239768, 1239660 | DESIGN numbers: 62.9-66.9, 45.0-47.5, 174-192, 171-188 GB/s reproduce | nothing above 25 nodes is planned |
| Qwen layouts | partial | A-Q at 13 nodes, weak-scaling runs at 2 and 26 | step means = `qwen-aq/results.tsv`; tokens differ for padded runs, on purpose (see section) | `fsdp` 24k BF16 reduction at 39; figure's top panel |
| Precision loss | not started | | | |
| Precision speed | not started | | | |
| Checkpoints | not started | | | |
| Qwen scaling | not started | | | |
| GLM-5 layouts | not started | | | |
| GLM-5 scaling | not started | | | |
| Startup, energy | not started | | | |
| Stability, stragglers | not started | | | F2 not run |
| Introduction, conclusions | not started | | | written for 52 nodes; update last |
