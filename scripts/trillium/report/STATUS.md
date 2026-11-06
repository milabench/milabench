# Report status

One row per block of `~/projects/tri-work/ops/ANALYSIS_BRIEF.md`. What ran against what's planned: `cd ../analysis && $PY analyze.py progress`.

| Block | State | Runs used | Checked against | Missing |
|---|---|---|---|---|
| Machine | partial | health 1238238-40, 1238571, 1245968, 1246233, 1246278; hw-probe, post-pstate; hw/post-pstate tables | clocks = `healthcheck_cpu_clocks.tsv` on 43 rows; step means = `before_after.tsv`; +8.5% from raw steps; 26.3 s from rank 0 of 1239932 | admin answers, `trig0122`, `trig0146` |
| Interference | partial | E1 12 runs; R 8 runs; 1241064, 1241200, 1240558 | agent's +0.40..-0.51% reproduced from rank 0 metrics with its window (batch 11 on); report uses measured steps only, +0.17%; +0.37% beside GLM recomputed (2.993 / 2.982 s) | more `tri-ref` |
| Health and network | done for now | as Machine; nccl 1239634, 1239766, 1239768, 1239660; 1239664, 1239767 logs | DESIGN numbers: 62.9-66.9, 45.0-47.5, 174-192, 171-188 GB/s reproduce | nothing above 25 nodes is planned; D4 not run |
| Qwen layouts | partial | A-Q at 13 nodes, weak-scaling runs at 2 and 26 | step means = `qwen-aq/results.tsv` on 42 runs (0.0005 s); tokens differ for padded runs, on purpose (see section) | `fsdp` 24k BF16 reduction at 39 |
| Precision loss | partial | Qwen C-loss 9 runs, GLM-5 2 | gaps from raw loss events, separately: +0.312, +0.275, -0.038, -0.075 = journal and `gap3.py` | GLM-5 same and other seed, tensorwise, blockwise |
| Precision speed | partial | C-speed 1, 25, 26, 39 nodes; GLM-5 13; p3-acfull; `fp8-glm/summary.txt` | `results.tsv`, `p3-results.tsv` (step means); FP8 FLOP shares recomputed from summary (97%, 76%) | GLM-5 39 |
| Checkpoints | partial | F1 Qwen 1, 13, 25; GLM-5 13, weights in 3 rounds (1243819, 1245503, 1246920) | `p3-results.tsv`, journal: all sizes and times match; 1246920 sync save 26.9 s from its recovered summary | GLM-5 weights and full 39; Qwen async and pinned 39 |
| Qwen scaling | partial | B-Q-weak 1-39; B-Q-strong 13, 26 | steps = `qwen-large/results.tsv` (39 nodes: 5.160, 5.075 s both); 94.4% recomputed | `fsdp` strong at 39, labelled 43, 50 if caps clear |
| GLM-5 layouts | partial | A-G 10, 12, 13 nodes | steps within 0.1 s of `p3-results.tsv` (their mean starts one step later); garbled summaries decoded by `runs.garbled_summary`; for 1240808 identical to the agent's recovered file | 20, 25, 26, 30, 36, 40 |
| GLM-5 scaling | partial | 13 and 26 nodes, weak and strong (1247135) | agent's 49.59 s, 84,584 tok/s, 69.97 GiB; 0.980, 0.892 by means against mine 0.978, 0.895 by medians; bubble formula against three batch pairs | 39 |
| Startup, energy | partial | all completed runs | telemetry: 1241805 7.40 kW, max 8.07, 19% low = journal; 1241694 4.41 kW against 4.47 (different window, not chased) | grows with each run |
| Stability, stragglers | partial | stragglers from largest FSDP run | | F2 not run |
| Introduction, conclusions | partial | | | final answers, after 39 nodes |
