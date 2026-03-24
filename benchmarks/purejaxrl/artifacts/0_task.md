# PureJaxRL DQN Optimization Task

## Setup and Initialization
- [x] Create branch `agent_A_throughput_opt`
- [x] Initialize repository artifacts directory structure
- [x] Verify SSH and GPU execution (`ssh delaunay@rg31703`)
- [x] Create and populate [artifacts/notes/event_log.md](file:///home/delaunap/milabench/benchmarks/purejaxrl/artifacts/notes/event_log.md)

## Baseline Benchmarking
- [ ] Run the baseline command on GPU
- [ ] Document baseline throughput and best_mean_reward tolerance
- [ ] Save output to `artifacts/benchmarks/baseline.txt`
- [ ] Record results to `artifacts/benchmarks/results.csv`

## Profiling
- [ ] Run profiler (e.g. `nsys` or `jax.profiler`)
- [ ] Save traces and profiler commands
- [ ] Identify bottlenecks

## Optimization
- [ ] Experiment with JAX/RL optimizations
- [ ] Keep track of changes in [event_log.md](file:///home/delaunap/milabench/benchmarks/purejaxrl/artifacts/notes/event_log.md)
- [ ] Verify correctness/reward checks

## Final Reporting
- [ ] Fill out `artifacts/FINAL_SUMMARY.md`
- [ ] Push branch
