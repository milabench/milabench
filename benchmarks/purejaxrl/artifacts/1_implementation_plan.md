# PureJaxRL DQN Throughput Optimization Plan

## Goal Description
The objective is to increase the throughput (`steps/sec`) of the `SpaceInvaders-MinAtar` DQN benchmark in [benchmarks/purejaxrl/dqn.py](file:///home/delaunap/milabench/benchmarks/purejaxrl/dqn.py) while preserving the best mean reward within a -1% tolerance.

## Proposed Changes
Currently we are in the baseline assessment and profiling phase. 

### [benchmarks/purejaxrl/dqn.py](file:///home/delaunap/milabench/benchmarks/purejaxrl/dqn.py)
We will establish the baseline throughput using the current source code configuration:
- `num_envs`: 10 (default)
- `total_timesteps`: 100,000

After baseline metrics and `nsys`/`jax.profiler` traces are collected, we anticipate making throughput-improving changes to:
- Overlapping host/device transfers (if any)
- Improving `jax.lax.scan` operations or `jax.lax.cond` branch prediction
- Optimizing epsilon-greedy exploration logic (e.g. batching random number generation)
- Reducing the frequency or overhead of `jax.debug.callback` metric logging

## Verification Plan

### Automated/Benchmarking Tests
1. **Quick Correctness Check**: Run the minimal benchmark over 100k timesteps to ensure no NaNs or crashes.
2. **Performance Measurement**: 
   - Warm-up run discard
   - N=3 measured runs to compute median `steps/sec`
   - Capture `best_mean_reward` and ensure it sits within the standard deviation or -1% of the baseline reward.

### Profiling
- Run `nsys profile` (NVIDIA Nsight Systems) before and after changes to verify the structural elimination of hardware and JAX bottlenecks.
