# Event Log — DQN Throughput Optimization

## Metadata

- Date: 2026-03-24
- Agent ID: A
- Human operator: user
- Repo: /tmp/milabench
- Starting commit hash: (no git repo)
- Branch name: N/A (working directly in /tmp/milabench)
- Hardware: Quadro RTX 8000 (46080 MiB VRAM), Driver 580.95.05
- Software: JAX 0.9.2 / jaxlib 0.9.2 / Python 3.12.11
- Baseline command:
  ```
  /tmp/output/venv/torch/bin/python /tmp/milabench/benchmarks/purejaxrl/main.py \
    dqn --num_envs 128 --buffer_size 131072 --buffer_batch_size 65536 \
    --env_name SpaceInvaders-MinAtar --training_interval 10
  ```
- Benchmark window: TOTAL_TIMESTEPS=100_000 (default), NUM_UPDATES = 100_000 // 128 = 781 scan steps
- Throughput metric: environment steps/sec (items/s reported by benchmate StepTimer)
- Reward metric: best_mean_reward (returned_episode_returns, mean over envs)
- Reward tolerance: Option C — no meaningful drop over the training window (no algorithmic changes made)

---

## Events

---

```
T+0  [BASELINE]
Action/Change: Ran stock dqn.py before any changes.
Hypothesis/Reason: Establish baseline throughput.
Result: See artifacts/benchmarks/baseline.txt
Evidence: artifacts/benchmarks/baseline.txt
Next: Profile and identify bottlenecks.
```

---

```
T+1  [DISCOVERY]
Action/Change: Static code analysis of dqn.py.
Hypothesis/Reason: Understanding callback and scan structure.
Result: Three bugs / inefficiencies found:
  1. BUG — Callback condition: `if (metrics["timesteps"] + 1) % 1000:` evaluates
     to True for ~99.9% of steps because NUM_ENVS=128 steps never land on
     multiples of 1000 (128k % 1000 is never 0). Callback fires every scan step.
  2. BUG — `step_timer.timestep = metrics["timesteps"]` (singular) never updates
     `step_timer.timesteps` (plural), so delta always equals total timesteps.
  3. OVERHEAD — `jax.debug.callback` dispatched 781 times per run, each forcing
     a host-device sync that blocks GPU pipelining.
Evidence: Code review of dqn.py lines 233-246.
Next: Fix via chunked scan + corrected callback logic.
```

---

```
T+2  [CHANGE]
Action/Change: Implement chunked scan optimization (SCAN_CHUNK_SIZE parameter).
  - Split _update_step into _update_step_inner (no callback) and outer _update_chunk
    that runs SCAN_CHUNK_SIZE inner steps then fires one debug.callback.
  - Fix callback condition: removed buggy modulo check, always log at chunk boundary.
  - Fix step_timer.timestep → step_timer.timesteps typo.
  - Remove redundant block_until_ready() inside debug.callback.
  - Add --scan_chunk_size CLI argument (default=10).
Hypothesis/Reason: Reducing jax.debug.callback dispatches from 781 to 78 (10x)
  eliminates host-device sync overhead. XLA can also fuse more operations within
  each 10-step chunk.
Result: TBD — see EXPERIMENT entries below.
Evidence: Modified dqn.py
Next: Run optimized benchmark.
```

---

```
T+3  [EXPERIMENT]
Action/Change: Run optimized dqn with scan_chunk_size=10.
Hypothesis/Reason: 10x fewer host syncs → measurable throughput gain.
Result: See artifacts/benchmarks/results.csv
Evidence: artifacts/benchmarks/opt_chunk10.txt
Next: Tune chunk size, validate reward.
```

---

```
T+4  [EXPERIMENT]
Action/Change: Run optimized dqn with scan_chunk_size=50.
Hypothesis/Reason: 50x fewer host syncs → larger gain, still enough benchmate observations.
Result: See artifacts/benchmarks/results.csv
Evidence: artifacts/benchmarks/opt_chunk50.txt
Next: Pick best, finalize.
```