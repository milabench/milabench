## Overview

## Entry point

`main.py::main()` -> `run()`. Parsed via `argparse`.

## Code flow

1. `run()` parses CLI args, calls `brax.training.agents.ppo.train()` directly.
2. Brax's internal PPO handles env creation, rollout collection, GAE, and gradient updates.
3. A `progress_fn` callback fires on each eval step, using `giving.give()` to emit `training/sps` and `eval/episode_reward`.
4. The `__main__` block wraps `main()` in a `given()` context that maps `training/sps` to `{"task": "train", "rate": sps, "units": "steps/s"}` and maps negative episode reward to `loss`.

## Voirfile

Standard `benchmate.monitor.voirfile_monitor`. Config: skip=5, stop=60. Catches `StopProgram` for early stop.

## Model / algorithm

No explicit model definition in this benchmark -- Brax's PPO trainer constructs its own actor-critic network internally. Key hyperparameters passed through:

| Param | Default | Notes |
|---|---|---|
| `num_envs` | 8192 | Parallel simulation instances |
| `batch_size` | 1024 | `auto_batch(1024)` in config |
| `num_minibatches` | 32 | Must divide `num_envs * batch_size` evenly |
| `episode_length` | 20 | Short episodes (config override from default 10) |
| `unroll_length` | 5 | Steps unrolled in `jax.lax.scan` |
| `num_timesteps` | 100M | Total training steps |
| `num_evals` | 500 | Eval frequency |
| `normalize_observations` | True | Hardcoded |
| `action_repeat` | 1 | Hardcoded |

## Environment setup

- `XLA_PYTHON_CLIENT_PREALLOCATE=False` set in both `main.py` and `benchfile.py::make_env()`.
- `torch` is imported before `brax` as a trick to make JAX locate CUDA shared libraries.

## Data pipeline

None. The Brax physics simulator generates all data on-the-fly in JAX. `prepare.py` is a no-op placeholder.

## Gotchas

- The `num_envs`, `batch_size`, and `num_minibatches` are tightly coupled. Changing one without adjusting the others will crash Brax's internal assertions.
- The commented-out line `args.num_envs = (args.batch_size * args.num_minibatches)` shows the intended constraint but is disabled -- the config hardcodes all three values.
- Loss is reported as `-episode_reward.item()`, so lower loss = better reward.
- `progress_fn` reports metrics on eval boundaries (every `num_timesteps / num_evals` steps), not every training step.

## Bottlenecks

- The benchmark is JAX XLA compile-bound on first iteration, then GPU compute-bound during simulation + PPO updates.
- Memory usage scales with `num_envs` since all environment states are held on GPU.
