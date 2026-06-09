# Brax

Reinforcement learning benchmark using [Brax](https://github.com/google/brax), Google's JAX-based rigid-body physics simulator. Runs PPO training on the **Ant** locomotion environment via Brax's built-in PPO trainer.

## What it measures

- JAX XLA compilation and execution throughput on GPU
- Vectorized physics simulation speed (8192 parallel environments by default)
- PPO policy optimization rate (steps/second)

## Key characteristics

- **Framework**: JAX + Brax (not PyTorch -- `torch` is imported only to make JAX find CUDA libs)
- **Algorithm**: PPO via `brax.training.agents.ppo.train`
- **Environment**: `ant` (configurable), 8192 parallel envs
- **Batch size**: 1024 (auto-scaled), 32 minibatches
- **Metric**: `training/sps` (steps per second); loss is negative episode reward
- **Plan**: `njobs: 1` (single job, multi-GPU)
- **Tags**: `rl`, `jax`, `multigpu`, `gym`, `nobatch`

## Constraints

Brax requires very specific sizes to work. The milabench auto-resizer cannot resize this benchmark -- batch size, num_envs, and minibatch count are tightly coupled and must satisfy Brax's internal constraints.

## Dependencies

`jax`, `brax`, `torch` (CUDA lib shim), `voir`
