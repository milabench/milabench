# CleanRL JAX

PPO benchmark for Atari games using [CleanRL](https://github.com/vwxyzjn/cleanrl)'s JAX implementation with [envpool](https://github.com/sail-sg/envpool) for high-throughput environment vectorization. Trains a CNN-based actor-critic on **Breakout-v5** by default.

## What it measures

- JAX PPO throughput on image-based (Atari) observations
- CNN forward/backward pass speed on GPU
- envpool XLA-native environment stepping performance

## Key characteristics

- **Framework**: JAX + Flax + envpool (XLA integration)
- **Algorithm**: PPO with GAE, clipped surrogate loss
- **Network**: 3-layer CNN (32/64/64 filters) + 512-unit FC, separate actor and critic heads
- **Environment**: `Breakout-v5` (Atari via envpool), episodic life, reward clipping
- **Observations**: Raw pixel frames (4-stacked, 84x84), normalized to [0,1]
- **Action space**: Discrete (Gumbel-softmax trick for sampling)
- **Optimizer**: Adam with LR annealing, global gradient norm clipping (0.5)
- **Plan**: per_gpu (monogpu)
- **Tags**: `monogpu`, `jax`

## Constraints

Batch size is derived at runtime: `num_envs * num_steps`. Changing `num_envs`, `num_steps`, or `num_minibatches` requires keeping `batch_size % num_minibatches == 0`.

## Dependencies

`jax`, `flax`, `optax`, `envpool`, `gym`, `tyro`, `torch` (tensorboard), `numpy`, and ~50 transitive deps (see `requirements.in`)
