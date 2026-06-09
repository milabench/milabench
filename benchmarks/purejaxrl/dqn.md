## Overview

## Config

- **Algorithm**: DQN (Deep Q-Network)
- **Environment**: MinAtar SpaceInvaders (discrete actions)

## Model

`QNetwork`: 3-layer MLP (120, 84, action_dim), ReLU activations.

## Replay buffer

Flashbax flat buffer: `buffer_size=131072`, `buffer_batch_size=65536`.
Transitions stored as `TimeStep(obs, action, reward, done)`.
Sampling is memory-bandwidth bound with large `buffer_batch_size`.

## Exploration

Epsilon-greedy with linear annealing from 1.0 to 0.05.

## Target network

Soft update via `optax.incremental_update` at configurable interval (tau=1.0 = hard copy).

## Training gating

Learning only starts after `LEARNING_STARTS` timesteps and fires every
`TRAINING_INTERVAL` steps, controlled by `jax.lax.cond`.

## Quirks

- Uses `jax.vmap` over multiple seeds (`NUM_SEEDS`) for seed parallelism.
- Entire training loop is one fused `jax.lax.scan` -- no Python-level loop.
- XLA compilation is the dominant startup cost (can take minutes for the long scan).
- `nvidia-smi` reports constant ~62GB regardless of buffer/env count because JAX
  pre-reserves the pool. Actual usage is ~80MB peak per JAX's own accounting.
- Metrics callback fires every 1000 timesteps.
- `XLA_PYTHON_CLIENT_PREALLOCATE=False` set at module level and in `benchfile.py`.
