## Overview

## Config

- **Algorithm**: PPO (Proximal Policy Optimization)
- **Environment**: Brax hopper (continuous control), wrapped via `BraxGymnaxWrapper`
- **Precision**: Configurable dtype (fp32/fp16/bf16) threaded through network layers

## Model

`ActorCritic` (Flax `nn.Module`): two hidden layers of 256 units, tanh activation.
Actor outputs continuous action mean + learned log-std; critic outputs scalar value.

Action distribution: `distrax.MultivariateNormalDiag`.

## Training

- **Loss**: Clipped surrogate (actor) + clipped value loss + entropy bonus
  (`VF_COEF=0.5`, `ENT_COEF=0.0`)
- **Optimizer**: Adam with optional LR annealing, global gradient norm clipping (0.5)
- **Loop**: Single `jax.lax.scan` over `NUM_UPDATES` steps (no Python-level loop)

## Environment wrapping chain

`BraxGymnaxWrapper` -> `LogWrapper` -> `ClipAction` -> `VecEnv` ->
`NormalizeVecObservation` -> `NormalizeVecReward`

`NormalizeVecObservation/Reward` use online running-mean normalization (Welford's).

## Quirks

- `num_envs` defaults to 2048 in code but overridden to `auto(cpu_per_gpu, 128)` by config.
- GAE computation uses `jax.lax.scan` with `unroll=16`, trading memory for speed.
- `distrax` + `tfp-nightly` are fragile -- versions must match the installed JAX.
- Metrics callback fires every 10 update steps via `jax.lax.cond` + `jax.debug.callback`.
- `XLA_PYTHON_CLIENT_PREALLOCATE=False` set at module level and in `benchfile.py`.
