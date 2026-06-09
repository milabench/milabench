## Overview

## Entry point

`main.py`, run as `__main__`. Args parsed by `tyro.cli(Args)`. No subcommands.

## Code flow

1. Parse args, compute derived values: `batch_size = num_envs * num_steps`, `minibatch_size = batch_size // num_minibatches`, `num_iterations = total_timesteps // batch_size`.
2. Create envpool environment via `envs.xla()` which returns `(handle, recv, send, step_env)` -- native XLA stepping, no Python overhead per step.
3. Initialize three separate Flax modules: `Network` (CNN backbone), `Actor` (policy head), `Critic` (value head). Params bundled into `AgentParams` struct.
4. **Outer Python loop** over `num_iterations` (unlike purejaxrl, this is NOT a single `jax.lax.scan`):
   - `rollout()`: Collects `num_steps` transitions via `jax.lax.scan` over `step_once`.
   - `compute_gae()`: GAE via reverse `jax.lax.scan`.
   - `update_ppo()`: Nested scan -- outer over `update_epochs`, inner over minibatches.
5. Logs to TensorBoard `SummaryWriter`. Optional W&B tracking.

## Network architecture

- **Network** (backbone): 3 conv layers (32@8x8/4, 64@4x4/2, 64@3x3/1) + flatten + Dense(512). Input transposed from NCHW to NHWC and divided by 255.
- **Actor**: Single Dense layer -> logits (num_actions). Action sampled via Gumbel-softmax trick.
- **Critic**: Single Dense layer -> scalar value.
- All layers use `orthogonal` initialization.

## Loss function

Standard PPO: `pg_loss - ent_coef * entropy + vf_coef * value_loss`.
- Policy: Clipped surrogate (`clip_coef=0.1`).
- Value: 0.5 * MSE (no value clipping despite the arg existing).
- Entropy: Computed manually via `-sum(p * log(p))` with log-sum-exp normalization.
- Gradients clipped by global norm (0.5).

## Environment integration

envpool provides XLA-native environment stepping. The `step_env_wrappeed` function (note: typo in source) wraps the raw XLA step to track episode statistics. Environment uses episodic life mode and reward clipping.

## Voirfile

Standard `benchmate.monitor.voirfile_monitor`. Config: skip=5, stop=20. No `StopProgram` handling.

## Config (training.yaml)

```
batch_size     = num_envs * num_steps
minibatch_size = batch_size // num_minibatches
num_iterations = total_timesteps // batch_size
```

Configurable via `--total_timesteps`, `--num_steps`, `--num_minibatches`.

## Gotchas

- `XLA_PYTHON_CLIENT_MEM_FRACTION=0.6` hardcoded at module level (limits JAX memory to 60% of GPU).
- `TF_XLA_FLAGS` set for deterministic reductions and autotune level 2.
- There is a typo: `TF_CUDNN DETERMINISTIC` (space instead of underscore) -- this env var is silently ignored.
- The outer training loop is a Python `for` loop, not a `jax.lax.scan`. This means Python overhead per iteration, but allows TensorBoard logging between iterations.
- `envpool` XLA integration requires the environment to be created before any JAX compilation. The `handle` object is threaded through JIT-compiled functions.
- `requirements.in` is very large (~100 packages) because CleanRL has many transitive dependencies.
- `prepare.py` is a no-op placeholder.

## Bottlenecks

- CNN forward pass on 84x84 Atari frames is the compute-dominant cost.
- envpool stepping is fast (C++ backend) but the `handle` threading through `jax.lax.scan` adds XLA graph complexity.
- LR annealing uses `optax.inject_hyperparams`, which makes the optimizer state larger than a standard Adam.
