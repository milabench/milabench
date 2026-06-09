# PureJaxRL

End-to-end JAX RL benchmark with two sub-benchmarks: **PPO** (continuous control via Brax/Gymnax) and **DQN** (discrete control via Gymnax/MinAtar). Based on the [PureJaxRL](https://github.com/luchris429/purejaxrl) project -- the entire training loop (env stepping, replay buffer, gradient updates) runs inside `jax.jit` / `jax.lax.scan`.

## What it measures

- JAX end-to-end RL throughput: compile + vectorized env simulation + network training
- PPO: continuous-action actor-critic on `hopper` (Brax)
- DQN: discrete-action Q-learning on `SpaceInvaders-MinAtar` (Gymnax)

## Key characteristics

| | PPO (`ppo`) | DQN (`dqn`) |
|---|---|---|
| **Algorithm** | PPO with GAE, clipped surrogate | DQN with epsilon-greedy, target network |
| **Network** | Actor-Critic MLP (2x256) | Q-Network MLP (120, 84) |
| **Environment** | `hopper` (Brax via BraxGymnaxWrapper) | `SpaceInvaders-MinAtar` (Gymnax) |
| **Env count** | `auto(cpu_per_gpu, 128)` | `auto(cpu_per_gpu, 128)` |
| **Replay buffer** | None (on-policy) | Flashbax flat buffer (131072) |
| **Total timesteps** | 2M | 2M |
| **Plan** | per_gpu | per_gpu |

## Tags

`monogpu`, `gym`, `rl`, `jax`

## Dependencies

`jax`, `flax`, `optax`, `gymnax`, `brax`, `flashbax`, `distrax`, `navix`, `tfp-nightly`, `argklass`, `torch` (CUDA shim)
