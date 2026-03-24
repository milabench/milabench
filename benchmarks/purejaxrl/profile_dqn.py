"""
Profiling script for DQN benchmark using JAX profiler.
Runs a short version and captures GPU timeline traces.
"""
import os
import time
import jax
import jax.numpy as jnp

# Import the DQN code
import sys
sys.path.insert(0, os.path.dirname(__file__))

import torch  # for torch libs
from dqn import make_train, Arguments

PROFILE_DIR = "/tmp/artifacts/profiles/jax_trace"
os.makedirs(PROFILE_DIR, exist_ok=True)

# Use same params as benchmark but shorter run so trace is manageable
args = Arguments(
    num_envs=128,
    buffer_size=131072,
    buffer_batch_size=65536,
    env_name="SpaceInvaders-MinAtar",
    training_interval=10,
    total_timesteps=200_000,   # longer run so we capture both phases
    seed=0,
    num_seeds=1,
    log_every_n=10,
)

config = {
    "NUM_ENVS": args.num_envs,
    "BUFFER_SIZE": args.buffer_size,
    "BUFFER_BATCH_SIZE": args.buffer_batch_size,
    "TOTAL_TIMESTEPS": args.total_timesteps,
    "EPSILON_START": args.epsilon_start,
    "EPSILON_FINISH": args.epsilon_finish,
    "EPSILON_ANNEAL_TIME": args.epsilon_anneal_time,
    "TARGET_UPDATE_INTERVAL": args.target_update_interval,
    "LR": args.lr,
    "LEARNING_STARTS": args.learning_starts,
    "TRAINING_INTERVAL": args.training_interval,
    "LR_LINEAR_DECAY": args.lr_linear_decay,
    "GAMMA": args.gamma,
    "TAU": args.tau,
    "ENV_NAME": args.env_name,
    "SEED": args.seed,
    "NUM_SEEDS": args.num_seeds,
    "PROJECT": args.project,
    "LOG_EVERY_N": args.log_every_n,
}

rng = jax.random.PRNGKey(config["SEED"])
rngs = jax.random.split(rng, config["NUM_SEEDS"])

print("Compiling...", flush=True)
t0 = time.perf_counter()
train_vjit = jax.jit(jax.vmap(make_train(config), in_axes=(0,)))
compiled_fn = train_vjit.lower(rngs).compile()
t1 = time.perf_counter()
print(f"Compilation time: {t1-t0:.2f}s", flush=True)

# Warmup run (not profiled)
print("Warmup run...", flush=True)
_ = jax.block_until_ready(compiled_fn(rngs))
print("Warmup done.", flush=True)

# --- JAX profiler trace ---
print(f"Starting JAX profiler trace -> {PROFILE_DIR}", flush=True)
with jax.profiler.trace(PROFILE_DIR, create_perfetto_trace=True):
    t_start = time.perf_counter()
    outs = jax.block_until_ready(compiled_fn(rngs))
    t_end = time.perf_counter()

total_steps = config["TOTAL_TIMESTEPS"]
elapsed = t_end - t_start
print(f"Profiled run: {total_steps} steps in {elapsed:.3f}s = {total_steps/elapsed:.0f} steps/s", flush=True)
print(f"Trace written to: {PROFILE_DIR}", flush=True)
print("Open in Perfetto: https://ui.perfetto.dev/  (upload the .perfetto-trace file)", flush=True)
