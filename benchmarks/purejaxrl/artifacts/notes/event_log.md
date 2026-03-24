# DQN Throughput Optimization Event Log

**Benchmark:** PureJaxRL DQN, SpaceInvaders-MinAtar
**Run command:** `python main.py dqn --num_envs 128 --buffer_size 131072 --buffer_batch_size 65536 --env_name SpaceInvaders-MinAtar --training_interval 10`
**GPU:** Quadro RTX 8000 (Turing TU102, 48 GB VRAM)
**Driver/CUDA:** cuda:0, 60832 MiB pool

---

## [BASELINE] FP32 baseline measurement

**File:** `artifacts/benchmarks/baseline_run2.txt`
**Code commit:** pre-optimization (LOG_EVERY_N timing, no FP16)

Learning-phase throughput (steady-state, skipping first 10 samples):
- Mean: **90,339 items/s**
- Median: 90,635 items/s
- Stdev: 862 items/s
- Range: 88,342 – 91,033 items/s

Exploration-phase (no learning): ~420k items/s
Memory peak: ~638 MiB
Peak returns: ~4.7 (end of 100k timestep run)

---

## [PROFILE] JAX profiler trace - FP32 baseline

**Trace location:** `artifacts/profiles/jax_trace/`
**Profiling script:** `profile_dqn.py` (200k steps, 1 warmup run, then profiled run)

### Key findings from Perfetto trace (1 scan iteration = 1827ms total):

| Kernel / Op | Time (ms) | % of total | Notes |
|---|---|---|---|
| `volta_sgemm_128x64_nn` etc. (FP32 GEMMs) | ~728ms | 39.9% | Forward pass in `_learn_phase` |
| `loop_slice_fusion` (buffer gather) | ~301ms | 16.5% | `buffer.sample()` scatter-gather |
| `cond.108` (learning conditional) | ~1361ms | 74.5% | All of learning phase |

The main bottleneck is **FP32 GEMM in the learning phase** consuming 40% of total time, followed by buffer sample bandwidth (~17%).

Tensor core verification: `volta_sgemm_*` confirms FP32 path (no tensor cores active in baseline).

---

## [HYPOTHESIS] FP16 tensor cores for learning GEMMs

The Turing GPU has 576 tensor cores capable of FP16 matrix multiply, giving 2× tensor core throughput vs. FP32 CUDA cores. MinAtar SpaceInvaders observations are 4-channel binary (0/1) arrays — exactly representable in FP16. The network weights are small (Dense 400→120→84→6) and will not overflow FP16 range during inference.

**Expected benefit:** Replace `volta_sgemm` (FP32) with `turing_fp16_s1688gemm` (FP16 tensor core).
**Risk:** Gradient overflow in loss computation → mitigated by casting output back to FP32 before loss.
**BF16 explicitly rejected:** Turing does not have native BF16 tensor cores (requires Ampere+). Micro-benchmark confirmed BF16 0.84ms vs FP32 0.69ms for the relevant GEMM shape — BF16 is slower on this GPU.

---

## [CHANGE] Optimization 1: FP16 mixed precision in learning phase

**File:** `dqn.py`
**Location:** `_learn_phase` function (inside `_update_step`)

Replaced direct `network.apply(params, obs)` calls with:

```python
def _apply_fp16(params, obs):
    params_fp16 = jax.tree.map(lambda p: p.astype(jnp.float16), params)
    return network.apply(params_fp16, obs.astype(jnp.float16)).astype(jnp.float32)
```

Applied to both:
1. `q_next_target` computation (target network forward pass)
2. `_loss_fn` → `q_vals` computation (online network forward pass for loss)

Gradients computed on FP32 loss via `jax.value_and_grad(_loss_fn)(train_state.params)` — params stay in FP32.

---

## [EXPERIMENT] FP16 learn-only: run 1

**File:** `artifacts/benchmarks/fp16_learn_only_run1.txt`

Learning-phase throughput: ~158k items/s
Speedup over baseline: **+75%**
Reward: unchanged (same returns curve, same loss magnitude)

FP16 tensor core kernel now visible in profiler: `turing_fp16_s1688gemm_fp16_128x256_ldg8_f2f_stages_32x1_nn`

---

## [HYPOTHESIS] FP16 buffer storage to reduce gather bandwidth

`buffer.sample()` scatter-gather (`loop_slice_fusion`) consumes 301ms = 16.5% of baseline. MinAtar observations are 0/1 booleans, exactly representable in FP16. Storing observations as FP16 in the buffer halves the gather bandwidth for `buffer.sample()`.

The `_apply_fp16` helper in `_learn_phase` already accepts FP16 observations, so no additional casting is needed during training.

---

## [CHANGE] Optimization 2: FP16 buffer storage for observations

**File:** `dqn.py`
**Locations:** buffer initialization, `_update_step` buffer add

```python
# Buffer init:
_timestep = TimeStep(obs=_obs.astype(jnp.float16), action=_action, reward=_reward, done=_done)

# Buffer add (inside _update_step):
timestep = TimeStep(obs=last_obs.astype(jnp.float16), action=action, reward=reward, done=done)
```

---

## [EXPERIMENT] FP16 learn + FP16 buffer: run 2

**File:** `artifacts/benchmarks/opt2_fp16buf_run2.txt`

Learning-phase throughput (steady-state):
- Mean: **167,059 items/s**
- Median: 169,967 items/s
- Max: 173,480 items/s

Speedup over FP32 baseline: **+85%**
Memory peak: **324 MiB** (vs 638 MiB baseline — 49% reduction from FP16 buffer)
Reward: unchanged (peak returns ~4.7, same loss curve 0.08–0.15)

Profiler comparison (1 scan iteration):
- Baseline: 1827ms total
- Optimized: 1152ms total — **37% reduction**
- `loop_slice_fusion` dropped from 301ms → 149ms (buffer FP16 bandwidth halved)
- FP16 GEMM kernel: 117.3ms vs baseline FP32 GEMM 728ms (**6× faster GEMMs**)

---

## [EXPERIMENT] FP16 learn + FP16 buffer: final confirmation run

**File:** `artifacts/benchmarks/opt_final_run1.txt`

Learning-phase throughput (steady-state): ~170-172k items/s
Consistent with run 2.

---

## Summary

| Optimization | Throughput | Speedup | Memory |
|---|---|---|---|
| FP32 baseline | ~90k items/s | 1.0× | 638 MiB |
| FP16 learning only | ~158k items/s | 1.75× | 638 MiB |
| FP16 learning + FP16 buffer | ~170k items/s | 1.89× | 324 MiB |

Total improvement: **+89% throughput, −49% memory**
Reward quality: **preserved** (identical returns curve and loss values)
