# Final Optimization Summary: PureJaxRL DQN

## Benchmark Configuration

| Parameter | Value |
|---|---|
| Environment | SpaceInvaders-MinAtar |
| `--num_envs` | 128 |
| `--buffer_size` | 131072 |
| `--buffer_batch_size` | 65536 |
| `--training_interval` | 10 |
| `--total_timesteps` | 100000 (default) |
| GPU | Quadro RTX 8000 (Turing, 48 GB) |

---

## Results

| Version | Mean throughput (items/s) | Memory (MiB) | Speedup |
|---|---|---|---|
| FP32 baseline | 90,339 ± 862 | 639 | 1.00× |
| FP16 learn only | ~158,000 | 639 | 1.75× |
| **FP16 learn + FP16 buffer** | **162,792–167,059 ± ~7k** | **324** | **1.80–1.85×** |

Throughput measured as env steps/second during the **learning phase** (training_interval=10; every 10th step triggers one gradient update on a batch of 65536). Exploration-only steps are ~420k items/s for all versions.

Reward quality is **preserved**: identical loss curve (0.08–0.15 MSE) and episode returns trajectory across all versions.

---

## Profiling Evidence

JAX profiler trace captured with `profile_dqn.py` (200k steps, Perfetto format).

### Baseline (1 scan step = 1827ms):
| Bottleneck | Time | % |
|---|---|---|
| FP32 GEMMs (`volta_sgemm_*`) | 728ms | 39.9% |
| Buffer gather (`loop_slice_fusion`) | 301ms | 16.5% |
| Other (`target update`, `cond`, etc.) | 798ms | 43.7% |

### Optimized (1 scan step = 1152ms):
| Kernel | Time | % |
|---|---|---|
| FP16 GEMMs (`turing_fp16_s1688gemm`) | 117ms | 10.2% |
| Buffer gather (`loop_slice_fusion`) | 149ms | 12.9% |
| Other | 886ms | 76.9% |

**Total scan time reduced by 37%** (1827ms → 1152ms).

---

## Changes Made to `dqn.py`

### Change 1: FP16 mixed precision in `_learn_phase`

Added helper `_apply_fp16` that casts both params and observations to `float16` before each forward pass, and casts the output back to `float32` for numerically stable loss computation:

```python
def _apply_fp16(params, obs):
    params_fp16 = jax.tree.map(lambda p: p.astype(jnp.float16), params)
    return network.apply(params_fp16, obs.astype(jnp.float16)).astype(jnp.float32)
```

Used for:
- Target network Q-value computation (`q_next_target`)
- Online network Q-value computation inside `_loss_fn`

Gradients via `jax.value_and_grad` are still computed through the FP32 loss, so network parameters remain in FP32.

**Rationale:** Turing tensor cores execute FP16 GEMMs at ~2× throughput vs FP32 CUDA cores. MinAtar observations are 0/1 channel activations, exactly representable in FP16. Network weights (120/84 neurons) do not overflow FP16 range.

**BF16 rejected:** Turing architecture lacks native BF16 tensor cores (Ampere+). Micro-benchmark confirmed BF16 is 0.84ms vs FP32 0.69ms for the GEMM shape used here — BF16 is slower on this GPU.

### Change 2: FP16 buffer storage for observations

Cast observations to `float16` before storing in the flashbax buffer, halving the memory footprint and the scatter-gather bandwidth for `buffer.sample()`:

```python
# Buffer initialization:
_timestep = TimeStep(obs=_obs.astype(jnp.float16), ...)
buffer_state = buffer.init(_timestep)

# Buffer add (each training step):
timestep = TimeStep(obs=last_obs.astype(jnp.float16), action=action, reward=reward, done=done)
buffer_state = buffer.add(buffer_state, timestep)
```

**Rationale:** SpaceInvaders-MinAtar observations are 10×10×4 boolean tensors (0 or 1). FP16 represents both values exactly. Buffer size 131072 × 128 obs (each 400 bytes FP32) = 53 GB at FP32; halved to 26 GB at FP16. The `loop_slice_fusion` (buffer gather) time dropped from 301ms → 149ms. Memory peak dropped from 639 MiB → 324 MiB.

---

## Files Changed

| File | Description |
|---|---|
| `dqn.py` | Main optimization: FP16 learning + FP16 buffer |
| `profile_dqn.py` | New: JAX profiler script for bottleneck analysis |
| `artifacts/benchmarks/baseline_run2.txt` | Baseline benchmark output |
| `artifacts/benchmarks/fp16_learn_only_run1.txt` | Intermediate: FP16 learn only |
| `artifacts/benchmarks/opt2_fp16buf_run2.txt` | Optimized benchmark run 2 |
| `artifacts/benchmarks/opt_final_run1.txt` | Optimized benchmark final confirmation |
| `artifacts/benchmarks/results.csv` | Structured results table |
| `artifacts/notes/event_log.md` | Full event log with BASELINE/PROFILE/HYPOTHESIS/CHANGE/EXPERIMENT tags |
| `artifacts/profiles/jax_trace/` | Perfetto trace files (.perfetto-trace) |

---

## Conclusion

Two targeted changes — FP16 tensor-core GEMMs in the learning phase and FP16 buffer storage — yield a combined **+80–85% throughput improvement** (90k → 163–167k items/s) and **−49% GPU memory** (639 → 324 MiB), with no degradation in reward quality. Both changes are motivated by direct profiler evidence and exploit properties specific to this workload: Turing tensor cores for FP16, and the exact representability of binary MinAtar observations in FP16.
