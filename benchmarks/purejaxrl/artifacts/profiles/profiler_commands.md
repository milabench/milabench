# Profiler Commands

## GPU Profiling (nsys)

### Check nsys availability
```bash
module spider cuda
module load cuda/<VERSION>
which nsys && nsys --version
```

### Profile short baseline run
```bash
nsys profile -o artifacts/profiles/baseline_nsys \
  --stats=true \
  /tmp/output/venv/torch/bin/python /tmp/milabench/benchmarks/purejaxrl/main.py \
    dqn --num_envs 128 --buffer_size 131072 --buffer_batch_size 65536 \
    --env_name SpaceInvaders-MinAtar --training_interval 10 --total_timesteps 10000
```

### Profile optimized run
```bash
nsys profile -o artifacts/profiles/opt_chunk10_nsys \
  --stats=true \
  /tmp/output/venv/torch/bin/python /tmp/milabench/benchmarks/purejaxrl/main.py \
    dqn --num_envs 128 --buffer_size 131072 --buffer_batch_size 65536 \
    --env_name SpaceInvaders-MinAtar --training_interval 10 --total_timesteps 10000 \
    --scan_chunk_size 10
```

## JAX-level timing (cProfile)

### CPU profiling with cProfile
```bash
/tmp/output/venv/torch/bin/python -m cProfile -s cumulative \
  /tmp/milabench/benchmarks/purejaxrl/main.py \
    dqn --num_envs 128 --buffer_size 131072 --buffer_batch_size 65536 \
    --env_name SpaceInvaders-MinAtar --training_interval 10 \
  2>&1 | tee artifacts/profiles/cprofile_baseline.txt
```

### py-spy (sampling profiler, attach to running process)
```bash
py-spy record -o artifacts/profiles/pyspy_baseline.svg -- \
  /tmp/output/venv/torch/bin/python /tmp/milabench/benchmarks/purejaxrl/main.py \
    dqn --num_envs 128 --buffer_size 131072 --buffer_batch_size 65536 \
    --env_name SpaceInvaders-MinAtar --training_interval 10
```

## Notes
- All profiling runs should use --total_timesteps 10000 to keep traces short.
- Baseline and optimized should be profiled under identical conditions.
- Traces stored in artifacts/profiles/ with descriptive names.