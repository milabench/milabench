# Profiler Commands

1. Baseline Profile (after StepTimer fix):
`nsys profile -o ~/milabench/benchmarks/purejaxrl/artifacts/profiles/baseline_profile --stats=true ~/output/venv/torch/bin/python ~/milabench/benchmarks/purejaxrl/main.py dqn --env_name SpaceInvaders-MinAtar --total_timesteps=10000`
