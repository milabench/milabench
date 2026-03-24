# Coding Agent Handoff — RL Throughput Optimization
## Workload: Milabench PureJaxRL DQN + SpaceInvaders-MinAtar

### One-sentence goal
Increase **steps/sec** on the DQN pipeline while **preserving best mean reward** (within the agreed tolerance).

---

## 1) Task definition (this exercise)

### Target workload
- Benchmark code: `benchmarks/purejaxrl/dqn.py`
- Entry point: `benchmarks/purejaxrl/main.py` (runs subcommands like `dqn`)
- Environment: `SpaceInvaders-MinAtar`
- Env reference (read-only, context only): `gymnax/environments/minatar/space_invaders.py`  
  **Do not modify env code** unless explicitly asked. It’s only referenced in docs.

### What “steps” means
Default definition: **environment steps** (env transitions aggregated across vectorized envs).  
If the benchmark defines “steps/sec” differently, you must:
- write the benchmark’s definition explicitly in logs, and
- report that definition consistently across baseline and best.

### Allowed work
- Systems/engineering optimizations: JAX compilation behavior, Python overhead, data movement, logging overhead, batching, replay buffer sampling, vectorization, synchronization points, etc.
- Small refactors when clearly tied to bottlenecks and backed by measurements.

### Disallowed unless explicitly approved (semantic changes)
- Changing the RL algorithm, reward definition, or environment dynamics
- Changing evaluation definition or metric definitions
- “Optimizations” that reduce work (fewer steps/episodes) without documenting and obtaining approval

If a change might be semantic, label it **SEMANTIC CHANGE** and run stronger reward checks.

---

## 2) Ownership: logging & interventions (IMPORTANT)

You (the coding agent) are the **primary scribe**.

You must:
- Create and maintain `artifacts/notes/event_log.md`
- Record every benchmark/eval result in a consistent form (`artifacts/benchmarks/results.csv` or equivalent)
- Save profiling artifacts and profiler commands under `artifacts/profiles/`
- Log **human interventions** using H-* tags whenever the human:
  - reframes goal/constraints (**H-STEER**)
  - finds bug/root cause (**H-DEBUG**)
  - makes an architecture decision (**H-ARCH**)
  - fixes ops/environment/run issues (**H-OPS**)

If unsure whether something counts as an intervention, **log it anyway**.  
If the human prompts “log that as H-OPS”, comply and add a short entry.

Human responsibility: **verify** you are logging correctly, and prompt you when you missed an intervention.

---

## 3) Fill-in metadata (do this first)

Create `artifacts/notes/event_log.md` and paste/fill:

- Date:
- Agent ID: A / B / C / D
- Human operator:
- Repo + remote:
- Starting commit hash:
- Branch name (create now): `agent_<ID>_throughput_opt`
- Hardware: GPU / CPU / RAM
- Software: driver / CUDA / JAX / jaxlib / Python
- Baseline command (exact):
- Benchmark window (fixed steps or fixed time):
- Throughput metric name & definition:
- Reward metric name & definition:
- Reward tolerance (explicit):
  - e.g., best_mean_reward not worse than baseline by more than X%

---

## 4) How to run (typical)

### Direct Python (recommended for iteration)
From repo root:

- `cd benchmarks/purejaxrl`
- `python main.py dqn --env_name SpaceInvaders-MinAtar`

### Milabench dev environment (if using Milabench tooling)

- `milabench dev --config benchmarks/purejaxrl/dev.yaml`
- inside dev shell:
  - `cd benchmarks/purejaxrl`
  - `python main.py dqn --env_name SpaceInvaders-MinAtar`

Note: ensure the actual run uses `SpaceInvaders-MinAtar` even if defaults differ.

---

## 5) Success criteria

### Primary objective
- Demonstrate a **measured** improvement in throughput (steps/sec) vs baseline.

### Constraint: preserve reward
Do not degrade reward beyond the agreed tolerance.

Choose ONE tolerance before starting and write it in metadata:
- Option A: best_mean_reward within **-1%** of baseline under agreed eval protocol
- Option B: within **-0.5 baseline std** (requires baseline variance estimate)
- Option C: explicit rule such as “no meaningful drop over X eval episodes”, documented

If reward is noisy, do not hand-wave. Report variance and decide PASS/FAIL/INCONCLUSIVE explicitly.

---

## 6) Benchmark protocol (REQUIRED)

### 6.1 Keep it comparable
- Same command/config for baseline and comparisons (unless testing config changes)
- Same benchmark window (either “run for X steps” or “run for Y seconds”; be explicit)
- Warmup then repeats

### 6.2 Warmup + repeats
- 1 warmup run (discard)
- Then N=3–5 measured runs
- Record median/min/max

### 6.3 Metrics to record per measured run
Minimum:
- Throughput (steps/sec) + definition of “steps”
- Reward snapshot (best_mean_reward or equivalent)
- Peak GPU memory (MiB), if available

Optional but helpful:
- GPU util average
- CPU util average
- Compile time (if significant)
- Notes on variability

Store:
- `artifacts/benchmarks/baseline.txt` (raw output)
- `artifacts/benchmarks/results.csv` (structured summary; OK to use TSV or markdown table if CSV is inconvenient)

---

## 7) Correctness / reward checks (REQUIRED)

Each candidate improvement must include at least one check:
- Smoke: no NaNs/inf; loss finite; no crash for short horizon
- Quick reward check: fixed small eval protocol; compare to tolerance
- If **SEMANTIC CHANGE**: stronger eval (more episodes, longer horizon, multiple seeds)

Rules:
- If throughput improves but reward fails tolerance → does not count as success.
- If reward is inconclusive (too noisy / too short), label **INCONCLUSIVE** and explain.

---

## 8) Profiling evidence (REQUIRED)

Produce at least one profiler artifact:
- GPU/system: Nsight Systems (`nsys`) or equivalent
- JAX-level: `jax.profiler` / instrumentation
- CPU-level: `py-spy`, `cProfile`, `perf`, etc.

For each profiling session, save:
- profiler command line
- trace filename(s)
- brief notes: bottlenecks observed, hypothesis formed

Store under:
- `artifacts/profiles/`
- `artifacts/profiles/profiler_commands.md`

---


---

## 8.1 Nsight Systems (`nsys`) on Alliance/Slurm clusters (REQUIRED when profiling GPU)
When you need to profile GPU behavior with Nsight Systems:

### A) Make `nsys` available via environment modules
1. Discover available CUDA toolkits / Nsight modules:
   - `module spider cuda`
   - (optionally) `module spider nvidia` or `module spider nsight`

2. Load an appropriate CUDA toolkit module (this typically provides `nsys`):
   - `module load cuda/<VERSION>`
   - Verify:
     - `which nsys`
     - `nsys --version`

If `nsys` is still not found after loading CUDA, try `module spider nsight` and load the suggested Nsight module.

### B) Running `nsys` in Slurm (profiling GPU tasks)
If you need to launch a job under Slurm for profiling, follow the Alliance guidance here:
- https://docs.alliancecan.ca/wiki/Using_GPUs_with_Slurm#Profiling_GPU_tasks

Key expectation:
- Use the Alliance-recommended Slurm/job-script pattern for profiling (cluster policies may require special handling).

**Important:** On some Alliance clusters, GPU profiling may require extra steps (e.g., temporarily disabling DCGM) per the Alliance documentation above. Do not guess—follow the doc’s procedure for your cluster.

### C) Minimal `nsys` usage pattern (example)
In a Slurm job (or within an interactive allocation), the common pattern is:

- `nsys profile -o artifacts/profiles/<name> --stats=true <your_command_here>`

Keep traces small:
- profile a short, representative window
- prefer fewer iterations/steps during profiling
- save the exact command you used in `artifacts/profiles/profiler_commands.md`


## 9) Logging during the session (REQUIRED)

### 9.1 Event log location
`artifacts/notes/event_log.md`

### 9.2 Minimal event entry format (use this)
```
T+___  [TAG]
Action/Change: ___
Hypothesis/Reason: ___
Result: ___  (metric: ___; baseline: ___; delta: ___)
Evidence: ___ (log path / trace filename)
Next: ___
```

### 9.3 Allowed tags
- BASELINE, PROFILE, HYPOTHESIS, CHANGE, EXPERIMENT, BUG, FIX, REVERT, DISCOVERY, BLOCKED, HANDOFF
- Human interventions: H-STEER, H-DEBUG, H-ARCH, H-OPS

### 9.4 If you get stuck
Log a BLOCKED entry: cause, what you tried, next hypothesis.
Then switch to profiling or add coarse timers to isolate bottlenecks.

---

## 10) Comparability rule: logging overhead & sync points
This benchmark includes periodic logging and may include synchronization points (e.g., scalar extraction, callbacks).

If you change logging frequency, callback behavior, or synchronization:
1) Mark the change as **MEASUREMENT/LOGGING OVERHEAD** optimization.
2) Report throughput:
   - with original logging behavior (comparable), and
   - with modified logging behavior (if you keep it).
3) Ensure reward checks remain valid.

---

## 11) Deliverables (MUST be produced)

### 11.1 Code
- Branch: `agent_<ID>_throughput_opt`
- Commits should mention the bottleneck addressed.

### 11.2 Artifacts folder (REQUIRED structure)
```
artifacts/
  benchmarks/
    baseline.txt
    results.csv
  profiles/
    <trace files>
    profiler_commands.md
  notes/
    event_log.md
  FINAL_SUMMARY.md
```

### 11.3 Final summary
Create `artifacts/FINAL_SUMMARY.md` using `FINAL_SUMMARY_TEMPLATE.md` (provided).

---

## 12) How you will be compared (scorecard)
Primary:
- Throughput gain (%) at preserved reward

Secondary:
- Time to first measurable win
- Dead-end rate (failed experiments / reverts)
- Human intervention count/severity
- Reproducibility (clear commands + stable results)
- Engineering quality (maintainable diff; minimal invasiveness)
- Evidence-based reasoning (profiling → targeted fixes)
