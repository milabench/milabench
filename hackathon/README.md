# RL Throughput Optimization Exercise — Human Guide (Verifier Role)

## Goal
Each human+agent pair works independently to **increase steps/sec** on DQN while keeping **best mean reward** near baseline (within 1 std of the baseline, measured on 5 runs).

Workload for today:
- Milabench PureJaxRL DQN: `benchmarks/purejaxrl/dqn.py` (via `benchmarks/purejaxrl/main.py`)
- Env: `SpaceInvaders-MinAtar`

Typical run:
- `cd benchmarks/purejaxrl`
- `python main.py dqn --env_name SpaceInvaders-MinAtar`

---

## Roles & responsibilities (IMPORTANT)

### Coding agent = primary owner (does the work *and* the logging)
The coding agent is expected to:
- Maintain `artifacts/notes/event_log.md` (timestamped entries)
- Record benchmark/eval outcomes (`artifacts/benchmarks/results.csv` or equivalent)
- Save profiling traces + commands (`artifacts/profiles/`)
- Log **human interventions** (H-STEER / H-DEBUG / H-ARCH / H-OPS)
- Write `artifacts/FINAL_SUMMARY.md` at the end (you review)

### Human = verifier only (your main job)
You are **not** responsible for writing logs during the session.

Your job is to **verify** the agent is logging properly and that critical facts are correct:
1) Confirm the following are written at the top of the log and correct:
   - baseline command (exact)
   - throughput metric definition (“what is a step?”)
   - reward metric and tolerance
   - benchmark window (fixed steps or fixed time)
2) Ensure every claimed improvement includes:
   - repeated benchmark runs (median/min/max), and
   - a reward/correctness check
3) If you do something the agent didn’t notice, prompt it:
   - “Log that as H-OPS / H-DEBUG / H-STEER / H-ARCH.”

If unsure whether something counts as intervention, **log it anyway**.

---

## Required protocol (what must happen)

### Benchmarking
- 1 warmup run (discard)
- Then 5 measured runs
- Report median/min/max
- Record: steps/sec + reward snapshot + peak GPU memory (if possible)

### Reward / correctness
- At least a quick reward check for any candidate improvement (mean over 5 runs)
- If a change could be semantic (async env, batch size, rollout logic, etc.), require stronger eval

---

## End deliverables (per agent)
- `artifacts/FINAL_SUMMARY.md`
- `artifacts/` folder with benchmarks + profiles + event log
- Git branch: `agent_<ID>_throughput_opt`
- Repro commands documented

---

## Comparison metrics (what we’ll use)
Primary:
- Throughput gain (%) at preserved reward (mean-of-runs)

Secondary:
- Time to first measurable win
- Dead-end rate / revert count
- Human intervention count
- Reproducibility + clarity of artifacts
- Diff size / invasiveness
