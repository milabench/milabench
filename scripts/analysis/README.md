# Analysis of the large-scale pre-training benchmarks

Raw results → tidy tables → figures → LaTeX report (NeurIPS 2026 template), with one command.

```bash
cd scripts/analysis
uv venv --python 3.12 .venv && VIRTUAL_ENV=$PWD/.venv uv pip install -r requirements.txt   # once

.venv/bin/python analyze.py all                 # collect from clusters, synth, build, plot, compile report
.venv/bin/python analyze.py all --no-collect    # from the local copy in data/raw
.venv/bin/python analyze.py all --no-synthetic  # measured data only (final report)
.venv/bin/python analyze.py quick               # during the window: latest runs + scaling/placement figures
```

Output: `report/main.pdf`, figures in `report/generated/figures/`, tables in `data/tables/`.

| path | content |
|---|---|
| `FORMAT.md` | result formats the benchmarks must produce (prescriptive) |
| `EXPERIMENTS.md` | checklist of experiments to run (before Oct 30 / window), by group |
| `WINDOW.md` | window runbook: blocks in order, the command of each, chaining, exclusions |
| `GLM_RUNS.md` | GLM-5 runs for the GLM owner: fixed settings, node orders, runs, prep checklist |
| `analyze.py` | CLI |
| `tt_analysis/runs.py` | milabench run data → runs, steps, ranks |
| `tt_analysis/health.py` | srygaard health checks, tt_healthcheck → node_health, comm |
| `tt_analysis/topology.py` | `scontrol show topology` → leaf map, placement |
| `tt_analysis/monitor.py` | nodemon TSVs → gpu_monitor |
| `tt_analysis/build.py` | all sources → CSV tables |
| `tt_analysis/synthetic.py` | synthetic data in the target formats (experiments not run yet) |
| `tt_analysis/style.py`, `figures.py` | seaborn style (minimal), one function per figure |
| `tt_analysis/tables.py` | LaTeX tables and macros |
| `report/` | LaTeX source; `sections/*.tex` hold the text |

Collection is read-only on the login nodes (rsync of
`$SCRATCH/tt38/{output/runs,health,logs,nodemon}` and `scontrol show topology`), following
`journals/cluster-guidelines.md`.
`data/` and `report/generated/` are not versioned: everything is regenerated.
