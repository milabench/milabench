# Trillium B200 campaign

Large-scale pre-training benchmarks on Trillium's B200 partition (52 nodes x 8 B200): design, result
formats, analysis and report. The Rorqual campaign (`scripts/analysis/`) is separate; its parsers and
style are reused by import.

```bash
cd scripts/trillium/analysis
PY=../../analysis/.venv/bin/python               # same requirements (requirements.txt)
$PY analyze.py all                               # collect, synth, build, plot, compile the report
$PY analyze.py all --no-collect                  # from the local copy in data/raw
$PY analyze.py all --no-synthetic                # measured data only (final report)
$PY analyze.py collect | synth | build | plot | report
```

| path | content |
|---|---|
| `DESIGN.md` | experiments, the question each answers, the figure each feeds, budget |
| `FORMAT.md` | result formats: the Trillium additions to `scripts/analysis/FORMAT.md` |
| `analysis/analyze.py` | CLI |
| `analysis/tri_analysis/machine.py` | nodes (`N_NODES`), subnets, peak FLOP/s, network |
| `analysis/tri_analysis/runs.py` | run-data additions (layout, recipe, loss, checkpoints); the rest via `tt_analysis.runs` |
| `analysis/tri_analysis/fabric.py` | D1 health (srygaard + matmul / HCA), D2 pairs, D3-D4 collectives |
| `analysis/tri_analysis/build.py` | raw sources -> CSV tables in `analysis/data/tables/` |
| `analysis/tri_analysis/synthetic.py` | synthetic data in the target formats, toy model in its docstring |
| `analysis/tri_analysis/style.py`, `figures.py` | style, one function per figure ID of DESIGN.md |
| `analysis/tri_analysis/tables.py` | `macros.tex` (`\Tri*`), `qwen_layouts.tex`, `ledger.tex` |
| `report/` | LaTeX report; `report/generated/` is written by `analyze.py`, never edited |
| launch scripts | to come |

Collection is read-only: rsync from `trillium-gpu:$SCRATCH/trillium/{runs,health,nccl,nodemon}` and
only the JSON metadata of `checkpoints/`; when ssh fails it says so and keeps the local copy.
`analysis/data/` and `report/generated/` are not versioned.

Synthetic data stands in for every experiment not run yet: hollow markers, a SYNTHETIC watermark in
the figure, `\syntheticnote{<id>}` in the text, and every macro computed from it wrapped in
`\TriSyn{}` (red, "syn").
