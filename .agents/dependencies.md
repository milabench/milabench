# Owned dependencies

Read this when a fix belongs in one of the libraries below rather than in milabench itself.

We own these and can modify them. Clone them in `dependencies/` at the root of the milabench repo
(gitignored). A workspace may keep them elsewhere (e.g. next to the milabench checkout).

| Package | Repo | What |
|---|---|---|
| voir | https://github.com/breuleux/voir | Instrumentation and monitoring (`voir.instruments.*`: GPU, CPU, network) |
| torchcompat | https://github.com/Delaunay/torchcompat | Device compatibility layer that hides the differences between accelerators |
| wheels | https://github.com/milabench/wheels | Custom built wheels, see [wheels.md](wheels.md) |
| cantilever | https://github.com/Delaunay/cantilever | Misc instrumentation |
| argklass | https://github.com/kiwi-lang/argklass | Argument parser from dataclasses (`milabench` CLI) |

benchmate lives in this repo (`benchmate/`).

## Where a change takes effect

There are two environments, and they do not get these packages the same way:

| Package | milabench's own env (runs the CLI) | Benchmark venv (`$MILABENCH_BASE/venv/*`) |
|---|---|---|
| benchmate | editable install of `benchmate/` | editable install of `benchmate/` |
| voir | editable install of the clone (if installed with `pip install -e`) | **pinned wheel** from `.pin/` |
| torchcompat, cantilever, argklass | PyPI | PyPI / pinned |

- A change to voir or torchcompat reaches the benchmarks only after a new release or wheel is built
  and pinned ([wheels.md](wheels.md)). A local fix can ship as a `.postN` wheel in the local wheels dir.
- To try a change before that, `pip install -e <clone>` into the benchmark venv, between runs, and
  remember that `milabench install` will put the pinned version back.
- Multi-node: workers have their own checkouts and their own copy of the venv, see [multinode.md](multinode.md).
