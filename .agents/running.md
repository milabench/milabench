# Running milabench: install, prepare, run, report

Read this when you are asked to install, prepare or run benchmarks, or to check the results.
Multi-node setup is in [multinode.md](multinode.md), failures in [debugging.md](debugging.md).

## Layout on disk

| Path | What |
|---|---|
| `$MILABENCH_BASE` (e.g. `/mnt/milabench/results`) | Everything milabench writes |
| `$MILABENCH_BASE/venv/torch` | The shared venv (`install_group: torch`): **one venv for most benchmarks** |
| `$MILABENCH_BASE/data` | Datasets and weights; `data/hub` is the HF hub cache |
| `$MILABENCH_BASE/cache` | Third-party caches (torch, triton, flashinfer), kept off `~/.cache` |
| `$MILABENCH_BASE/runs/<run name>/*.data` | One JSON-lines file per benchmark: events, metrics, logs, `return_code` |
| `env.sh` next to `$MILABENCH_BASE` | Created by `scripts/setup_new_machine.sh --yes`; exports `MILABENCH_*` and tokens |

Environment variables that drive every command:
- `MILABENCH_CONFIG`: the benchmark config (`config/standard.yaml`, `config/experimental.yaml`, ...).
  Benchmark `definition:` paths are resolved relative to it, so point it at **this** checkout.
- `MILABENCH_SYSTEM`: the system file (node list, `system.env`), see [multinode.md](multinode.md).
- `MILABENCH_BASE`, `MILABENCH_GPU_ARCH` (`cuda`, `rocm`, `xpu`, ...).

A workspace may wrap these in `install.sh` / `prepare.sh` / `run.sh` that source `env.sh`
and forward extra args. Use the wrappers when they exist.

## Steps

1. **Install**: `milabench install --variant <arch> --set torch=<X.Y> <arch>=<ver>`
   (e.g. `--variant rocm --set torch=2.15 rocm=10.0`). Installs from the pin
   `.pin/constraints.<arch><ver>.torch<ver>.txt`, see [wheels.md](wheels.md).
   - Install **swallows per-benchmark errors** and still exits 0. Check `return_code` in
     `$MILABENCH_BASE/runs/install.*/*.data`, not the console.
   - Install **never uninstalls**: after a pin drops a package, uninstall it by hand from the venv on every node.
   - Benchmarks with `install_variant: unpinned` still install from the pin when `--variant` is forced.
     Keep it that way when they share the venv: unpinned would float shared packages (e.g. transformers).
2. **Prepare** (datasets, weights): `milabench prepare`. Gated models need `MILABENCH_HF_TOKEN`.
   Check `return_code` the same way. After prepare, `run` must work offline.
3. **Run**: `milabench run --run-name <name> [--select a,b,c]`.
4. **Report**: `MILABENCH_PUBLISH_KEY= milabench report --runs $MILABENCH_BASE/runs/<name>`
   (empty publish key: don't push results).

Useful selections: `--select <bench>[,<bench>]`, or another config with `MILABENCH_CONFIG`
(e.g. `config/experimental.yaml` for torchtitan, openinstruct and the serving benchmarks).

## Rules while a run is going

- **Don't install**: the venv is shared, and the workers get it re-synced.
- **Don't touch the GPUs**, not even for a 5-second smoke test. milabench's GPU warden kills every
  process holding a GPU when a benchmark starts or ends. Smoke-test between runs.
- **Never build GPU code on the host** during a run, for the same reason. Build in a container or
  a `PrivateDevices=yes` systemd unit ([wheels.md](wheels.md)).
- Large downloads (`hf download <repo>` with `HF_HUB_CACHE=$MILABENCH_BASE/data/hub`) are fine during
  a run: they are not milabench commands, so the warden does not fire, and prepare then finds them cached.

## Expected results

Write the reference score of each machine here once a clean run exists.

| Machine | Config | Score | Failure rate | Noisy benchmarks |
|---|---|---|---|---|
| 2× MI355X (torch 2.15 / ROCm 10.0) | standard | 7420–7540 | 0% | brax ±20%, recursiongfn and whisper ±8% |
