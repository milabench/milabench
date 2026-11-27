# Debugging failed or hung benchmarks

Read this when a benchmark fails, hangs, or reports a strange rate.

## Where the logs are

- `$MILABENCH_BASE/runs/<run name>/<bench>.data` is JSON lines. The useful events:
  `"event": "line"` (stdout/stderr lines), `"event": "error"`, `"event": "end"` (with `return_code`), and the metrics.
- **Servers and their workers (vLLM, ATOM, Ray) log to stdout**, so the root cause is in the `.data`
  file's `line` events, not in `.stderr`.
- **Grepping for "error" is misleading**: install and prepare logs list file names such as `cv2/Error`.
  Look at `return_code` first, then at the last lines before it.
- **Never print `env.sh`**: it holds HF and W&B tokens in plain text. Those tokens also appear in
  `.data` files (the config is logged), so don't paste `.data` files wholesale either.

## Hangs

- A multi-node benchmark that hangs at start: check `system.env` (NCCL vars) and that the workers' code
  is in sync ([multinode.md](multinode.md)).
- A serving benchmark that waits out `max_duration` with idle GPUs: the server's engine died but its API
  server kept running (ATOM does this). If the log shows `All runners are shutdown` and no rates, kill
  the server process (`atom_server_entry.py`); the benchmark then fails fast and the run moves on.
  Don't use `milabench tools warden` for this: it refuses during a run, or with `--force` ends the whole run.
- After the run: `milabench tools warden --dry-run` on every node, see [multinode.md](multinode.md).

## Strange rates

- Rates around `-4e17`, or a deadlock at the end of a multi-GPU benchmark: a collective (`dist.reduce`)
  called on only some ranks. Every rank must call `benchmate.metrics._make_rate`.
- Rates that are about half of the expected multi-node value: a stale worker checkout, or a worker that
  never joined (check its lines in the `.data` file).

## Crashes at exit

`terminate called after throwing an instance of '<garbage>'` / `FATAL: exception not rethrown` at exit
(seen with jax): Python ≤ 3.13 calls `pthread_exit` on any thread that takes the GIL after finalization
started, and inside C++ `noexcept` frames that ends in `std::terminate`. Find the threads still inside
native code at exit and stop them before leaving: monitor threads must be joined (voir's `Monitor.stop()`),
async work must be waited for (`jax.block_until_ready(...)`, `jax.effects_barrier()`).
