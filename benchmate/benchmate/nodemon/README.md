# nodemon: the node sampler

`benchmate.nodemon` records node-wide system health while a benchmark runs, so a slow run or a straggler node can be explained afterwards. It runs as one process per host, next to the benchmark rather than inside it, and writes one TSV file per category of metric.

It records:

| Collector | What | Source |
| --- | --- | --- |
| `gpu` | SM clock, clock event (throttle) reasons, power, temperature, utilization | NVML (`pynvml`) |
| `nvlink` | NVLink traffic per GPU, and its share of the links' peak | NVML (`pynvml`) |
| `ib` | InfiniBand traffic, link rate, `port_xmit_wait`, `symbol_error`, `link_downed` | `/sys/class/infiniband` |
| `cpu` | CPU clocks, utilization, frequency cap, temperature, throttle count, governor | `/proc/stat`, `/proc/cpuinfo`, cpufreq, hwmon |
| `mem` | Used and total host memory with a running peak, anonymous memory, page cache, OOM kills | `/proc/meminfo`, `/proc/vmstat` |
| `pcie` | Link width and speed, AER error counts, for GPUs, HCAs and the bridges above them | `/sys/bus/pci` |

`gpu`, `ib` and `mem` sample every second, `cpu` every 2 s, and `nvlink` and `pcie` every 10 s. The periods are fixed in `default_registry()`; to sample at other periods from Python, register the collectors on a `CollectorRegistry` yourself.

All values are node-wide. Nodes are assumed to be used by one job at a time, so node-wide counters are the job's counters.

## Running it

`nodemon` only runs when it has an output directory, from `BENCHMATE_NODEMON_DIR` or `--out`.  The directory must not already contain `nodemon` results.

```bash
export BENCHMATE_NODEMON_DIR=$SCRATCH/nodemon/$SLURM_JOB_ID
python -m benchmate.nodemon --bench llama &
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--out DIR` | `BENCHMATE_NODEMON_DIR` | Output directory |
| `--flush-every SECONDS` | 10 | How often pending rows are written to the files. 0 writes on every sample. |
| `--bench NAME` | empty | Recorded in the status file |
| `--nice N` | 10 | CPU scheduling niceness, -20 to 19. The default lowers the sampler's priority below the benchmark's (0), so it yields the CPU when cores are contended. Going below the current niceness needs privileges; if refused, it logs an error and keeps running. |
| `--log-level` | `INFO` | Logging on stderr. `INFO` includes a CPU-usage summary at exit. |

**Start it from anywhere.** Each host gets exactly one sampler. The first process takes the lock `nodes/<host>.lock`, and any other start on that host exits 0 at once, before opening anything. It is therefore safe to start it from every Slurm task or every local rank. The lock is released when the process exits, even when it is killed.

**Stopping:**

| Signal or event | Effect |
| --- | --- |
| SIGTERM, SIGINT, SIGHUP | Stop: pending rows are written, the status file is closed |
| SIGUSR1 | Write pending rows now, keep running (e.g. Slurm's `--signal=USR1@60`, or before copying the files) |
| Parent process gone | Stop: covers a launcher that died without signalling |
| Interpreter exit, after `run()` | Stop: `run()` registers `stop("exit")` with `atexit`, so this covers `sys.exit()` in your own SIGTERM handler |
| SIGKILL, node crash | Up to `--flush-every` seconds of rows are lost; the status stays `running` |

## Output

Node metrics are written to `$BENCHMATE_NODEMON_DIR/nodes/`, named by the short host name:

```text
nodes/
  <host>.gpu.tsv
  <host>.nvlink.tsv
  <host>.ib.tsv
  <host>.cpu.tsv
  <host>.mem.tsv
  <host>.pcie.tsv
  <host>.status.json   # how the sampler started and stopped
  <host>.lock          # ensures one sampler per host
```

A collector that is unavailable on a host (no GPU, no InfiniBand) writes no file; the status file says which collectors ran.

**A last line without a newline** is a write cut short: the sampler was killed, or the node went down, partway through writing rows.

### Status file

`<host>.status.json` is written twice. On sampler start and stop. It tells a clean stop apart from a crash:

It also records:

- `bench`: the `--bench` name;
- `start` and `end`: in milliseconds since the Unix epoch, like the `time_ms` column;
- `collectors`: each collector, whether it ran, its period, and `cpu_s`, the CPU seconds its samples used (0 while running, final once stopped);
- `flush_every`;
- `clock` and `clock_end`: whether the kernel clock was NTP-synchronized and its maximum error in µs. A node with `synced: false` has timestamps that may not line up with other nodes.
- `stuck`, if present: a collector call hung at shutdown, typically NVML on a failing GPU. Rows collected before it were still written.


## Overhead

About 0.4% of one core on a 4-GPU node, measured with no effect on benchmark rates.

The costly reads have longer periods:

- NVLink counters (~2 ms per GPU) every 10 s,
- `/proc/cpuinfo` (~2 ms on 96 cores) every 2 s.

Each collector's CPU time are in the status file (`cpu_s`).

## Using it from Python

```python
from pathlib import Path
from benchmate.nodemon import run

out = Path("/scratch/nodemon/123/")
sampler = run(out, bench="Qwen-3.8")
...
state = sampler.stop(
    "bench done"
)  # writes the pending rows and the final status, then returns
```

`run()` starts the sampler in the background. `state()` is `RUNNING` until the run ends, then `CLOSED` or `FAILED`; `BUSY` means another sampler already runs on the host, and `REFUSED` that the directory holds an earlier run. `host` defaults to the short host name, and `registry` to the default collectors writing TSV files to `out`.

`stop()` waits up to `timeout_s` (10 s) for the final status and returns the state. Don't call it from a Python signal handler, which can deadlock. A process that exits without calling it still stops cleanly; see **Stopping**.

## Tests

```bash
python -m pytest benchmate/tests/nodemon
```

## Not yet done

- milabench does not start the sampler yet. A `NodeMonitor` that launches it next to each benchmark, one per node, is planned.
- A summarizer that follows all nodes' files live.
- Upstream the SM clock and throttle reasons into voir's `gpudata`.
