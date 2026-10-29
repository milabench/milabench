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
| `ecc` | Uncorrected memory errors, pending and failed row remapping | NVML (`pynvml`) |
| `edac` | Correctable and uncorrectable host memory errors per controller | `/sys/devices/system/edac/mc` |
| `gpm` | SM and tensor-pipe activity, warp occupancy, DRAM bandwidth, on Hopper and newer | NVML GPM (`pynvml`) |
| `dcgm` | The same SM and tensor-pipe activity, where a dcgm-exporter runs on the host | dcgm-exporter |

Each collector's `metrics` documents its columns, in file order: the name, the unit, a description,
and whether the column names the device a row is for (`device`), is a reading at the tick (`sample`)
or counts since boot or driver load (`counter`). `watch` reads the same objects for its rules.

The `gpu` rows also carry `xid`: the critical XID errors NVML delivered since the previous row,
empty on every other row. It is a column rather than its own file so the GPU grid stays fixed;
an event is attributed to the row that drains it, so its time is up to one period late.

`ecc` is deliberately separate from `gpu`, although it is only four columns. Its queries are the
ones that go slow on a failing GPU — exactly when their values matter — and in their own
collector a hang costs ECC alone, not the one-second series. For the same reason only the
run-scoped values are rows: the lifetime error and remapped-row totals are written once into the
status file instead.

`gpu`, `ib`, `mem`, `gpm` and `dcgm` sample every second, `cpu` every 2 s, `nvlink` and `pcie` every
10 s, and `ecc` and `edac` every 60 s. The periods are fixed in `default_registry()`; to sample at other periods from Python, register the collectors on a `CollectorRegistry` yourself.

All values are node-wide. Nodes are assumed to be used by one job at a time, so node-wide counters are the job's counters.

## Running it

`nodemon` only runs when it has an output directory, from `BENCHMATE_NODEMON_DIR` or `--out`.  The directory must not already contain `nodemon` results, so name it after the run and give every attempt (a relaunch, a second bench of the same job) its own run name.

```bash
export BENCHMATE_NODEMON_DIR=$SCRATCH/nodemon/$MILABENCH_RUN_NAME
python -m benchmate.nodemon --bench llama &
```

`gpm` and `dcgm` measure the same thing two ways, and both are off where they cannot answer.
GPM reads the SM counters through NVML, with no exporter and no host engine, but needs Hopper or
newer; its metrics are the difference between two samples, so the first row of a run is empty.
Where GPM is unavailable the exporter is the fallback, and where neither answers GPU activity is
the duty cycle in the `gpu` row's `util_mean`.

The `dcgm` collector only reads a `dcgm-exporter` that is already running (by default
`http://localhost:9400/metrics`, `url=` to change it). Scraping one is a plain read of what the
host already publishes; the sampler never runs `dcgmi`, never starts a host engine and never
creates a watch, because the DCGM engine is shared with everything else on the node. Without an
exporter, or when it does not publish the `DCGM_FI_PROF_*` fields, the collector switches itself
off and GPU activity comes from NVML alone.

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

Node metrics are written under `$BENCHMATE_NODEMON_DIR`, named by the short host name. The rows
go in `nodes/`, so each host's status and lock stay visible beside it:

```text
$BENCHMATE_NODEMON_DIR/
  <host>.status.json   # how the sampler started and stopped
  <host>.lock          # ensures one sampler per host
  nodes/
    <host>.gpu.tsv
    <host>.nvlink.tsv
    <host>.ib.tsv
    <host>.cpu.tsv
    <host>.mem.tsv
    <host>.pcie.tsv
    <host>.ecc.tsv
    <host>.edac.tsv
    <host>.gpm.tsv     # only on GPUs that support GPM (Hopper and newer)
    <host>.dcgm.tsv    # only where a dcgm-exporter answers
```

A collector that is unavailable on a host (no GPU, no InfiniBand) writes no file; the status file says which collectors ran.

**A last line without a newline** is a write cut short: the sampler was killed, or the node went down, partway through writing rows.

### Status file

`<host>.status.json` is written twice. On sampler start and stop. It tells a clean stop apart from a crash:

It also records:

- `schema`: the format of everything in the directory, currently `benchmate.nodemon.v1`;
- `bench`: the `--bench` name;
- the run it belongs to, from the environment: `run_name` (`MILABENCH_RUN_NAME`), `campaign`
  (`TT_CAMPAIGN`), `slurm_job_id`, `slurm_step_id` and `slurm_nodelist`. Whatever is unset is
  left out;
- `start` and `end`: in milliseconds since the Unix epoch, like the `time_ms` column;
- `collectors`: each collector, whether it ran, its period, and `cpu_s`, the CPU seconds its samples used (0 while running, final once stopped). A collector may add its own fields here: the NVML ones publish `gpus`, the index-to-UUID map the rows leave out; `ecc` adds each GPU's `lifetime` error totals; `dcgm` records the exporter `url` it read and, as `exporter_gpus`, the exporter's own GPU numbering, which is not NVML's and is kept apart from `gpus` so the two can be compared;
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

## Following a run

```bash
python -m benchmate.nodemon.watch $BENCHMATE_NODEMON_DIR --follow 30
```

One block per refresh: the lowest SM clock with its throttle reasons (a Note column names the GPU or node behind a reading), how busy the GPUs are, their power and temperature,
NVLink and IB rates, IB link errors and congestion (a port whose `xmit_wait` rises far faster than the other ports'), the median CPU busy-core clock, how busy the CPUs are
and any CPU package or GPU whose clock is well below its peers' (the stragglers), host memory, XIDs and pending or failed row remaps,
uncorrected ECC and EDAC errors, OOM kills, and any node whose sampler has died while its status still says
`running`.

Nodes ranked by score, worst first: the nodes to look at, or drain, before the others.

`--flagged-hosts` (8) is how many hosts or devices a line names before counting the rest. `--window` sets how much history each refresh covers (60 s by default). With `--follow` on a terminal the block is redrawn in place, below a blank line instead of its time header; piped or
redirected, each refresh is appended. Without `--follow` it
prints one block and exits.

## Not yet done

- milabench does not start the sampler itself yet. A pack can start it from its own code (the GLM-5 pack calls
  `run()` from every rank, see `benchmarks/megatron/main.py`); a `NodeMonitor` that launches it next to every
  benchmark, one per node, is planned.
- Upstream the SM clock and throttle reasons into voir's `gpudata`.
