# nodemon: the node sampler

`benchmate.nodemon` records node-wide system health while a benchmark runs, so a slow run or a straggler node can be explained afterwards. It runs as one process per host, next to the benchmark rather than inside it, and writes one TSV file per category of metric.

It records:

| Collector | Every | What | Source |
| --- | --- | --- | --- |
| [`gpu.py`](collectors/gpu.py) | 1 s | SM clock, clock event (throttle) reasons, power, enforced power limit, temperature, utilization | NVML (`pynvml`) |
| [`nvlink.py`](collectors/nvlink.py) | 10 s | NVLink traffic per GPU, and its share of the links' peak | NVML (`pynvml`) |
| [`ib.py`](collectors/ib.py) | 1 s | InfiniBand traffic, link rate, `port_xmit_wait`, `symbol_error`, `link_downed` | `/sys/class/infiniband` |
| [`cpu.py`](collectors/cpu.py) | 2 s | CPU clocks, utilization, frequency cap, temperature, throttle count, governor | `/proc/stat`, `/proc/cpuinfo`, cpufreq, hwmon |
| [`mem.py`](collectors/mem.py) | 1 s | Used and total host memory with a running peak, anonymous memory, page cache, OOM kills | `/proc/meminfo`, `/proc/vmstat` |
| [`pcie.py`](collectors/pcie.py) | 10 s | Link width and speed, AER error counts, for GPUs, HCAs and the bridges above them | `/sys/bus/pci` |
| [`ecc.py`](collectors/ecc.py) | 60 s | Uncorrected memory errors, pending and failed row remapping | NVML (`pynvml`) |
| [`edac.py`](collectors/edac.py) | 60 s | Correctable and uncorrectable host memory errors per controller | `/sys/devices/system/edac/mc` |
| [`gpm.py`](collectors/gpm.py) | 1 s | SM, pipe and tensor activity, warp occupancy, DRAM bandwidth and PCIe throughput, on Hopper and newer | NVML GPM (`pynvml`) |
| [`dcgm.py`](collectors/dcgm.py) | disabled | The same SM and tensor-pipe activity, where a dcgm-exporter runs on the host | dcgm-exporter |

The `dcgm` collector is disabled in favor of `gpm`, which needs Hopper or newer.

The periods are fixed in [`default_registry()`](collectors/__init__.py); to sample at other periods from Python, register the collectors on a `CollectorRegistry` yourself.

## Running it

`nodemon` only runs when it has an output directory, from `BENCHMATE_NODEMON_DIR` or `--out`.  The directory must not already contain `nodemon` results, so name it after the run and give every attempt (a relaunch, a second bench of the same job) its own run name.

```bash
export BENCHMATE_NODEMON_DIR=$SCRATCH/nodemon/$MILABENCH_RUN_NAME
python -m benchmate.nodemon --bench llama &
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--out DIR` | `BENCHMATE_NODEMON_DIR` | Output directory |
| `--flush-every SECONDS` | 10 ± 2.5% | How often pending rows are written to the files. Each flush is shifted by a random ±2.5% of the period (±250 ms at 10 s) so the nodes don't all write to network storage at once. 0 writes on every sample. |
| `--bench NAME` | empty | Recorded in the status file |
| `--nice N` | 10 | CPU scheduling niceness, -20 to 19. The default lowers the sampler's priority below the benchmark's (0), so it yields the CPU when cores are contended. Going below the current niceness needs privileges; if refused, it logs an error and keeps running. |
| `--log-level` | `INFO` | Logging on stderr. `INFO` includes a CPU-usage summary at exit. |

**Start it from anywhere.** One sampler per host: the first takes the lock `<host>.lock` and any other exits 0 at once, so every Slurm task or rank can start it. The lock is released on exit, even on a kill.

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

`<host>.status.json` is written twice, when the sampler starts and when it stops, so it tells a clean stop apart from a crash:

```jsonc
{
  "schema": "benchmate.nodemon.v3",      // format of everything in the directory
  "state": "closed",                     // running | closed | failed; "running" after the run ended = killed
  "bench": "llama",                      // --bench
  // The run, from the environment; unset ones are left out.
  "run_name": "tt-w-n63-long",           // MILABENCH_RUN_NAME
  "campaign": "tt38",                    // TT_CAMPAIGN
  "slurm_job_id": "11431300",
  "slurm_step_id": "0",
  "slurm_nodelist": "rg[21701-21763]",
  "pid": 1692897,
  "start": 1793133997863,                // ms since the Unix epoch, like time_ms
  "end": 1793134142545,
  "reason": "signal SIGTERM",
  "flush_every": 10.0,
  "collectors": {
    "gpu": {
      "enabled": true,                   // false: probed unavailable, wrote no file
      "period_s": 1.0,
      "cpu_s": 0.026,                    // CPU seconds its samples used; 0 while running
      "skipped": 0,                      // due instants a late tick passed without a row
      "gpus": {"0": "GPU-ee57a905-..."}, // NVML index -> UUID, which the rows leave out
      "power_limits": {                  // at the first sample: they change only when an admin sets them
        "0": {"power_default_w": 700.0, "power_limit_w": 700.0}
      }
    },
    "ecc": {
      "enabled": true, "period_s": 60.0, "cpu_s": 0.004, "skipped": 0,
      "gpus": {"0": "GPU-ee57a905-..."},
      // Lifetime error counts at the first sample, at the latest, and their difference.
      // Both read on the sampling thread: no NVML query at startup or shutdown can block this file.
      "snapshot_start": {"0": {"ecc_aggregate_uncorrected": 0, "remapped_corrected": 0, "remapped_uncorrected": 0}},
      "snapshot":       {"0": {"ecc_aggregate_uncorrected": 0, "remapped_corrected": 0, "remapped_uncorrected": 0}},
      "snapshot_delta": {"0": {"ecc_aggregate_uncorrected": 0, "remapped_corrected": 0, "remapped_uncorrected": 0}}
    },
    "dcgm": {
      "enabled": true, "period_s": 10.0, "cpu_s": 0.02, "skipped": 0,
      "url": "http://localhost:9400/metrics",
      "exporter_gpus": {"0": "GPU-ee57a905-..."} // the exporter's numbering, not NVML's: compare with gpus
    },
    "gpm": {"enabled": false, "period_s": 1.0, "cpu_s": 0.0, "skipped": 0}
  },
  // Kernel clock at start and stop. synced: false = timestamps may not line up with other nodes.
  "clock": {"synced": true, "maxerror_us": 53126},
  "clock_end": {"synced": true, "maxerror_us": 53130},
  "stuck": "gpu"                         // only if a collector hung at shutdown (typically NVML on a
                                         // failing GPU); the rows read before it were still written
}
```


## Overhead

CPU time over 90 s on idle Trillium nodes (96 cores), all collectors at their default periods:

| Node | CPU | `gpu` (1 s) | `nvlink` (10 s) | everything else |
| --- | --- | --- | --- | --- |
| 4 × H100 | 1.5% of one core | ~11 ms per sample | ~12 ms per sample | 0.3% |
| 8 × B200 | 4.1% of one core | ~26 ms per sample | ~107 ms per sample | 0.4% |

NVML reads dominate and grow with the GPU count; NVLink is the costliest per sample, hence its 10 s period. `/proc/cpuinfo` takes ~3 ms on 96 cores, read every 2 s.

Each collector's CPU time is in the status file (`cpu_s`) and in the summary logged at exit.

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

## Monitoring a live run

```bash
python -m benchmate.nodemon.watch $BENCHMATE_NODEMON_DIR --follow 30
```

Each refresh prints one block: GPU clocks with throttle reasons, utilization, power and temperature; NVLink and IB rates, IB errors and congestion; CPU busy-core clock and utilization; stragglers (a GPU or CPU package well below its peers); and faults (XIDs, row remaps, ECC/EDAC, PCIe errors or degraded links, OOM kills, unsynchronized clocks, dead samplers). It ends with the nodes ranked by score, worst first: the ones to look at or drain.

| Option | Default | Meaning |
| --- | --- | --- |
| `--follow SECONDS` | off | Refresh every SECONDS until every sampler finishes: redrawn in place on a terminal, appended when piped. Off prints one block and exits. |
| `--window SECONDS` | 60 | History each refresh covers |
| `--flagged-hosts N` | 8 | Hosts or devices a line names before counting the rest |

## Not yet done

- milabench does not start the sampler itself yet. A pack can start it from its own code (the GLM-5 pack calls
  `run()` from every rank, see `benchmarks/megatron/main.py`); a `NodeMonitor` that launches it next to every
  benchmark, one per node, is planned.
- Upstream the SM clock and throttle reasons into voir's `gpudata`.
