# Result formats for the large-scale pre-training benchmarks

Prescriptive spec: what the benchmarks (TorchTitan Qwen, Megatron GLM-5) and the
window tooling must produce so that `analyze.py` can turn it into the report with no
manual step. Status: **draft, for review** (2026-10-28).

Worked examples, written by `analyze.py synth`:
`data/synthetic/rorqual/runs/syn-h6-1/tt-w-n6.0.data` (run data),
`data/synthetic/rorqual/srygaard/1030001/` (health check),
`data/synthetic/rorqual/nodemon/syn-all63-long/` (node telemetry).

## Principles

1. **Milabench run data is the container** for everything a benchmark run reports as a whole:
   one JSONL file per run (`<run>/<bench>.<n>.data`), written by voir. The only side series are
   the per-node and per-rank ones that cannot go through it (sections 3 and 4), and both are
   keyed by the run name.
2. **Every attempt leaves a record**, including runs that fail before measuring.
3. **The analysis reads raw sources only**: run data, health-check directories, node
   telemetry. Journals and summary tables are outputs, never inputs.
4. **Measured values are never overwritten.** Corrections and extrapolations go in
   `estimates`, with their method.
5. **Metadata is recorded when the run happens**, not reconstructed afterwards.

## 1. Run data (milabench JSONL)

| event | written by | content | status |
|---|---|---|---|
| `config`, `meta`, `start`, `end` | milabench | run name, system, command, **`return_code`** | exists |
| `data: {run_start: {...}}` | benchmark, once, right after `torch.distributed` init | the `run`, `placement` and `workload` blocks below | **new** |
| `data: {rate, units, task: "train", time, elapsed, step}` | benchmate, one per measured step | global tokens/s for that step | exists; **add `step`** (optional: event order is enough when no step is skipped) |
| `data: {loss, task}` | benchmark | every log step | exists |
| `data: {summary: {...}, task: "train"}` | benchmark, rank 0, after training | schema below | exists as TorchTitan v2; **move to `milabench.pretrain.v1`** |

`run_start` makes a failed run identifiable: today a crash before the summary leaves only
`end` with a return code, so job, variant and nodes are lost (run ledger, Table 3).

The official throughput is computed by the analysis from the `rate` events (median),
not read from the summary.

### Summary schema `milabench.pretrain.v1`

```jsonc
{
  "schema": "milabench.pretrain.v1",
  "run": {
    "cluster": "rorqual",            // $CC_CLUSTER
    "campaign": "window-1030",       // set by the launcher; groups runs of one session
    "variant": "H6",                 // placement family + node count, or a config name
    "repetition": 1,                 // 1, 2, ... for the same variant in the campaign
    "slurm_job_id": "25045106",
    "slurm_step_id": "3",            // when launched inside an allocation
    "slurm_reservation": "MILABENCH2026",  // $SLURM_JOB_RESERVATION, null if none
    "exclusive_cluster": false       // true in a window where the cluster is ours (set by the launcher)
  },
  "workload": {
    "model": "qwen3_8-27B", "framework": "torchtitan", "framework_commit": "51b37c2...",
    "torch_version": "2.15.0.dev20261005+cu130", "nccl_version": "2.28.3",
    "n_layers": 64, "n_layers_full": 64,          // truncated models: n_layers < n_layers_full
    "params_trainable_fraction": 1.0,             // < 1 when part of the model is frozen (e.g. GLM-5 EP4)
    "params_total": 26.9e9, "params_active": 26.9e9,
    "seq_len": 4096, "microbatch_sequences": 3, "grad_accum": 1,
    "tokens_per_gpu_per_step": 12288, "global_batch_tokens": 294912,
    "precision": "bf16",                          // "bf16" | "fp8"
    "optimizer": "AdamW", "optimizer_state_dtype": "fp32",
    "activation_checkpointing": "full", "compile": "regions",
    "moe": null                                   // MoE only, see below
  },
  "parallelism": {"gpus": 24, "nodes": 6, "dp_replicate": 1, "dp_shard": 24, "tp": 1, "pp": 1, "cp": 1, "ep": 1},
  "placement": {
    "hosts": ["rg21701", "..."],
    "family": "H",                                // S | H | O (1 per switch) | P | ALL | EP | slurm
    "ep_groups": [["rg13401", "rg13402"], ["rg13403", "rg13501"]],  // MoE: hosts of each EP group, in rank order
    "leaves": {"ihpc217u25": ["rg21701", "rg21702", "rg21703"], "...": []},
    "excluded_nodes": {"rg32101": "intra-node all-reduce 281 < 90% of 336 GB/s"},
    "health_check": "srygaard:/scratch/.../results/25045100",
    "topology_snapshot_epoch": 1793361600
  },
  "measurement": {"warmup_steps": 30, "measured_steps": 30, "measure_seconds_budget": 300,
                  "measurement_start_epoch": 1793368170.4, "measurement_end_epoch": 1793368475.9},
  "mfu_inputs": {"flops_per_token": 159258915840, "peak_flops_per_gpu": 989e12, "formula": "..."},
  "memory_gib": {"peak_allocated": {"min": 35.2, "max": 35.5}, "peak_reserved": {"min": 44.8, "max": 45.2}},
  "startup": {                                    // seconds since the earliest rank's process start
    "process_start_epoch": {"min": 0, "max": 0},
    "model_initialized": {"min_s": 10.8, "max_s": 10.9},
    "first_step_done": {"min_s": 29.3, "max_s": 29.3},
    "measurement_start": {"min_s": 324.1, "max_s": 324.1},
    "compile_cache": "warm"                       // warm | cold | none
  },
  "data_loading": {"median_s_per_step": 3.6e-6, "max_s_per_step": 1.2e-5},
  "per_rank": [{"rank": 0, "host": "rg21701", "local_rank": 0, "gpu_uuid": "GPU-...",
                "step_median": 10.131, "step_p90": 10.148, "step_max": 10.194,
                "peak_alloc_gib": 35.5, "peak_reserved_gib": 45.2}],
  "per_ep_group": [{"hosts": ["rg13401", "rg13402"], "a2a_s_per_step": 1.21}],  // MoE, optional (Megatron MoE timers)
  "deviations": [{"id": "optimizer", "text": "AdamW (not Muon)"},
                 {"id": "mtp", "text": "MTP off"}, {"id": "data", "text": "random tokens"}],
  "estimates": [{"name": "tokens_per_s_mtp_corrected", "value": 1180.0, "unit": "tok/s/GPU",
                 "method": "analytic: +x% FLOPs for one MTP layer"}]
}
```

`gpu_uuid` per rank joins the run with the node telemetry (section 3), which records the
same UUID. Without it, the join relies on `local_rank` equal to the sampler's GPU index, which
holds only with `CUDA_DEVICE_ORDER=PCI_BUS_ID`.

### MoE (GLM-5) additions

```jsonc
"moe": {"n_experts": 256, "top_k": 8, "n_shared_experts": 1, "ep": 8,
        "routing": "aux-loss-free", "capacity_factor": null, "dispatcher": "alltoall"}
```

Optional per-step event `data: {expert_load: [tokens per expert], step, task: "train"}`
every N steps, for routing imbalance.

**Megatron/slime** must produce the same events. Preferred: run under milabench with a
voirfile (as TorchTitan does). Fallback: a converter from Megatron's logs that writes the
same JSONL (`config`, `start`, `rate` per step, `summary`, `end`). The analysis then needs
no Megatron-specific code.

### What the launcher sets

`tt_variants.sh` already chooses the variant and the healthy nodes; it passes them to the
run through the environment, as it does today for `TT_RUN_TAG`:

| variable | goes to |
|---|---|
| `TT_CAMPAIGN` | `run.campaign` |
| `TT_VARIANT`, `TT_REPETITION` | `run.variant`, `run.repetition` (a real repetition counter, not the variant's position in the job) |
| `TT_FAMILY` | `placement.family` |
| `TT_HEALTH_DIR` | `placement.health_check`; `excluded_nodes` read from it |
| `TT_TOPOLOGY` (file: `scontrol show topology` saved at allocation time) | `placement.leaves`, `topology_snapshot_epoch` |

## 2. Health checks and NCCL tests: srygaard/slurm-healthcheck

Format: the tool's results directory, unchanged
(`report.json`, `switches.json`, `groups/<job>.json`, `nodes/<host>.json`).
The window's run directory must be copied with the run data (`analyze.py collect --srygaard rorqual:<path>`):
it is under the colleague's `$SCRATCH` (not readable by us today).

Each `groups/<job>.json` test carries the per-size table (for bandwidth vs. message size):

```jsonc
{"kind": "group", "nodes": [...], "status": "ok", "peak_busbw": 24.6, "avg_busbw": 12.1,
 "rows": [{"size": 8, "busbw": 0.0}, {"size": 1073741824, "busbw": 24.6}]}
```

Runs from before that change have no `rows`; the adapter then parses the raw logs in `groups/<job>/`.

### Other NCCL tests

Plain nccl-tests stdout (any `*_perf` binary), one file per test, under
`data/raw/<cluster>/nccl_tests/<job>/<condition>-<host>-<host>.txt` (e.g. the 10-24 uplink
tests: `alone-rg21701-rg31601.txt`, `concurrent-...`). Hosts come from the `Rank ... on <host>`
lines; the condition is the first word of the file name when it is not a host name.

## 3. System telemetry: nodemon TSV on `$SCRATCH`

`benchmate.nodemon` samples every node of the job, one sampler per host, started from inside the
benchmark. It records CPU clocks, IB, PCIe, XID, GPU ECC and host EDAC; on Hopper it also reads the SM and tensor-pipe counters (GPM) NVML's ordinary
calls cannot give.

### Layout

```
$SCRATCH/nodemon/<run_name>/<host>.status.json
                                 <host>.lock
                                 nodes/<host>.<collector>.tsv
```

`BENCHMATE_NODEMON_DIR` points at that directory; unset, the sampler is disabled. **Its name
must be the milabench run name** — the whole join to the run data
(`gpu_monitor.source == runs.run_name`). One run per directory (nodemon refuses to start on an
existing `status.json`), one sampler per host (`<host>.lock`): every rank may start it, only
the first collects.

Both launchers set `BENCHMATE_NODEMON_DIR=$SCRATCH/nodemon/<run_name>`. The Rorqual GLM-5
launcher always sets it, for every entry including the check ladder (`scripts/rorqual/job.sbatch`);
the Megatron pack starts it from every rank (`benchmarks/megatron/main.py`). TorchTitan's
`scripts/slurm/tt_native.sh` sets the same path; an empty value disables it.

Identity is in the paths, not the rows: a row carries `time_ms` alone, the host is in the file
name, the benchmark in the status file. One wide file per collector, melted to
`time, host, gpu, metric, value`, timestamps aligned to whole multiples of the period so hosts
join without binning. An empty cell means **unavailable, never zero**; a collector that cannot
run writes no file, and `status.json` says which.

### Collectors

| file | period | one row per | columns after `time_ms` |
|---|---|---|---|
| `<host>.gpu.tsv` | 1 s | GPU | `gpu sm_clock_mhz clocks_event_reasons power_w temp_c util util_mean util_min util_samples xid` |
| `<host>.nvlink.tsv` | 10 s | GPU | `gpu nvlink_tx_bytes nvlink_rx_bytes nvlink_tx_bytes_per_s nvlink_rx_bytes_per_s nvlink_utilization` |
| `<host>.ib.tsv` | 1 s | HCA port | `hca port rate_gbps tx_bytes rx_bytes ib_tx_bytes_per_s ib_rx_bytes_per_s ib_utilization xmit_wait symbol_error link_downed` |
| `<host>.cpu.tsv` | 2 s | package | `package busy_cores cpu_mhz_busy_mean cpu_mhz_min cpu_mhz_max cpu_mhz_mean cpu_mhz_cap cpu_util cpu_temp_c thermal_throttle_count governor` |
| `<host>.mem.tsv` | 1 s | node | `mem_used_gib mem_total_gib mem_used_peak_gib anon_gib file_gib oom_kill` |
| `<host>.pcie.tsv` | 10 s | device | `device driver link_width max_link_width link_speed max_link_speed aer_fatal aer_nonfatal aer_correctable` |
| `<host>.ecc.tsv` | 60 s | GPU | `gpu ecc_uncorrected remap_pending remap_failure` |
| `<host>.edac.tsv` | 60 s | memory controller | `mc mc_name ce_count ue_count` |
| `<host>.gpm.tsv` | 1 s | GPU | `gpu sm_util sm_occupancy tensor_util dram_bw_util` — Hopper and newer only |

```
time_ms	gpu	sm_clock_mhz	clocks_event_reasons	power_w	temp_c	util	util_mean	util_min	util_samples	xid
1793134142000	1	1410	0x0	106.832	42	0.97	0.93	0.71	71	
```

### Metric details

Units: `time_ms` is wall-clock epoch milliseconds (the run data's clock); clocks in MHz, power in
W, temperature in °C, memory in GiB, PCIe speed in GT/s; `util`, `cpu_util`, `*_utilization` and
the `gpm` columns are fractions in 0–1; `clocks_event_reasons` is the NVML bitmask as hex.
`*_bytes`, `xmit_wait`, `symbol_error`, `link_downed`, `oom_kill`, `aer_*`, `ce_count` and
`ue_count` are cumulative (the analysis takes deltas); IB bytes are already converted from the
sysfs 4-byte lanes.

- `util` is one NVML reading at the tick; `util_mean` and `util_min` cover the samples the driver
  buffered since the previous tick (`util_samples` says how many, empty when none).
  **`util_mean` is the duty cycle, and the column to compare GPUs with.**
- `gpm` columns cover the interval since the previous tick, so the first row of a run is empty.
  `tensor_util` is tensor-pipe occupancy, near 0 on fp32 work.
- `xid` is empty on almost every row; a critical XID carries its code (several `;`-joined),
  stamped with the tick that drained the event queue, up to one period late.

| signal | columns | used for |
|---|---|---|
| SM clock, clock-event reasons, power, temperature | `gpu.sm_clock_mhz`, `gpu.clocks_event_reasons`, `gpu.power_w`, `gpu.temp_c` | throttling and its cause (stability, Fig. 9) |
| GPU activity, SM clock, CPU busy-core clock | `gpm.tensor_util` else `gpu.util_mean` else `gpu.util`, `gpu.sm_clock_mhz`, `cpu.cpu_mhz_busy_mean` | stragglers (Fig. 10) |
| NVLink and IB rates, IB errors | `nvlink.*_per_s`, `ib.*_per_s`, `ib.xmit_wait`, `ib.symbol_error`, `ib.link_downed` | traffic, link faults |
| memory, OOM kills, AER counts | `mem.*`, `pcie.aer_*` | failures and node triage |
| XID, GPU ECC, host ECC | `gpu.xid`, `ecc.*`, `edac.ce_count`, `edac.ue_count` | hardware faults during the run |

The straggler figure takes the sharpest activity signal the node offers, per GPU the median of
its busiest 10-minute window, and its y-axis names which. Beside it, per node, the CPU busy-core
clock: the rg21701 case, where the CPU fell from ~3,700 to ~2,560 MHz and the run lost ~25 % of
its throughput, is visible in `cpu_mhz_busy_mean` alone.

### Joining with the run

- telemetry → run: the directory name, which is `runs.run_name`;
- row → host: the TSV file name; the node-level collectors need nothing but that and `time_ms`;
- GPU index → UUID: `status.json`, `collectors.<name>.gpus` — every NVML collector publishes it;
- UUID → rank, host, local rank: `per_rank[].gpu_uuid` in the run summary (section 1), so no
  assumption about device order.

### Status file

`<host>.status.json`, written at start and rewritten at exit:

```jsonc
{"schema": "benchmate.nodemon.v1", "state": "closed",    // running | closed | failed
 "bench": "glm5", "run_name": "tt-w-n63-long", "campaign": "tt38",
 "slurm_job_id": "11431300", "slurm_step_id": "0", "slurm_nodelist": "rg[21701-21763]",
 "pid": 1692897, "start": 1793133997863, "end": 1793134142545, "reason": "signal SIGTERM",
 "flush_every": 10.0,
 "collectors": {"gpu": {"enabled": true, "period_s": 1.0, "cpu_s": 0.026,
                        "gpus": {"0": "GPU-ee57a905-...", "1": "GPU-..."}},
                "ecc": {"...": null,
                        "lifetime": {"0": {"ecc_aggregate_uncorrected": 0,
                                           "remapped_corrected": 0, "remapped_uncorrected": 0}}},
                "gpm": {"enabled": false}},
 "clock": {"synced": true, "maxerror_us": 53126}, "clock_end": {"...": null}}
```

Reader rules: `state: "running"` with a last row older than ~2 × `flush_every` means the
sampler was killed, so the series is truncated; a final line without a newline is dropped.
`enabled: false` means the collector probed false and wrote no file (`gpm` before
Hopper, `ib` without an HCA). `clock.synced: false` or a large `maxerror_us` means this
node's timestamps cannot be lined up with the others. The run-identity fields appear only when
the launcher exported them, so the directory name stays the join of record. `gpus` and
`lifetime` are inventory, not a series: read once.

## 4. Per-rank step metrics: JSONL per rank, mirrored to WandB

The run data (section 1) gives each rank's median, p90 and max step: *that* ranks differ, not
*when*. This is the per-step series behind it, one file per rank rather than more events in the run
JSONL (voir writes that from one process, and gathering every rank's step time into it is the
measurement being made). Status: **new** — specified here first, as section 3 was.

**The record is the JSONL; WandB is for monitoring.** Each rank appends its own rows to
`<run>/metrics.<bench>.<host>.<rank>.jsonl` in the run directory (benchmate's rankstore: no
collective, flushed on the logging steps), so the join to the run is the directory itself and the
analysis reads these raw files (principle 3), never WandB. The same rows are logged to WandB for
live views: `wandb.init(group=<run_name>, name=rank<rank:05d>)`, group = `runs.run_name`, project
and entity *to fill in*, offline runs `wandb sync`ed afterwards. The two may differ if a sync is
late or lost; the files win. The files use rankstore's field names (`batch_id`, `elapsed`, `next`,
`rate`, ...), which the adapter maps to the columns below, and their first row names the worker
(host, GPU index, UUID).

| column | required | meaning |
|---|---|---|
| `time_ms` | yes | epoch ms at the end of the step, the clock of the `rate` events and section 3. In WandB, **not `_timestamp`** (upload time) |
| `rank`, `step` | yes | global rank; the step index of the `rate` events (not WandB's `_step`) |
| `tokens` | yes | tokens this rank consumed in the step: micro-batch x sequence x gradient accumulation |
| `data_wait_ms` | yes | host time blocked waiting for the batch |
| `step_time_ms` | yes | host wall time, end of the previous step to the end of this one |
| `fwd_ms`, `bwd_ms` | yes | forward and backward, device time on the compute stream (includes exposed waits, see below); `fwd_bwd_ms` instead when the schedule interleaves them; `optimizer_ms` optional |
| `local_rank`, `host`, `gpu_uuid`, `mem_alloc_peak_gib`, `mem_reserved_peak_gib` | — | `gpu_uuid` joins to section 3 and `per_rank[]` |

- **`fwd_ms` and `bwd_ms` do not separate compute from waiting, and nothing here will.** CUDA events on the compute
  stream include the exposed collective waits (all-gather in forward, reduce-scatter in backward
  under FSDP), so `step_time_ms - (fwd_ms + bwd_ms + optimizer_ms)` is mostly host-side gaps, not
  collective wait. And every rank waits for the slowest, so a slow rank computes longer and waits
  less: the columns differ little across ranks (24 FSDP ranks' step times within 0.2 ms on
  Rorqual). Splitting them per rank would need kernel- or collective-level timing, and profiler-level
  timing is not available. So the columns are recorded, not interpreted: the straggler figure rests
  on section 3 (SM clock, throttle reasons, CPU busy-core clock, GPM tensor activity), and for
  GLM-5 on Megatron's per-rank straggler detector (forward time, leaving out the waits between
  pipeline stages, including the expert all-to-all; off in the G entries, `--log-straggler`).
  A figure that needs compute and waiting apart is out of scope.
- `bwd_ms` is only comparable at the same recompute setting (backward includes it), and is
  unavailable with pipeline parallelism: use `fwd_bwd_ms` when present, else `fwd_ms + bwd_ms`.
- An empty cell means **unavailable, never zero**.

**Every rank that has the timer logs its own rows; nothing is aggregated before it is logged**, so
the spread, the slowest rank and a straggling host come from the data. **That is not every rank
yet:** the step timer is attached to the data loader, which Megatron builds only on the first and
last pipeline stages, so at PP 25 only 2 of 25 stages (about 8 % of the GLM-5 ranks) have rows, and
`tokens` and `data_wait_ms` exist only there. Until the timer moves off the loader, a rank with no
run is a rank without a timer, not necessarily a failed one. The group's `config` (the same on
every rank) says what to expect:

```jsonc
{"schema": "benchmate.rankmetrics.v1", "run_name": "glm5-g1-ep8-aligned", "world_size": 200,
 "parallelism": {"dp": 8, "expert_dp": 25, "tp": 1, "pp": 25, "cp": 1, "ep": 8}}
```

For MoE, Megatron has two data-parallel sizes: `dp` = world / (TP x PP x CP) for the dense layers, and
`expert_dp` = world / (EP x TP x PP) for the experts. Both are recorded, so no reader has to guess.

- Statistics (spread, slowest rank, p90 across ranks) are computed when reading, per `step`, over the
  runs in the group, and say how many ranks they cover.
- Read the run directory's `metrics.*.jsonl` (benchmate's `LocalMetricStore.read()` merges them). WandB
  is only a fallback when the files are gone: `wandb.Api().runs(<project>, filters={"group": run_name})`
  with `scan_history()`, not the sampled `history()`.
- Throughput stays the `rate` events of section 1; scaling `tokens` to the job is an `estimates`
  entry with its method (principle 4), not a column.
- **MFU is derived, never logged**: `flops_per_token x tokens / step_time_ms / peak_flops_per_gpu`,
  with `mfu_inputs` from the run summary. Two limits, to state wherever a per-rank MFU is shown:
  it is only defined where `tokens` is (a rank without a dataloader takes its data-parallel
  group's value, so it is an `estimates` entry), and under pipeline parallelism a stage runs part
  of the model, so a stage's MFU needs that stage's FLOPs, not the model total.

| signal | columns | used for |
|---|---|---|
| per-rank step spread | `step_time_ms` by `rank`, `step` | stragglers (Fig. 10), beside section 3 |
| compute vs. waiting | `fwd_ms`, `bwd_ms`, `optimizer_ms` | recorded only: they do not separate them, see above |
| dataloader stalls | `data_wait_ms` | `data_loading` in the summary, with the step |
| tokens per rank | `tokens` | rate vs. parallelism check, MoE per-replica accounting, per-rank MFU (estimate, see above) |

## 5. Versioning

- `schema` names the format and version. Additive changes (new optional fields) keep the
  version; renaming or removing a field bumps it.
- The analysis has one adapter per schema version. Current adapters:
  `milabench.torchtitan_native.v2` (mapping in `tt_analysis/runs.py:_v2_fields`) and
  `milabench.pretrain.v1`; for telemetry, `benchmate.nodemon.v1`
  (`tt_analysis/monitor.py`); `benchmate.rankmetrics.v1` (section 4, per-rank JSONL) has no reader yet.
- Runs before the v2 measurement path (2026-10-21) are excluded.

## Changes requested from the benchmark code (to route)

| # | change | where | why | size |
|---|---|---|---|---|
| 1 | summary as `milabench.pretrain.v1` (move `config` fields into `run` / `workload` / `parallelism` / `placement`; add `deviations`, `estimates`) | `tt_native/instrument.py` | one schema for TorchTitan and Megatron | small |
| 2 | `run_start` event after init | `tt_native/instrument.py` | failed runs keep their identity | small |
| 3 | launcher variables above + topology snapshot | `tt_variants.sh` | placement recorded at run time; real repetition index | small |
| 3c | `run.slurm_reservation` ($SLURM_JOB_RESERVATION) and `run.exclusive_cluster` (launcher flag in the window) | `instrument.py`, `tt_variants.sh` | traffic isolation of each run (placement comparisons only within one level); today inferred from the reservation tracker and the campaign | small |
| 4 | `gpu_uuid`, `local_rank` in `per_rank`; `CUDA_DEVICE_ORDER=PCI_BUS_ID` | `instrument.py`, `benchfile.py` | join with the node telemetry | small |
| 5 | `step` in `rate` events | benchmate `ManualTimedIterator` | explicit step index | small, optional |
| 6c | run both `--job-per switch` and `--job-per cluster` in the window (one leaf and whole machine); optionally `--switches A B --job-per cluster` for 2-leaf scopes (e.g. 217+316) | window plan | communication scopes (Fig. 4) | none (operational) |
| 3b | family **O** (1 node per switch, one node on each of N leaves) in the node picker | `tt_pick_nodes.py` | nodes-per-switch figure (S / H / O at 3, 6, 9, 12 nodes; O up to 16) | small |
| 7b | GLM-5 expert-group layouts on Rorqual: EP8 aligned / mixed (~1 group in 4 crossing) / crossed on the **same 52 nodes** (even number per leaf: 13 x 2 + 12 + 8 + 6; fall back to 50 / 48), only the node order differs; plus mixed on every healthy node (largest achievable, Megatron's default order, even count); full model if it fits on 208 GPUs (to confirm); record `placement.ep_groups`; per-group all-to-all time if Megatron's MoE timers allow. Picker: "even per leaf" selection + the three orders | GLM-5 pipeline, `tt_pick_nodes.py` | expert-placement figure | medium |
| 7 | same events from Megatron (voirfile or converter) | GLM-5 pipeline | MoE section | medium |
| 9 | per-rank step metrics as JSONL in the run directory, mirrored to WandB (`benchmate.rankmetrics.v1`, section 4) on every rank, with the timer off the data loader so middle pipeline stages write too, and the coverage config | benchmate (`rankmetrics.py`), `tt_native/instrument.py`, GLM-5 pipeline | per-step rank spread and the collective-wait share; today only an end-of-run aggregate exists | medium |

Not available: per-rank compute vs. NCCL time. The section 4 columns do not give it (they include the
exposed waits) and profiler-level timing is not on offer. `util_mean` is a duty cycle, not
tensor-pipe occupancy, so on nodes without GPM Fig. 10 may not separate the GPUs, and that is a
limit of the figure. Not requested: fixing voir's empty `gpudata` (nodemon covers it; still a bug).
