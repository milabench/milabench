# Result formats for the large-scale pre-training benchmarks

Prescriptive spec: what the benchmarks (TorchTitan Qwen, Megatron GLM-5) and the
window tooling must produce so that `analyze.py` can turn it into the report with no
manual step. Status: **draft, for review** (2026-10-25).

Worked examples, written by `analyze.py synth`:
`data/synthetic/rorqual/runs/syn-h6-1/tt-w-n6.0.data` (run data),
`data/synthetic/rorqual/srygaard/1030001/` (health check),
`data/synthetic/rorqual/dcgm/syn-all63-long.csv` (GPU telemetry).

## Principles

1. **Milabench run data is the container** for everything a benchmark run produces:
   one JSONL file per run (`<run>/<bench>.<n>.data`), written by voir. No side files.
2. **Every attempt leaves a record**, including runs that fail before measuring.
3. **The analysis reads raw sources only**: run data, health-check directories, GPU
   telemetry exports. Journals and summary tables are outputs, never inputs.
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

`gpu_uuid` per rank joins the run with DCGM telemetry (section 3). Without it, the join
relies on `local_rank` equal to the DCGM GPU index, which holds only with
`CUDA_DEVICE_ORDER=PCI_BUS_ID`.

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

Requested addition (for bandwidth vs. message size): in each `groups/<job>.json` test, the
per-size table that `nccl.parse()` already computes:

```jsonc
{"kind": "group", "nodes": [...], "status": "ok", "peak_busbw": 24.6, "avg_busbw": 12.1,
 "rows": [{"size": 8, "busbw": 0.0}, {"size": 1073741824, "busbw": 24.6}]}
```

Until then, the adapter can parse the raw logs in `groups/<job>/`.

### Other NCCL tests

Plain nccl-tests stdout (any `*_perf` binary), one file per test, under
`data/raw/<cluster>/nccl_tests/<job>/<condition>-<host>-<host>.txt` (e.g. the 10-24 uplink
tests: `alone-rg21701-rg31601.txt`, `concurrent-...`). Hosts come from the `Rank ... on <host>`
lines; the condition is the first word of the file name when it is not a host name.

## 3. GPU telemetry: DCGM via Prometheus

Export of a Prometheus range query over the window, flattened to CSV, one file per
export (`dcgm/<name>.csv`):

```
time,host,gpu,uuid,metric,value
1793368170,rg21701,0,GPU-...,DCGM_FI_DEV_SM_CLOCK,1755
```

| metric | used for |
|---|---|
| `DCGM_FI_DEV_SM_CLOCK`, `DCGM_FI_DEV_CLOCKS_EVENT_REASONS` | throttling (stability, Fig. 9) |
| `DCGM_FI_DEV_POWER_USAGE`, `DCGM_FI_DEV_GPU_TEMP` | cause of throttling |
| `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE`, `DCGM_FI_PROF_SM_ACTIVE` | stragglers (Fig. 10): the slow GPU is the busiest |
| `DCGM_FI_DEV_XID_ERRORS`, ECC counters | failures |
| `DCGM_FI_PROF_NVLINK_TX_BYTES`, `DCGM_FI_PROF_PCIE_TX_BYTES` | traffic (optional) |

Requirements to check with the admins: the `DCGM_FI_PROF_*` fields are enabled in the
exporter (they are not in every default config); scrape interval ≤ 15 s (a 5-minute run
must have ≥ 20 samples); host labels match Slurm node names.

## 4. Versioning

- `schema` names the format and version. Additive changes (new optional fields) keep the
  version; renaming or removing a field bumps it.
- The analysis has one adapter per schema version. Current adapters:
  `milabench.torchtitan_native.v2` (mapping in `tt_analysis/runs.py:_v2_fields`) and
  `milabench.pretrain.v1`.
- Runs before the v2 measurement path (2026-10-21) are excluded.

## Changes requested from the benchmark code (to route)

| # | change | where | why | size |
|---|---|---|---|---|
| 1 | summary as `milabench.pretrain.v1` (move `config` fields into `run` / `workload` / `parallelism` / `placement`; add `deviations`, `estimates`) | `tt_native/instrument.py` | one schema for TorchTitan and Megatron | small |
| 2 | `run_start` event after init | `tt_native/instrument.py` | failed runs keep their identity | small |
| 3 | launcher variables above + topology snapshot | `tt_variants.sh` | placement recorded at run time; real repetition index | small |
| 3c | `run.slurm_reservation` ($SLURM_JOB_RESERVATION) and `run.exclusive_cluster` (launcher flag in the window) | `instrument.py`, `tt_variants.sh` | traffic isolation of each run (placement comparisons only within one level); today inferred from the reservation tracker and the campaign | small |
| 4 | `gpu_uuid`, `local_rank` in `per_rank`; `CUDA_DEVICE_ORDER=PCI_BUS_ID` | `instrument.py`, `benchfile.py` | join with DCGM | small |
| 5 | `step` in `rate` events | benchmate `ManualTimedIterator` | explicit step index | small, optional |
| 6 | per-size rows in group JSON | srygaard/slurm-healthcheck | busbw vs. message size | small (their tool) |
| 6b | per-GPU compute number (e.g. bf16 matmul TFLOPS, as `tt_health.py matmul`) in `nodes/<host>.json`; DCGM `diagnostic` gives pass/fail only | srygaard/slurm-healthcheck | per-GPU baseline in the window (Figs. 1, 3) | small (their tool) |
| 6c | run both `--job-per switch` and `--job-per cluster` in the window (one leaf and whole machine); optionally `--switches A B --job-per cluster` for 2-leaf scopes (e.g. 217+316) | window plan | communication scopes (Fig. 4) | none (operational) |
| 3b | family **O** (1 node per switch, one node on each of N leaves) in the node picker | `tt_pick_nodes.py` | nodes-per-switch figure (S / H / O at 3, 6, 9, 12 nodes; O up to 16) | small |
| 7b | GLM-5 expert-group layouts on Rorqual: EP8 aligned / mixed (~1 group in 4 crossing) / crossed on the **same 52 nodes** (even number per leaf: 13 x 2 + 12 + 8 + 6; fall back to 50 / 48), only the node order differs; plus mixed on every healthy node (largest achievable, Megatron's default order, even count); full model if it fits on 208 GPUs (to confirm); record `placement.ep_groups`; per-group all-to-all time if Megatron's MoE timers allow. Picker: "even per leaf" selection + the three orders | GLM-5 pipeline, `tt_pick_nodes.py` | expert-placement figure | medium |
| 7 | same events from Megatron (voirfile or converter) | GLM-5 pipeline | MoE section | medium |
| 8 | DCGM export (fields, interval, access) | admins / whoever runs Prometheus | stability, stragglers | to discuss |

Not requested anymore: per-rank compute vs. NCCL time (DCGM tensor activity covers
stragglers), fixing voir's empty `gpudata` (DCGM covers it; still a bug).
