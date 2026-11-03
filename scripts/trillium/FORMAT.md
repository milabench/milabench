# Result formats: Trillium additions

Prescriptive spec of what the Trillium campaign adds to `scripts/analysis/FORMAT.md` (the Rorqual
formats: run data `milabench.pretrain.v1`, srygaard health checks, nodemon telemetry, per-rank
metrics). Everything there still holds; this file lists only the additions, so that
`analysis/analyze.py` turns the raw data into the report with no manual step. Status: **draft**
(2026-11-02).

Worked examples, written by `analysis/analyze.py synth` under `analysis/data/synthetic/trillium/`:
`runs/syn-aq-hsdp8-24k-n13-r1/tt-qwen.0.data`, `runs/syn-f1-glm-full-async-n52-r1/glm5.0.data`,
`health/3000001/`, `nccl/3000002/pairs.jsonl`, `nccl/3000003/collectives.jsonl`,
`nodemon/syn-f2-long/`.

Legend: **add** = the benchmarks or launchers must add it; **exists** = already produced.

## 0. Layout on Trillium

Everything under `$SCRATCH/trillium/`, copied read-only to `analysis/data/raw/trillium/`:

| directory | content | written by |
|---|---|---|
| `runs/<run_name>/<bench>.<n>.data` | milabench run data (section 1) | milabench, both workloads |
| `runs/<run_name>/metrics.*.jsonl` | per-rank step metrics (Rorqual FORMAT.md section 4) | benchmate rankstore |
| `health/<job>/` | srygaard results directory (section 2) | D1 |
| `nccl/<job>/pairs.jsonl`, `collectives.jsonl` (+ raw logs beside them) | D2, D3, D4 (section 3) | sweep scripts |
| `nodemon/<run_name>/` | node telemetry (Rorqual FORMAT.md section 3), **every run** | benchmate.nodemon |
| `checkpoints/` | the checkpoints themselves; only `*.json`/`*.jsonl` are ever copied | F1 |

The run name is the join key everywhere (`nodemon/<run_name>` = `runs/<run_name>`).

## 1. Run data additions

The `run_start` event and the `summary` (`milabench.pretrain.v1`, additive: same version) carry:

```jsonc
"run": {
  "experiment": "A-Q",          // add: DESIGN.md ID: D1..D4 | E1 | R | P | A-Q | A-G | B-Q-weak | B-Q-strong
                                //      | B-G-weak | B-G-strong | C-speed | C-loss | F1 | F2 | smoke
  "node_set": "d1-s3",          // add, E1 only: <draw>-<set> (4 disjoint sets of 13 nodes per draw)
  "condition": "alone",         // add, E1 only: alone | together
  "exclusive_cluster": true     // exists
},
"parallelism": {
  "layout": "hsdp8",            // add: launcher's name. Qwen: fsdp | hsdp8 | hsdp16 | tp2-fsdp | tp4-fsdp
                                //      | pp2-fsdp | pp4-fsdp. GLM-5: pp13-ep8 | pp13-ep16 | pp13-ep32 | pp26-ep8 | pp6-ep16
  "gpus": 104, "nodes": 13,     // exists
  "dp_replicate": 13, "dp_shard": 8, "tp": 1, "pp": 1, "cp": 1, "ep": 1,   // exist; all always written
  "pp_schedule": "1F1B",        // add: null when pp = 1 (no interleaving on GLM-5: it needs micro-batches >= PP)
  "expert_dp": 1,               // add, MoE: world / (EP x TP x PP)
  "copies": 13,                 // add: disjoint GPU groups that each hold the whole model state
                                //      (Qwen HSDP: dp_replicate; FSDP: 1; GLM-5: gpus / (PP x EP))
  "microbatches_per_pipeline": 32   // add, PP > 1: micro-batches per step per pipeline (sets the bubble)
},
"workload": {
  "precision": "mxfp8",         // extended: bf16 | fp8-tensorwise | fp8-rowwise | fp8-blockwise | mxfp8
                                //   (Qwen/TorchTitan: bf16 | mxfp8; GLM-5/Megatron-TE: bf16 | fp8-tensorwise
                                //    | fp8-blockwise | mxfp8)
  "grad_reduce_dtype": "fp32",  // add: fp32 | bf16 (dtype of the gradient reduction across ranks)
  "activation_checkpointing": "full",  // exists; values fixed: none | selective | full
  "params_total": 26.9e9, "params_active": 26.9e9,  // exist
  "seed": 1234                  // add: C-loss compares recipes from the same seed and data order
},
"mfu_inputs": {"peak_flops_per_gpu": 2.25e15}   // B200 dense BF16 spec, for every recipe (MFU in BF16 units)
```

`microbatch_sequences`, `grad_accum`, `tokens_per_gpu_per_step` and `global_batch_tokens` keep their
Rorqual meaning; with gradient accumulation `tokens_per_gpu_per_step` includes it.

### Loss events (add for GLM-5; exists for Qwen)

One event per logged step, **every step** in C-loss runs (500 steps, `measurement.warmup_steps: 0`):

```jsonc
{"loss": 7.412, "batch_id": 31, "task": "train", "time": 1795...}   // TorchTitan: batch_id
{"loss": 7.415, "step": 31, "task": "train", "time": 1795...}       // Megatron: step
```

`batch_id` / `step` is the same index as the `rate` events' `batch_id`; the loss is the global
mean over data-parallel ranks (rank 0, after the reduction). The analysis joins on it.

### Checkpoint events (add, F1)

One event per operation, `data: {checkpoint: {...}, task: "train"}`, same shape for both workloads:

```jsonc
{"op": "save",               // save | load
 "step": 40, "repetition": 1,
 "mode": "async",            // sync | async | async_pinned (load: sync)
 "content": "full",          // full (model + optimizer + scheduler + data loader) | weights
 "format": "torch.dcp",      // torch.dcp (TorchTitan) | torch_dist (Megatron)
 "bytes": 3.8e11, "files": 832,
 "blocked_s": 18.4,          // training blocked: call -> control back to the training loop (max over ranks)
 "total_s": 81.2,            // until the last byte is written (async: background completion) / all ranks loaded
 "path": "/scratch/<user>/trillium/checkpoints/<run>", "filesystem": "vast-nfs",
 "start_epoch": 1795..., "end_epoch": 1795...}
```

Times are the maximum over ranks; `{"min": .., "max": ..}` is also accepted (the analysis keeps
`max`, and `*_min`). Throughput is derived (`bytes / total_s`), not logged. The TorchTitan pack's
own names are mapped on read (`kind: model` -> `content: weights`, `async_mode: disabled |
async_with_pinned_mem` -> `mode: sync | async_pinned`, `fs_type` -> `filesystem`, `rep` ->
`repetition`; `analysis/tri_analysis/runs.py`), but new code should write the names above. The
nodes are diskless: no local staging.

## 2. D1 health check: srygaard additions

The srygaard results directory is unchanged (`report.json`, `groups/<job>.json`, `nodes/<host>.json`);
NVLink all-reduce is its existing `intra` group test. **Add** two checks to `nodes/<host>.json`:

```jsonc
"matmul": {"status": "ok", "data": {"gpus": [
  {"gpu": 0, "uuid": "GPU-...", "size": 8192, "bf16_tflops": 1668.2, "fp8_tflops": 3041.0}]}},
"hca": {"status": "ok", "data": {"hcas": [
  {"hca": "mlx5_0", "port": 1, "rate_gbps": 200, "loopback_gbs": 23.6,
   "numa": 0, "gpus": [0, 1]}]}}       // gpus: GPUs whose nearest HCA this is (GPU-HCA affinity)
```

## 3. D2-D4 network sweeps

Raw tool output (nccl-tests stdout, ib_* logs) stays beside the JSONL in `nccl/<job>/logs/`; the
analysis reads only the JSONL, one object per line, **written by the sweep script** (add).

### D2: `nccl/<job>/pairs.jsonl` (`trillium.pairs.v1`), one line per node pair

```jsonc
{"schema": "trillium.pairs.v1", "job_id": "3000002", "round": 3, "a": "trig0101", "b": "trig0117",
 "tool": "ib_write_bw/ib_write_lat, all HCAs", "size_bytes": 1073741824,
 "bw_gbs": 90.4,            // one direction, node to node, all HCAs at once, GB/s
 "lat_us": 2.1, "lat_size_bytes": 8,   // one-way latency
 "time": 1795..., "status": "ok"}      // ok | timeout | error (bw_gbs, lat_us null)
```

### D3, D4: `nccl/<job>/collectives.jsonl` (`trillium.collectives.v1`)

D3: one line per (collective, nodes, message size, repetition), nccl-tests out-of-place numbers:

```jsonc
{"schema": "trillium.collectives.v1", "experiment": "D3", "job_id": "3000003",
 "collective": "all_reduce",          // all_reduce | all_gather | reduce_scatter | alltoall
 "nodes": 13, "gpus": 104, "hosts": ["trig0101", "..."],
 "size_bytes": 1073741824, "time_us": 25310.4, "algbw_gbs": 42.4, "busbw_gbs": 84.0,
 "cut": null, "repetition": 1, "time": 1795..., "nccl_version": "2.27.7", "status": "ok"}
```

D4 (bisection): one line per cut; the two halves exchange in disjoint random pairs, all at once:

```jsonc
{"schema": "trillium.collectives.v1", "experiment": "D4", "collective": "bisection", "cut": 1,
 "nodes": 52, "gpus": 416, "hosts": ["..."], "halves": [["trig0101", "..."], ["..."]],
 "pairs": [{"a": "trig0101", "b": "trig0133", "bw_gbs": 89.1}],
 "size_bytes": 1073741824, "busbw_gbs": 88.7,    // median over pairs, per node, one direction
 "algbw_gbs": null, "time_us": null, "repetition": 1, "time": 1795..., "status": "ok"}
```

## 4. Telemetry

Unchanged format (Rorqual FORMAT.md section 3), but **every training run** sets
`BENCHMATE_NODEMON_DIR=$SCRATCH/trillium/nodemon/<run_name>`: the energy figure (F4) integrates
`gpu.power_w` over each run's measurement window. On Blackwell the `gpm` collector should be on.

## Changes requested (to route)

| # | change | where | for |
|---|---|---|---|
| T1 | `run.experiment`, `node_set`, `condition`; `parallelism.layout`, `pp_schedule`, `expert_dp`, `copies`, `microbatches_per_pipeline`; `workload.grad_reduce_dtype`, `seed`; the extended `precision` values | launchers, both summaries | every figure |
| T2 | loss events with `step` from Megatron | GLM-5 pack | `precision_loss` |
| T3 | checkpoint events in the shape above | both packs | `checkpoint` |
| T4 | `matmul` and `hca` checks in srygaard `nodes/<host>.json` | slurm-healthcheck | `health_gpus`, `health_links` |
| T5 | pair and collective sweep scripts writing the two JSONL files | `scripts/trillium/` | `pair_matrix`, `collectives_scale` |
| T6 | nodemon on for every run | launchers | `energy`, `stability`, `stragglers` |
