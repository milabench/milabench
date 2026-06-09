## Overview

## Entry points

- `benchfile.py`: `Recursiongfn` package. Clones `gflownet` repo into `benchmarks/recursiongfn/gflownet/` from `github.com/Delaunay/gflownet` (branch `milabench`). Sets `FORCE_CUDA=1`.
- `prepare.py`: Downloads the pretrained `bengio2021flow_proxy.pkl.gz` proxy model to `MILABENCH_DIR_DATA`.
- `main.py`: Training entry point. Monkey-patches `SEHFragTrainer` and `SEHTask` from the gflownet library.
- `voirfile.py`: Standard `voirfile_monitor` -- skips 5, logs 20, GPU poll 1s.

## Code flow

1. `main()` creates a `gflownet.config.Config` with GFlowNet hyperparameters, then instantiates `SEHFragTrainerMonkeyPatch` which wraps the upstream `SEHFragTrainer`.
2. The monkey-patch does three things:
   - Wraps the training dataloader with `BenchObserver.loader()` for throughput measurement.
   - Intercepts `_maybe_resolve_shared_buffer()` to count total nodes across all graphs in each batch (batch size = sum of node counts).
   - Intercepts `step()` to call `observer.record_loss()`.
3. `SEHTaskMonkeyPatch` overrides `_load_task_models()` to load the proxy model from `MILABENCH_DIR_DATA` instead of the default download location.
4. The trainer's `run()` method handles the GFlowNet training loop: sample trajectories from policy, compute rewards via frozen proxy, compute trajectory balance loss, update policy.

## Batch size measurement

Batch size is measured as total number of **nodes** across all molecular graphs in the batch, not the number of molecules. This is accumulated in `_maybe_resolve_shared_buffer()` via `elem.x.shape[0]`.

## Key config values

- `sampling_tau=0.9`: temperature for off-policy sampling
- `clip_grad_type="total_norm"`: gradient clipping by total norm
- `lr_decay=20000`: LR decay schedule
- `mp_buffer_size=32MB`: shared memory buffer for multiprocessing
- `replay.use=False`: no experience replay (purely on-policy)
- `validate_every=0`, `num_final_gen_steps=0`: no validation or final generation

## Known gotchas

- `sys.path.append` is used to import from the cloned gflownet repo -- fragile if the repo structure changes.
- `BenchObserver` is created with `earlystop=65` and `raise_stop_program=False`, so the observer signals stop but the trainer continues until `num_training_steps` is reached.
- The batch size function uses a list as a stack (`self.batch_size_in_nodes`). If `_maybe_resolve_shared_buffer` isn't called before the observer reads batch size, it returns 0.
- The proxy model is on GPU (`get_worker_device()`) and wrapped for multi-processing via `_wrap_for_mp`.
- Heavy dependency list: includes `pyro-ppl`, `botorch`, `gpytorch`, `wandb`, `tensorboard` even though most are unused in the benchmark itself.
- `checkpoint_every=5` means checkpoint writes happen frequently, but `_save_checkpoint` is not overridden, so checkpoints do get written.

## Computational profile

| Aspect | Detail |
|--------|--------|
| Bottleneck | GFlowNet rollout generation (sequential graph construction) |
| VRAM | ~12 GB @ batch 112 |
| Typical throughput | ~12k nodes/s (MI325), ~10k nodes/s (L40S) |
| GPU utilization | Peaks ~25%, CPU-bound on trajectory sampling |

## Key dependencies

`gflownet` (vendored clone), `torch-geometric`, `rdkit`, `pyro-ppl`, `botorch`, `gpytorch`
