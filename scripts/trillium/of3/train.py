"""`run_openfold train` (OpenFold3's own CLI, same arguments) with one JSON line per training step and rank in
$OF3_STEPLOG (a path with {rank}): the step's wall time from batch start to the end of the optimizer step (`step_s`),
the time from the previous step's end to this step's start (`gap_s`: waiting for the batch, plus the EMA update of the
previous step), the recycles drawn, the gradient sync (`sync_s`: grad_manager.sync_and_average_grads, from its call
with the GPU idle to its end: waiting for the slowest rank, then the all-reduces; the fastest rank's is the all-reduce
itself), the loss, the sample's size and the peak allocated and reserved GPU memory so far. OpenFold3 only logs to
wandb, every log_every_n_steps. With OpenFold3's profiler on (variant prof), only each node's local rank 0 traces (a
trace is about 1.4 GB per rank and 2 steps)."""
import json
import os
import time
from functools import cached_property

import pytorch_lightning as pl
import torch
from openfold3.entry_points import experiment_runner
from openfold3.run_openfold import cli


class StepLog(pl.Callback):
    def __init__(self, path):
        self.path = path
        self.start = self.end = None
        self.sync = 0.0

    def on_train_start(self, trainer, pl_module):
        self.file = open(self.path.format(rank=trainer.global_rank), "a", buffering=1)
        params = sum(p.numel() for p in pl_module.parameters() if p.requires_grad)
        self.file.write(json.dumps({"event": "start", "t": time.time(), "params": params,
                                    "world_size": trainer.world_size, "host": os.uname().nodename}) + "\n")
        pl_module.model.register_forward_hook(lambda module, args, out: setattr(self, "recycles", out[1]["recycles"]))
        sync = pl_module.grad_manager.sync_and_average_grads

        def timed_sync():
            torch.cuda.synchronize()
            start = time.perf_counter()
            sync()
            torch.cuda.synchronize()
            self.sync = time.perf_counter() - start

        pl_module.grad_manager.sync_and_average_grads = timed_sync

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.start = time.perf_counter()
        self.gap = None if self.end is None else self.start - self.end
        self.batch = {"pdb_id": batch["pdb_id"], "tokens": int(batch["token_mask"].sum()),
                      "atoms": int(batch["atom_mask"].sum()), "msa_shape": list(batch["msa"].shape)}

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        torch.cuda.synchronize()
        self.end = time.perf_counter()
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs
        self.file.write(json.dumps({
            "step": trainer.global_step, "t": time.time(), "step_s": self.end - self.start, "gap_s": self.gap,
            "sync_s": self.sync, "recycles": self.recycles, "loss": None if loss is None else float(loss), **self.batch,
            "alloc_peak_gib": torch.cuda.max_memory_allocated() / 2**30,
            "reserved_peak_gib": torch.cuda.max_memory_reserved() / 2**30}) + "\n")


callbacks = experiment_runner.TrainingExperimentRunner.callbacks.func
prop = cached_property(lambda self: callbacks(self) + [StepLog(os.environ["OF3_STEPLOG"])])
prop.__set_name__(experiment_runner.TrainingExperimentRunner, "callbacks")
experiment_runner.TrainingExperimentRunner.callbacks = prop
build = experiment_runner.ExperimentRunner._build_profiler
experiment_runner.ExperimentRunner._build_profiler = lambda self: build(self) if os.environ["SLURM_LOCALID"] == "0" else None

if __name__ == "__main__":
    cli()
