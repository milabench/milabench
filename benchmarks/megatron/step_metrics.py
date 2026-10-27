"""benchmate's TimedIterator on Megatron's training loop.

Megatron owns the loop, so the timer is attached from two sides:
  - the train dataloader is wrapped (``wrap_loader``): data wait, laps and the CUDA-event clock come for free;
  - one ``step(tokens)`` call per optimizer step closes the measurement (``rate``).
"""
import time

from benchmate.metrics import ManualTimedIterator, default_device, default_event
from benchmate.monitor import get_rank


class LocalBatchTimer(ManualTimedIterator):
    """No all-reduce of the batch size: Megatron builds the train loader on some ranks only (first / last pipeline stage),
    a collective here would wait forever on the others. Rank 0 always has a loader and reports the global tokens."""

    def batch_size(self, bs):
        return bs


class StepTimer:
    def __init__(self, log):
        self.log = log
        self.timer = None

    def _push(self, **event):
        event.setdefault("task", "train")
        self.log({k: v for k, v in event.items() if v is not None})

    def wrap_loader(self, loader):
        """The wrapped loader is a drop-in replacement: Megatron only iterates it."""
        self.timer = LocalBatchTimer(
            loader,
            event_fn=default_event(),
            rank=max(get_rank(), 0),
            push=self._push,
            device=default_device(),
            earlystop=10**9,
            raise_stop_program=False,
            batch_size_fn=lambda microbatch: 0,  # counted per step instead (see step)
        )
        return self.timer

    def step(self, global_tokens):
        """Call on every rank after each optimizer step (ranks without a train loader have no timer)."""
        if self.timer is not None:
            self.timer.step(batch_override=global_tokens)
            self.timer._push()


def install(log):
    """Wrap the train dataloader Megatron builds and return the StepTimer."""
    import megatron.training.training as training

    steps = StepTimer(log)
    build = training.build_train_valid_test_data_loaders

    def build_wrapped(*args, **kwargs):
        train, valid, test = build(*args, **kwargs)
        if train is not None:
            train = steps.wrap_loader(train)
        return train, valid, test

    training.build_train_valid_test_data_loaders = build_wrapped
    return steps
