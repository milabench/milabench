"""benchmate's TimedIterator on Megatron's training loop.

Megatron owns the loop, so the timer is attached from two sides:
  - the train dataloader is wrapped (``wrap_loader``): data wait, laps and the CUDA-event clock come for free;
  - one ``step(tokens)`` call per optimizer step closes the measurement (``rate``).
"""
import os
import time

from benchmate.metrics import ManualTimedIterator, default_device, default_event
from benchmate.monitor import get_rank


class LocalBatchTimer(ManualTimedIterator):
    """No all-reduce of the batch size: Megatron builds the train loader on some ranks only (first / last pipeline stage),
    a collective here would wait forever on the others. Rank 0 always has a loader and reports the global tokens."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # normally set by __iter__ / wrapped(), which only run if Megatron iterates the loader before the first step
        self.last_time = time.time()
        self.start = None

    def batch_size(self, bs):
        return bs


class SlowLoader:
    """Fake dataloading stall, to see what a starved GPU looks like in the timers: ``BENCHMATE_FAKE_DATA_LATENCY``
    seconds of sleep before every ``BENCHMATE_FAKE_DATA_EVERY``-th batch (default 5), 0 = off."""

    def __init__(self, loader, latency, every):
        self.loader, self.latency, self.every = loader, latency, max(every, 1)

    def __iter__(self):
        for i, batch in enumerate(self.loader):
            if i % self.every == self.every - 1:
                time.sleep(self.latency)
            yield batch

    def __len__(self):
        return len(self.loader)

    def __getattr__(self, item):
        return getattr(self.loader, item)


class StepTimer:
    """``push_every``: flush the queued step events every N steps (each flush waits for the GPU) and at ``flush()``."""

    def __init__(self, log, push_every=None):
        self.log = log
        self.timer = None
        self.push_every = max(int(push_every or os.getenv("BENCHMATE_PUSH_EVERY", 10)), 1)
        self.steps = 0

    def _push(self, **event):
        event.setdefault("task", "train")
        self.log({k: v for k, v in event.items() if v is not None})

    def wrap_loader(self, loader):
        """The wrapped loader is a drop-in replacement: Megatron only iterates it."""
        latency = float(os.getenv("BENCHMATE_FAKE_DATA_LATENCY", 0))
        if latency > 0:  # inside the timer: the stall is part of ``next``
            loader = SlowLoader(loader, latency, int(os.getenv("BENCHMATE_FAKE_DATA_EVERY", 5)))
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
            if self.timer.start is None:  # the loader was never iterated: the clock starts now
                self.timer.start = self.timer.event_fn(enable_timing=True)
                self.timer.start.record()
            self.timer.step(batch_override=global_tokens)
            self.steps += 1
            if self.steps % self.push_every == 0:
                self.timer._push()

    def flush(self):
        """Push what is still queued (end of the run: the loader's own end-of-iteration flush never runs under Megatron)."""
        if self.timer is not None and self.timer.events:
            self.timer._push()

    def restart_clock(self):
        """Restart the step clock now: whatever ran since the last ``step`` (e.g. a metrics report) is not counted."""
        if self.timer is not None and self.timer.start is not None:
            self.timer.start = self.timer.event_fn(enable_timing=True)
            self.timer.start.record()


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
