from dataclasses import dataclass

from voir import configurable
from voir.phase import StopProgram
from benchmate.metrics import (
    LazyLossPusher,
    ManualTimedIterator,
    default_device,
    default_event,
    sumggle_push,
)
from benchmate.monitor import get_rank, voirfile_monitor
from benchmate.benchrun import forward_voir_file

from torch_compat import ensure_torchtitan_torch_compat


def _milabench_rank() -> int:
    rank = get_rank()
    return 0 if rank < 0 else rank


class _TrainStepMetrics:
    """CUDA-event train throughput via ManualTimedIterator (openinstruct pattern)."""

    def __init__(self) -> None:
        self.timer = ManualTimedIterator(
            iter(()),
            event_fn=default_event(),
            rank=_milabench_rank(),
            push=sumggle_push(),
            device=default_device(),
            earlystop=10**9,
            raise_stop_program=False,
        )
        self.losses = LazyLossPusher("train")
        self._started = False
        self._state: dict = {}

    def _ensure_started(self) -> None:
        if self._started:
            return
        self.timer.start = self.timer.event_fn(enable_timing=True)
        self.timer.start.record()
        self._started = True

    def reset_step(self) -> None:
        self._state = {"local_tokens": 0, "loss": None}

    def note_local_tokens(self, count: int) -> None:
        self._state["local_tokens"] = int(count)

    def note_loss(self, loss) -> None:
        self._state["loss"] = loss

    def finish_step(self) -> None:
        tokens = self._state.get("local_tokens", 0)
        if tokens > 0:
            self._ensure_started()
            self.timer.step(batch_override=tokens)
            self.timer._push()

        loss = self._state.get("loss")
        if loss is not None:
            self.losses.record(loss)
            self.losses.push(self.timer.message_push)


_metrics: _TrainStepMetrics | None = None


def _install_torchtitan_metrics() -> None:
    global _metrics
    if _metrics is not None:
        return

    ensure_torchtitan_torch_compat()
    from torchtitan.trainer import Trainer

    _metrics = _TrainStepMetrics()

    if not getattr(Trainer.train_step, "_milabench_timed", False):
        _orig_train_step = Trainer.train_step

        def _wrapped_train_step(self, data_iterator):
            _metrics.reset_step()
            _orig_train_step(self, data_iterator)
            _metrics.finish_step()

        _wrapped_train_step._milabench_timed = True  # type: ignore[attr-defined]
        Trainer.train_step = _wrapped_train_step

    if not getattr(Trainer.forward_backward_step, "_milabench_loss", False):
        _orig_fwb = Trainer.forward_backward_step

        def _wrapped_fwb(self, *args, **kwargs):
            loss = _orig_fwb(self, *args, **kwargs)
            _metrics.note_loss(loss)
            return loss

        _wrapped_fwb._milabench_loss = True  # type: ignore[attr-defined]
        Trainer.forward_backward_step = _wrapped_fwb

    try:
        from torchtitan.observability.structured_logger import structured_logging as sl

        if not getattr(sl.log_trace_scalar, "_milabench_tokens", False):
            _orig_log_scalar = sl.log_trace_scalar

            def _wrapped_log_trace_scalar(scalars, *args, **kwargs):
                if local := scalars.get("local_valid_tokens"):
                    _metrics.note_local_tokens(int(local))
                return _orig_log_scalar(scalars, *args, **kwargs)

            _wrapped_log_trace_scalar._milabench_tokens = True  # type: ignore[attr-defined]
            sl.log_trace_scalar = _wrapped_log_trace_scalar
    except ImportError:
        pass

    print("[voirfile] ManualTimedIterator metrics installed on Trainer.train_step", flush=True)


@dataclass
class Config:
    """voir configuration"""

    dash: bool = False
    interval: str = "1s"
    skip: int = 5
    stop: int = 20
    gpu_poll: float = 1


@configurable
def instrument_main(ov, options: Config):
    yield ov.phases.init

    voirfile_monitor(ov, options)

    yield ov.phases.load_script

    _install_torchtitan_metrics()

    with forward_voir_file():
        try:
            yield ov.phases.run_script
        except StopProgram:
            print("early stopped", flush=True)
