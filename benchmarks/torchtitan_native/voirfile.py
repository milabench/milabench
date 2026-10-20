"""Report benchmark throughput to milabench (rank 0 of the main node only).

The authoritative numbers are in the JSON results record written by
``tt_native.trainer``; this only feeds milabench's live metrics/report.
"""

from dataclasses import dataclass

from voir import configurable
from voir.phase import StopProgram
from benchmate.benchrun import forward_voir_file
from benchmate.metrics import sumggle_push
from benchmate.monitor import voirfile_monitor


@dataclass
class Config:
    """voir configuration"""

    dash: bool = False
    interval: str = "1s"
    gpu_poll: float = 1.0
    # The bench trainer decides when to stop (warm-up + measurement window);
    # voir's early stop must never cut a run short.
    stop: int = 10**9


def _install_step_hook() -> None:
    import tt_native.trainer as bench_trainer

    push = sumggle_push()

    def on_step(step, step_time, tokens, loss):
        # Global rank 0 only: rank 0 of every node runs this voirfile, and
        # their stdout lines interleave (garbled JSON for milabench).
        import torch.distributed as dist

        if dist.is_initialized() and dist.get_rank() != 0:
            return
        push(rate=tokens / step_time, units="tokens/s", task="train")
        if loss is not None:
            push(loss=float(loss), task="train")

    bench_trainer.STEP_HOOK = on_step


@configurable
def instrument_main(ov, options: Config):
    yield ov.phases.init

    voirfile_monitor(ov, options)

    yield ov.phases.load_script

    _install_step_hook()

    with forward_voir_file():
        try:
            yield ov.phases.run_script
        except StopProgram:
            print("early stopped", flush=True)
