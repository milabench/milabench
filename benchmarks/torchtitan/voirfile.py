"""voir instrumentation for the native TorchTitan benchmark.

Same structure as the other milabench voirfiles: TorchTitan's Trainer is
wrapped (see tt_native/instrument.py), benchmate times the steps and pushes
one ``rate`` (global tokens/s) per measured step, plus ``loss`` and a final
``summary`` event. The official throughput is the median of the rates.

Options (milabench ``voir.options``):
    skip          warm-up steps, not timed
    stop          measured steps
    stop_seconds  if > 0, measure for this long instead of ``stop`` steps
"""

from dataclasses import dataclass, replace

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
    skip: int = 40
    stop: int = 100
    stop_seconds: float = 0.0


@configurable
def instrument_main(ov, options: Config):
    yield ov.phases.init

    # The stop is applied on every rank by the instrumented trainer at the
    # same step. voir's own early stop acts on rank 0 only, which would leave
    # the other ranks blocked in collectives: keep it out of reach.
    voirfile_monitor(ov, replace(options, stop=10**9))

    yield ov.phases.load_script

    from tt_native.instrument import TrainerInstrument, install

    install(
        TrainerInstrument(
            skip=options.skip,
            stop=options.stop,
            stop_seconds=options.stop_seconds or None,
            push=sumggle_push(),
        )
    )

    with forward_voir_file():
        try:
            yield ov.phases.run_script
        except StopProgram:
            print("early stopped", flush=True)
