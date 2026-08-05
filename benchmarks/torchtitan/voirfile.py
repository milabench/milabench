from dataclasses import dataclass

from voir import configurable
from voir.phase import StopProgram
from benchmate.monitor import voirfile_monitor
from benchmate.observer import BenchObserver
from benchmate.benchrun import forward_voir_file


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

    # tokens ≈ local_batch * seq_len; use 1 so rate ≈ steps/s until we refine.
    observer = BenchObserver(
        earlystop=options.stop + options.skip,
        batch_size_fn=lambda _batch: 1,
        raise_stop_program=True,
    )

    def on_loss(loss):
        try:
            value = float(loss.detach().item()) if hasattr(loss, "detach") else float(loss)
        except Exception:
            value = float(loss)
        observer.record_loss(value)
        observer.step()
        return loss

    # Probe the HF / core trainer step loss when available.
    for probe_path in (
        "//HFTransformerTrainer/forward_backward_step > loss",
        "//Trainer/forward_backward_step > loss",
        "//HFTransformerTrainer/train_step > loss",
        "//Trainer/train_step > loss",
    ):
        try:
            probe = ov.probe(probe_path, overridable=True)
            probe["loss"].override(on_loss)
            break
        except Exception:
            continue

    with forward_voir_file():
        try:
            yield ov.phases.run_script
        except StopProgram:
            print("early stopped")
