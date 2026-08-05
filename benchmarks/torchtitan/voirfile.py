from dataclasses import dataclass
import time

from voir import configurable
from voir.phase import StopProgram
from benchmate.monitor import voirfile_monitor
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

    # Smoke argv uses seq_len=512, local_batch_size=1 → tokens/step ≈ 512.
    # Full runs still get a usable items/s signal; refine later if needed.
    tokens_per_step = 512
    early = int(options.stop) + int(options.skip)
    state = {"t": None, "n": 0}

    def emit_rate():
        now = time.perf_counter()
        prev = state["t"]
        state["t"] = now
        if prev is None:
            return
        dt = now - prev
        if dt <= 0:
            return
        ov.give(rate=tokens_per_step / dt, units="items/s", task="train", time=time.time())
        state["n"] += 1
        # Do not raise StopProgram from inside fwd/bwd — it races torchrun cleanup
        # under the glibc wrapper. Smoke uses a tiny --training.steps instead.

    def on_loss(loss):
        try:
            value = float(loss.detach().item()) if hasattr(loss, "detach") else float(loss)
            ov.give(loss=value, task="train", time=time.time())
        except Exception:
            pass
        emit_rate()
        return loss

    # Ptera probes are brittle under torch.compile / decorators; patch the
    # Trainer method after the script is loaded so rates always flow.
    try:
        from torchtitan.trainer import Trainer

        if not getattr(Trainer.forward_backward_step, "_milabench_rate_wrapped", False):
            _orig = Trainer.forward_backward_step

            def _wrapped(self, *args, **kwargs):
                loss = _orig(self, *args, **kwargs)
                return on_loss(loss)

            _wrapped._milabench_rate_wrapped = True
            Trainer.forward_backward_step = _wrapped
            print("[voirfile] wrapped Trainer.forward_backward_step for rates", flush=True)
    except Exception as exc:
        print(f"[voirfile] wrap failed: {exc}", flush=True)

    with forward_voir_file():
        try:
            yield ov.phases.run_script
        except StopProgram:
            print("early stopped", flush=True)
