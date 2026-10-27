"""Entrypoint for torchtitan: instruments the trainer and reports metrics on stdout."""

import os

from benchmate.monitor import get_rank, setupvoir

from torch_compat import ensure_torchtitan_torch_compat

ensure_torchtitan_torch_compat()

from torchtitan.train import main

from tt_metrics import install_torchtitan_metrics


if __name__ == "__main__":
    install_torchtitan_metrics()

    monitor = None
    if get_rank() in (-1, 0):
        monogpu = os.environ.get("WORLD_SIZE", "1") == "1"
        _, monitor = setupvoir(monogpu=monogpu, interval=1)

    try:
        main()
    finally:
        if monitor is not None:
            monitor.stop()
