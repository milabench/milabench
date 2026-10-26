"""Thin entrypoint so milabench can wrap torchtitan with voir."""

import sys

from torch_compat import ensure_torchtitan_torch_compat

ensure_torchtitan_torch_compat()

from torchtitan.train import main


def _restore_config_flag(argv):
    """voir swallows every ``--config`` in argv (its own yaml option), so the
    torchtitan config name is passed as ``--tt-config`` and renamed here."""
    out = []
    for a in argv:
        if a == "--tt-config":
            a = "--config"
        elif a.startswith("--tt-config="):
            a = "--config=" + a.split("=", 1)[1]
        out.append(a)
    return out


if __name__ == "__main__":
    sys.argv[:] = _restore_config_flag(sys.argv)
    main()
