"""Thin entrypoint so milabench can wrap torchtitan with voir."""

from torch_compat import ensure_torchtitan_torch_compat

ensure_torchtitan_torch_compat()

from torchtitan.train import main


if __name__ == "__main__":
    main()
