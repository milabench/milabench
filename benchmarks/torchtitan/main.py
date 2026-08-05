"""Thin entrypoint so milabench can wrap torchtitan with voir."""

from torchtitan.train import main


if __name__ == "__main__":
    main()
