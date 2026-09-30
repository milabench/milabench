#!/usr/bin/env python
"""Clone SimpleTuner and materialize the shared fine-tuning dataset."""

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from benchmate import diffusion


@dataclass
class PrepareConfig:
    cache: str = None
    dataset: str = "naruto-blip-captions"
    dataset_id: str = diffusion.DEFAULT_DATASET
    train_images: int = 1000
    val_images: int = 32
    install_deps: bool = True  # pip install the repo (platform=cpu deps)


def install_repo(repo: Path):
    # The 1.x deps come from the package metadata; SIMPLETUNER_PLATFORM=cpu
    # avoids the CUDA/ROCm torch pins so the XPU torch in the venv is kept.
    uv = shutil.which("uv")
    if uv:
        cmd = [uv, "pip", "install", "--python", sys.executable, str(repo)]
    else:
        cmd = [sys.executable, "-m", "pip", "install", str(repo)]
    subprocess.run(cmd, check=True, env=dict(os.environ, SIMPLETUNER_PLATFORM="cpu"))


def main():
    from argklass import ArgumentParser

    parser = ArgumentParser()
    parser.add_arguments(PrepareConfig)
    args, _ = parser.parse_known_args()

    if args.cache:
        os.environ.setdefault("XDG_CACHE_HOME", str(args.cache))

    repo = diffusion.clone_repo("simpletuner", cache=args.cache)

    if args.install_deps:
        install_repo(repo)

    diffusion.materialize_dataset(
        diffusion.dataset_dir(args.dataset, args.cache),
        dataset_id=args.dataset_id,
        train_count=args.train_images,
        val_count=args.val_images,
    )


if __name__ == "__main__":
    main()
