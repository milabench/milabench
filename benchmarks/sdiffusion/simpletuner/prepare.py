#!/usr/bin/env python
"""Clone SimpleTuner and materialize the shared fine-tuning dataset."""

import os
from dataclasses import dataclass
from pathlib import Path

from benchmate import diffusion


@dataclass
class PrepareConfig:
    cache: str = None
    rev: str = None            # git tag/branch/commit to pin SimpleTuner
    dataset: str = "naruto-blip-captions"
    dataset_id: str = diffusion.DEFAULT_DATASET
    train_images: int = 1000
    val_images: int = 32
    install_deps: bool = True  # pip install the repo's own requirements files


def candidate_requirements(repo: Path):
    # Layout changed over time: root requirements.txt (older) or
    # requirements/{cuda,torch,sdxl}.txt (newer split files).
    for name in ["requirements.txt"]:
        yield repo / name
    for name in ["cuda.txt", "torch.txt", "sdxl.txt"]:
        yield repo / "requirements" / name


def main():
    from argklass import ArgumentParser

    parser = ArgumentParser()
    parser.add_arguments(PrepareConfig)
    args, _ = parser.parse_known_args()

    if args.cache:
        os.environ.setdefault("XDG_CACHE_HOME", str(args.cache))

    repo = diffusion.clone_repo("simpletuner", rev=args.rev, cache=args.cache)

    if args.install_deps:
        for req in candidate_requirements(repo):
            if req.exists():
                diffusion.install_requirements(req)

    diffusion.materialize_dataset(
        diffusion.dataset_dir(args.dataset, args.cache),
        dataset_id=args.dataset_id,
        train_count=args.train_images,
        val_count=args.val_images,
    )


if __name__ == "__main__":
    main()
