#!/usr/bin/env python
"""Clone kohya-ss/sd-scripts and materialize the shared fine-tuning dataset."""

import os
from dataclasses import dataclass

from benchmate import diffusion


@dataclass
class PrepareConfig:
    cache: str = None
    rev: str = None            # git tag/branch/commit to pin sd-scripts
    dataset: str = "naruto-blip-captions"
    dataset_id: str = diffusion.DEFAULT_DATASET
    train_images: int = 1000
    val_images: int = 32
    install_deps: bool = True  # pip install the repo's own requirements.txt


def main():
    from argklass import ArgumentParser

    parser = ArgumentParser()
    parser.add_arguments(PrepareConfig)
    args, _ = parser.parse_known_args()

    if args.cache:
        os.environ.setdefault("XDG_CACHE_HOME", str(args.cache))

    repo = diffusion.clone_repo("kohya", rev=args.rev, cache=args.cache)

    if args.install_deps:
        req = repo / "requirements.txt"
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
