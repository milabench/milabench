#!/usr/bin/env python
"""SimpleTuner (bghira) fine-tuning benchmark.

Runs a fixed number of SDXL LoRA training steps with `python train_sdxl.py`
on a vendored checkout of https://github.com/bghira/SimpleTuner.
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from benchmate import diffusion


@dataclass
class Arguments:
    cache: str = None
    model: str = diffusion.DEFAULT_MODEL
    dataset: str = "naruto-blip-captions"
    dataset_id: str = diffusion.DEFAULT_DATASET
    train_images: int = 1000
    val_images: int = 32
    steps: int = 200
    batch_size: int = 1
    gradient_accumulation: int = 1
    resolution: int = 1024
    learning_rate: float = 1e-4
    lora_rank: int = 16
    num_workers: int = 8
    mixed_precision: str = "bf16"
    output: str = None


def build_command(args: Arguments, data_dir: Path, out_dir: Path):
    return [
        sys.executable, "train_sdxl.py",
        f"--pretrained_model_name_or_path={args.model}",
        f"--instance_data_dir={data_dir}",
        "--caption_strategy=textfile",
        f"--resolution={args.resolution}",
        f"--train_batch_size={args.batch_size}",
        f"--gradient_accumulation_steps={args.gradient_accumulation}",
        f"--max_train_steps={args.steps}",
        f"--learning_rate={args.learning_rate}",
        "--lr_scheduler=constant",
        f"--mixed_precision={args.mixed_precision}",
        f"--dataloader_num_workers={args.num_workers}",
        "--add_lora=True",
        f"--lora_rank={args.lora_rank}",
        f"--output_dir={out_dir}",
        "--checkpointing_steps=1000000",
        "--validation_steps=1000000",
        "--report_to=none",
        "--checkpoint_with_model=True",
    ]


def main():
    from argklass import ArgumentParser

    parser = ArgumentParser()
    parser.add_arguments(Arguments)
    args, _ = parser.parse_known_args()

    if args.cache:
        os.environ.setdefault("XDG_CACHE_HOME", str(args.cache))

    repo = diffusion.vendor_dir("simpletuner", cache=args.cache)
    data_dir, _ = diffusion.materialize_dataset(
        diffusion.dataset_dir(args.dataset, args.cache),
        dataset_id=args.dataset_id,
        train_count=args.train_images,
        val_count=args.val_images,
    )

    out_dir = Path(args.output or Path.cwd() / "output-simpletuner")
    out_dir.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ, USE_DEEPSPEED="false", BENCH_NAME="sdxl")

    seconds = diffusion.timed_run(
        build_command(args, data_dir, out_dir), cwd=repo, env=env
    )

    diffusion.report_metrics(
        tool="simpletuner",
        **diffusion.throughput(
            args.steps,
            seconds,
            args.batch_size * args.gradient_accumulation,
        ),
    )


if __name__ == "__main__":
    main()
