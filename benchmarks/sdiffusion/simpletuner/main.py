#!/usr/bin/env python
"""SimpleTuner (bghira) fine-tuning benchmark.

Runs a fixed number of SDXL LoRA training steps with the 1.x config-driven
CLI (`simpletuner train` + config.json + multidatabackend.json) on a vendored
checkout of https://github.com/bghira/SimpleTuner.
"""

import json
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


def build_config(args: Arguments, data_dir: Path, out_dir: Path) -> dict:
    return {
        "model_family": "sdxl",
        "model_type": "lora",
        "pretrained_model_name_or_path": args.model,
        "output_dir": str(out_dir),
        "data_backend_config": "user_data/milabench-multidatabackend.json",
        "train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation,
        "max_train_steps": args.steps,
        "num_train_epochs": 0,
        "learning_rate": args.learning_rate,
        "lr_scheduler": "constant",
        "optimizer": "adamw_bf16",
        "mixed_precision": args.mixed_precision,
        "gradient_checkpointing": True,
        "resolution": args.resolution,
        "resolution_type": "pixel_area",
        "minimum_image_size": 0,
        "aspect_bucket_rounding": 2,
        "caption_dropout_probability": 0.0,
        "lora_rank": args.lora_rank,
        "use_ema": False,
        "checkpoint_step_interval": 100000000,
        "checkpoints_total_limit": 1,
        "validation_steps": 100000000,
        "validation_prompt": "a ninja with an orange",
        "validation_resolution": f"{args.resolution}x{args.resolution}",
        "num_validation_images": 1,
        "validation_seed": 42,
        "seed": 42,
        "push_to_hub": False,
        "push_checkpoints_to_hub": False,
        "report_to": "none",
        "disable_benchmark": True,
    }


def build_data_backend(args: Arguments, data_dir: Path) -> list:
    return [
        {
            "id": "milabench-instance",
            "type": "local",
            "instance_data_dir": str(data_dir),
            "crop": True,
            "crop_style": "random",
            "crop_aspect": "square",
            "resolution": args.resolution,
            "resolution_type": "pixel_area",
            "minimum_image_size": 0,
            "repeats": 1,
            "shuffle_tokens": False,
            "caption_strategy": "textfile",
            "metadata_backend": "discovery",
            "dataset_type": "image",
        },
        {
            "id": "milabench-text-embeds",
            "dataset_type": "text_embeds",
            "default": True,
            "type": "local",
            "cache_dir": "cache/text/milabench",
        },
        {
            "id": "milabench-image-embeds",
            "dataset_type": "image_embeds",
            "default": True,
            "type": "local",
            "cache_dir": "cache/image/milabench",
        },
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

    out_dir = Path(args.output or diffusion.output_dir("simpletuner"))
    out_dir.mkdir(parents=True, exist_ok=True)

    config_json = json.dumps(build_config(args, data_dir, out_dir), indent=2)
    (repo / "config.json").write_text(config_json)
    (repo / "config" / "config.json").write_text(config_json)
    user_data = repo / "user_data"
    user_data.mkdir(exist_ok=True)
    (user_data / "milabench-multidatabackend.json").write_text(
        json.dumps(build_data_backend(args, data_dir), indent=2)
    )

    env = dict(os.environ, TRACKER_DISABLED="1")

    seconds = diffusion.timed_run(
        [sys.executable, "st_cli.py", "train"], cwd=repo, env=env
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
