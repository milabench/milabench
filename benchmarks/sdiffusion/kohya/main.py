#!/usr/bin/env python
"""Kohya sd-scripts fine-tuning benchmark.

Runs a fixed number of SDXL LoRA training steps with
`accelerate launch sdxl_train_network.py` on a vendored checkout of
https://github.com/kohya-ss/sd-scripts.
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


def kohya_data_root(data_dir: Path, out_dir: Path) -> Path:
    # kohya's auto-discovery only accepts sub-folders named <num_repeats>_<tokens>
    root = out_dir / "train-data"
    root.mkdir(parents=True, exist_ok=True)
    link = root / "1_dataset"
    if not link.is_symlink():
        if link.exists():
            import shutil
            shutil.rmtree(link)
        link.symlink_to(data_dir, target_is_directory=True)
    return root


def build_command(args: Arguments, data_dir: Path, out_dir: Path):
    return [
        sys.executable, "-m", "accelerate.commands.launch",
        "--num_processes", "1",
        "--mixed_precision", args.mixed_precision,
        "sdxl_train_network.py",
        f"--pretrained_model_name_or_path={args.model}",
        f"--train_data_dir={kohya_data_root(data_dir, out_dir)}",
        f"--resolution={args.resolution},{args.resolution}",
        f"--train_batch_size={args.batch_size}",
        f"--gradient_accumulation_steps={args.gradient_accumulation}",
        f"--max_train_steps={args.steps}",
        f"--learning_rate={args.learning_rate}",
        "--lr_scheduler=constant",
        "--lr_warmup_steps=0",
        "--optimizer_type=AdamW",
        f"--mixed_precision={args.mixed_precision}",
        f"--save_precision={args.mixed_precision}",
        "--full_bf16" if args.mixed_precision == "bf16" else "--fp16",
        "--network_module=networks.lora",
        f"--network_args=rank={args.lora_rank}",
        "--network_train_unet_only",
        "--enable_bucket",
        "--gradient_checkpointing",
        "--cache_latents",
        "--cache_text_encoder_outputs",
        f"--max_data_loader_n_workers={args.num_workers}",
        "--persistent_data_loader_workers",
        "--seed=42",
        f"--output_dir={out_dir}",
        "--output_name=milabench",
        "--save_every_n_epochs=999999",
    ]


def main():
    from argklass import ArgumentParser

    parser = ArgumentParser()
    parser.add_arguments(Arguments)
    args, _ = parser.parse_known_args()

    if args.cache:
        os.environ.setdefault("XDG_CACHE_HOME", str(args.cache))

    repo = diffusion.vendor_dir("kohya", cache=args.cache)
    data_dir, _ = diffusion.materialize_dataset(
        diffusion.dataset_dir(args.dataset, args.cache),
        dataset_id=args.dataset_id,
        train_count=args.train_images,
        val_count=args.val_images,
    )

    out_dir = Path(args.output or Path.cwd() / "output-kohya")
    out_dir.mkdir(parents=True, exist_ok=True)

    seconds = diffusion.timed_run(
        build_command(args, data_dir, out_dir), cwd=repo, env=dict(os.environ)
    )

    if not list(out_dir.glob("*.safetensors")):
        raise SystemExit("no LoRA checkpoint produced — training did not complete")

    diffusion.report_metrics(
        tool="kohya",
        **diffusion.throughput(
            args.steps,
            seconds,
            args.batch_size * args.gradient_accumulation,
        ),
    )


if __name__ == "__main__":
    main()
