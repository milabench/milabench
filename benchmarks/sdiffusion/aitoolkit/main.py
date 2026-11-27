#!/usr/bin/env python
"""ai-toolkit (ostris) fine-tuning benchmark.

Runs a fixed number of LoRA training steps with `python run.py <job.yaml>`
on a vendored checkout of https://github.com/ostris/ai-toolkit.

The base model's architecture is ``--arch`` (ai-toolkit's model.arch: sdxl, the default, or flux, zimage,
flux2_klein_4b, qwen_image_2, wan22_5b); flow-matching models train with ``--noise_scheduler flowmatch`` (their
default), SDXL with ddpm. ``--quantize`` / ``--qtype`` / ``--quantize_te`` quantize the frozen base (optimum-quanto,
e.g. qfloat8 / uint4), ``--assistant_lora_path`` loads a training adapter (Z-Image Turbo's de-distill adapter),
``--num_frames`` sets the frames per clip of a video dataset. Unknown flags are an error: a misspelt option would
otherwise train another job than the one asked for.
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
    arch: str = "sdxl"
    noise_scheduler: str = None
    quantize: bool = False
    qtype: str = None
    quantize_te: bool = False
    qtype_te: str = None
    assistant_lora_path: str = None
    num_frames: int = None


# ai-toolkit model.arch values this driver knows; every one but sdxl is a flow-matching model
ARCHS = ("sdxl", "flux", "zimage", "flux2_klein_4b", "qwen_image_2", "wan22_5b")


def model_config(args: Arguments) -> dict:
    model = {"name_or_path": args.model, "arch": args.arch}
    if args.arch == "flux":
        model["is_flux"] = True  # older ai-toolkit checkouts read is_flux instead of arch
    if args.quantize:
        model["quantize"] = True
        if args.qtype:
            model["qtype"] = args.qtype
    if args.quantize_te:
        model["quantize_te"] = True
        if args.qtype_te:  # else ai-toolkit's default (qfloat8)
            model["qtype_te"] = args.qtype_te
    if args.assistant_lora_path:
        model["assistant_lora_path"] = args.assistant_lora_path
    return model


def build_job(args: Arguments, data_dir: Path, out_dir: Path) -> dict:
    return {
        "job": "extension",
        "config": {
            "name": "milabench",
            "process": [{
                "type": "sd_trainer",
                "training_folder": str(out_dir),
                "device": "cuda:0",
                "datasets": [{
                    "folder_path": str(data_dir),
                    "caption_ext": "txt",
                    "caption_dropout_rate": 0.0,
                    "cache_latents_to_disk": False,
                    "resolution": [args.resolution],
                    **({"num_frames": args.num_frames} if args.num_frames else {}),
                }],
                "train": {
                    "train_unet": True,
                    "train_text_encoder": False,
                    "gradient_accumulation": args.gradient_accumulation,
                    "batch_size": args.batch_size,
                    "steps": args.steps,
                    "lr": args.learning_rate,
                    "lr_scheduler": "constant",
                    "optimizer": "adamw",
                    "noise_scheduler": args.noise_scheduler or ("ddpm" if args.arch == "sdxl" else "flowmatch"),
                    "dtype": args.mixed_precision,
                    "gradient_checkpointing": True,
                    "skip_first_sample": True,
                    "disable_sampling": True,
                },
                "model": model_config(args),
                "lora": {
                    "type": "lora",
                    "linear": args.lora_rank,
                    "linear_alpha": args.lora_rank,
                },
                "save_every": 100000,
            }],
        },
        "run": True,
    }


def main():
    import yaml
    from argklass import ArgumentParser

    parser = ArgumentParser()
    parser.add_arguments(Arguments)
    args, unknown = parser.parse_known_args()
    if unknown:
        raise SystemExit(f"aitoolkit: unrecognized arguments: {' '.join(unknown)}")
    if args.arch not in ARCHS:
        raise SystemExit(f"aitoolkit: --arch {args.arch!r}: choose from {', '.join(ARCHS)}")

    if args.cache:
        os.environ.setdefault("XDG_CACHE_HOME", str(args.cache))

    repo = diffusion.vendor_dir("aitoolkit", cache=args.cache)
    data_dir, _ = diffusion.materialize_dataset(
        diffusion.dataset_dir(args.dataset, args.cache),
        dataset_id=args.dataset_id,
        train_count=args.train_images,
        val_count=args.val_images,
    )

    out_dir = Path(args.output or diffusion.output_dir("aitoolkit"))
    out_dir.mkdir(parents=True, exist_ok=True)

    job_file = out_dir / "milabench.yaml"
    job_file.write_text(yaml.safe_dump(build_job(args, data_dir, out_dir)))

    shim = str(Path(__file__).parent / "shim")
    env = dict(
        os.environ,
        ACCELERATE_NUM_PROCESSES="1",
        MILABENCH_DATASET_NUM_WORKERS=str(args.num_workers),
        PYTHONPATH=shim + os.pathsep + os.environ.get("PYTHONPATH", ""),
    )

    seconds = diffusion.timed_run(
        [sys.executable, "run.py", str(job_file)], cwd=repo, env=env, steps=args.steps, tool="ai-toolkit"
    )

    diffusion.report_metrics(
        tool="ai-toolkit",
        **diffusion.throughput(
            args.steps,
            seconds,
            args.batch_size * args.gradient_accumulation,
        ),
    )


if __name__ == "__main__":
    main()
