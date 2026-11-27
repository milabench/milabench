#!/usr/bin/env python
"""LLM LoRA/QLoRA fine-tuning benchmark via unsloth + trl SFTTrainer.

Mirrors the sdiffusion packs: fixed job shape, wall-clock throughput metrics.
unsloth must be imported before transformers/trl (kernel patching).
"""

import os
import time
from dataclasses import dataclass

from argklass import ArgumentParser

import torch

if hasattr(torch, "xpu"):
    # Auto padding-free injects packed_seq_lengths whose metadata path
    # deadlocks on Intel XPU (torch 2.10+xpu, unsloth 2026.9.12).
    os.environ.setdefault("UNSLOTH_DISABLE_AUTO_PADDING_FREE", "1")

import unsloth  # noqa: E402  (must precede trl/transformers)
from unsloth import FastLanguageModel

if hasattr(torch, "xpu"):
    import unsloth.models.llama as _ul

    def _safe_set_cos_sin_cache(self, seq_len, device, dtype):
        # unsloth computes the rope cache on CPU then copies it with
        # non_blocking=True; async H2D from pageable memory faults the xe
        # driver (UR_RESULT_ERROR_DEVICE_LOST). Build it on device instead.
        self.current_rope_size = seq_len
        inv_freq = self.inv_freq.to(device=device, dtype=torch.float32)
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        t = self._apply_time_scaling(t)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = (emb.cos() * self.attention_scaling).to(dtype)
        sin = (emb.sin() * self.attention_scaling).to(dtype)
        self.multi_gpu_cos_cached[device.index] = cos
        self.multi_gpu_sin_cached[device.index] = sin
        return cos, sin

    _ul.LlamaRotaryEmbedding._set_cos_sin_cache = _safe_set_cos_sin_cache
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer


@dataclass
class Arguments:
    """unsloth LoRA fine-tuning benchmark."""

    #: base model id on the hub
    model: str = "unsloth/Llama-3.2-1B"
    #: dataset id (text-field dataset)
    dataset: str = "Trelis/tiny-shakespeare"
    #: dataset text column
    text_field: str = "Text"
    #: optimizer steps to run
    steps: int = 60
    #: sequences per step
    batch_size: int = 1
    #: effective batch multiplier
    gradient_accumulation: int = 1
    #: sequence length
    seq_length: int = 2048
    #: LoRA rank
    lora_r: int = 16
    #: 4-bit QLoRA via bitsandbytes
    load_in_4bit: bool = False
    #: gradient checkpointing: unsloth | true | none
    grad_ckpt: str = "unsloth"
    #: optimizer
    optim: str = "adamw_8bit"
    #: device map override, e.g. xpu:0 (default: unsloth auto-detect)
    device: str = None
    #: hub token
    hf_token: str = os.environ.get("HF_TOKEN")
    #: checkpoint output dir (default: under $MILABENCH_BASE/outputs)
    output: str = None
    #: seed
    seed: int = 3407
    #: pack sequences into full-length blocks (unsloth fast path)
    packing: bool = False
    #: random
    random: bool = False


def main():
    parser = ArgumentParser()
    parser.add_arguments(Arguments)
    args, _ = parser.parse_known_args()
    if args.random:
        import random as _r

        args.seed = _r.randint(0, 2**32 - 1)

    device = args.device
    if device is None:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            device = f"xpu:{os.environ.get('MILABENCH_GPU_ID', '0')}"
        elif torch.cuda.is_available():
            device = f"cuda:{os.environ.get('MILABENCH_GPU_ID', '0')}"

    grad_ckpt = {"unsloth": "unsloth", "true": True, "none": False}[args.grad_ckpt]

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.seq_length,
        load_in_4bit=args.load_in_4bit,
        device_map=device,
        token=args.hf_token,
        use_gradient_checkpointing=grad_ckpt,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=args.lora_r,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=grad_ckpt,
        random_state=args.seed,
    )

    dataset = load_dataset(args.dataset, split="train")

    output_dir = args.output
    if output_dir is None:
        import tempfile
        from pathlib import Path

        base = os.environ.get("MILABENCH_BASE")
        root = Path(base) if base else Path(tempfile.gettempdir()) / "milabench"
        output_dir = str(root / "outputs" / "unsloth")

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation,
            max_steps=args.steps,
            warmup_steps=0,
            learning_rate=2e-4,
            logging_steps=1,
            optim=args.optim,
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=args.seed,
            report_to=[],
            output_dir=output_dir,
            include_num_input_tokens_seen="non_padding",
            include_tokens_per_second=True,
            dataset_num_proc=1,
            dataset_text_field=args.text_field,
            max_length=args.seq_length,
            packing=args.packing,
        ),
    )

    t0 = time.perf_counter()
    result = trainer.train()
    seconds = time.perf_counter() - t0

    metrics = dict(result.metrics)
    tokens = metrics.get("train_input_tokens_per_second") or metrics.get("train_tokens_per_second")
    if not tokens:
        tokens = metrics.get("train_samples_per_second", 0.0) * args.seq_length

    from benchmate import diffusion

    diffusion.report_metrics(
        tool="unsloth",
        model=args.model,
        steps=args.steps,
        batch_size=args.batch_size * args.gradient_accumulation,
        seq_length=args.seq_length,
        lora_r=args.lora_r,
        qLoRA=args.load_in_4bit,
        seconds=round(seconds, 3),
        tok_per_s=round(tokens, 2),
        rate=round(tokens, 2),
    )


if __name__ == "__main__":
    main()
