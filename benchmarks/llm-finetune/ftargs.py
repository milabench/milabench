"""Arguments shared by prepare.py and main.py (no torch / unsloth import here: prepare runs CPU only)."""

from dataclasses import dataclass


@dataclass
class Arguments:
    """LoRA / QLoRA fine-tuning benchmark (unsloth + TRL SFTTrainer)."""

    #: base model id on the hub (bf16 checkpoint, or an unsloth pre-quantized bnb-4bit one)
    model: str = "unsloth/Llama-3.2-1B"
    #: model revision (commit sha) to pin
    model_revision: str = None
    #: instruction dataset (ShareGPT "conversations" or "messages" column)
    dataset: str = "teknium/OpenHermes-2.5"
    #: dataset revision (commit sha) to pin
    dataset_revision: str = "b82037821055c377bed0d495e72e46de3bc72e84"
    #: rows of the dataset used (the first N, in file order: a fixed slice)
    num_samples: int = 20000
    #: optimizer steps measured (after the warmup steps)
    steps: int = 300
    #: optimizer steps run first and not measured (compile, allocator, autotune)
    warmup_steps: int = 10
    #: sequences per micro-batch
    batch_size: int = 1
    #: micro-batches per optimizer step
    gradient_accumulation: int = 4
    #: sequence length: sequences are packed into blocks of exactly this many tokens
    seq_length: int = 2048
    #: LoRA rank (alpha = 2 * rank)
    lora_r: int = 16
    #: QLoRA: quantize the base to 4-bit NF4 at load (bitsandbytes); a pre-quantized repo needs it too
    load_in_4bit: bool = False
    #: gradient checkpointing: unsloth | true | none
    grad_ckpt: str = "unsloth"
    #: optimizer
    optim: str = "adamw_8bit"
    #: learning rate (linear decay, no LR warmup: the loss check sees the whole schedule)
    learning_rate: float = 2e-4
    #: loader: auto (FastModel for multimodal checkpoints, FastLanguageModel otherwise) | language | model
    loader: str = "auto"
    #: sanity check: mean loss of the last 10 steps must be at least this fraction below the first 10
    min_loss_drop: float = 0.02
    #: fail the bench (exit 1) when the loss check fails
    require_loss_drop: bool = True
    #: seed
    seed: int = 3407
    #: checkpoint dir (nothing is saved; trainer scratch only)
    output: str = None
