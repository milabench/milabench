"""TorchTitan config functions for the native Qwen3.8 benchmark.

TorchTitan config functions take no arguments, so run options are passed by
``main.py`` through the ``TT_NATIVE_OPTIONS`` environment variable (JSON) and
parsed into :class:`Options` here.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from typing import Literal

from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optim import (
    AdamW,
    LRSchedulersContainer,
    Optim,
    OptimizersContainer,
)
from torchtitan.config import TrainingConfig
from torchtitan.config.configs import CommConfig, DebugConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.activation_checkpoint import FullAC, SelectiveAC
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.qwen3_8 import build_model_config
from torchtitan.observability.metrics import MetricsProcessor

from .data import NullTokenizer, RandomTokenDataLoader
from .trainer import BenchConfig, BenchTrainer

OPTIONS_ENV = "TT_NATIVE_OPTIONS"

CompileMode = Literal["eager", "regions", "regions+graphs"]
ACMode = Literal["none", "selective", "full"]


@dataclass(kw_only=True)
class Options:
    # Model
    flavor: str = "27B"
    n_layers: int | None = None
    """Truncate to the first N layers (multiple of 4 keeps the 3:1 GDN/attention mix)."""
    attn_backend: str = "flex"

    # Batch (per GPU, fixed across scales: weak scaling)
    seq_len: int = 8192
    microbatch_sequences: int = 1
    grad_accum: int = 1

    # Performance
    compile: CompileMode = "regions+graphs"
    ac: ACMode = "selective"
    dp_replicate: int = 1
    tp: int = 1
    fsdp_reshard_after_forward: Literal["default", "always", "never"] = "default"

    # Measurement
    warmup_steps: int = 40
    measure_steps: int = 100
    measure_seconds: float | None = None
    results_dir: str = "./results"
    run_name: str = "qwen38"

    # Distributed timeouts. init covers process-group setup, model init and the
    # first step (compilation, CUDA graph capture); train covers later steps.
    init_timeout_seconds: int = 600
    train_timeout_seconds: int = 180

    # Logging
    log_freq: int = 10
    structured_logging: bool = False
    """TorchTitan per-rank JSONL traces (one file per rank: off by default)."""
    dump_folder: str = "./outputs"
    seed: int = 1234

    @classmethod
    def from_env(cls) -> "Options":
        raw = os.environ.get(OPTIONS_ENV, "{}")
        values = json.loads(raw)
        known = {f.name for f in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"Unknown {OPTIONS_ENV} keys: {sorted(unknown)}")
        return cls(**values)


def _model_config(opts: Options):
    model = build_model_config(
        opts.flavor, seq_len=opts.seq_len, attn_backend=opts.attn_backend
    )
    # Text-only pre-training: the 27B flavor ships a vision encoder, which would
    # run on dummy inputs every step and is not part of the decoder FLOPs.
    model.vision_encoder = None
    if opts.n_layers is not None:
        if not 0 < opts.n_layers <= len(model.layers):
            raise ValueError(f"n_layers must be in [1, {len(model.layers)}]")
        model.layers = model.layers[: opts.n_layers]
    if opts.compile == "eager":
        model.local_compile_regions = []
    return model


def _activation_checkpoint(mode: ACMode):
    if mode == "none":
        return None
    if mode == "selective":
        return SelectiveAC.Config()
    if mode == "full":
        return FullAC.Config()
    raise ValueError(f"unknown activation checkpointing mode {mode!r}")


def qwen38() -> BenchTrainer.Config:
    opts = Options.from_env()
    model = _model_config(opts)
    vocab_size = decoder_vocab_size(model)
    tokens_per_microbatch = opts.microbatch_sequences * opts.seq_len
    tokens_per_train_step = -1
    if opts.grad_accum > 1:
        # TorchTitan takes the global step size; derive it from the world size.
        dp_degree = int(os.environ["WORLD_SIZE"]) // opts.tp
        tokens_per_train_step = tokens_per_microbatch * dp_degree * opts.grad_accum

    description = {
        "model": "qwen3_8",
        "flavor": opts.flavor,
        "n_layers": len(model.layers),
        "text_only": True,
        "attn_backend": opts.attn_backend,
        "precision": "bf16 (fp32 master weights, FSDP mixed precision)",
        "compile": opts.compile,
        "cuda_graphs": opts.compile == "regions+graphs",
        "local_compile_regions": list(model.local_compile_regions),
        "activation_checkpointing": opts.ac,
        "microbatch_sequences_per_gpu": opts.microbatch_sequences,
        "data": "random tokens (packed, fixed length)",
        "options": asdict(opts),
    }

    return BenchTrainer.Config(
        model=model,
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(global_vocab_size=vocab_size),
        ),
        hf_assets_path=None,
        tokenizer=NullTokenizer.Config(vocab_size=vocab_size),
        dataloader=RandomTokenDataLoader.Config(
            vocab_size=vocab_size,
            seed=opts.seed,
            max_num_documents=opts.microbatch_sequences,
        ),
        optim=Optim.Config(
            optimizer=OptimizersContainer.Config(
                optimizers=[AdamW.Config(pattern=r".*", lr=3e-4)]
            ),
            lr_scheduler=LRSchedulersContainer.Config(warmup_steps=10),
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=tokens_per_microbatch,
            num_tokens_per_train_step=tokens_per_train_step,
            max_context_length=opts.seq_len,
            # The bench trainer decides when to stop; keep the LR schedule long.
            steps=1_000_000,
            disable_cuda_graphs=opts.compile != "regions+graphs",
            mixed_precision_param="bfloat16",
            mixed_precision_reduce="float32",
        ),
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=opts.dp_replicate,
            data_parallel_shard_degree=-1,
            tensor_parallel_degree=opts.tp,
            fsdp_reshard_after_forward=opts.fsdp_reshard_after_forward,
        ),
        comm=CommConfig(
            init_timeout_seconds=opts.init_timeout_seconds,
            train_timeout_seconds=opts.train_timeout_seconds,
        ),
        debug=DebugConfig(
            seed=opts.seed,
            enable_structured_logging=opts.structured_logging,
        ),
        metrics=MetricsProcessor.Config(log_freq=opts.log_freq),
        checkpointer=None,
        activation_checkpoint=_activation_checkpoint(opts.ac),
        dump_folder=opts.dump_folder,
        bench=BenchConfig(
            warmup_steps=opts.warmup_steps,
            measure_steps=opts.measure_steps,
            measure_seconds=opts.measure_seconds,
            results_dir=opts.results_dir,
            run_name=opts.run_name,
            description=description,
        ),
    )
