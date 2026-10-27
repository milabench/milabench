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

from torchtitan.components.data import (
    ConcatThenSplitPackingConfig,
    SingleDatasetConfig,
)
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
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.distributed.activation_checkpoint import FullAC, SelectiveAC
from torchtitan.hf_datasets.text_datasets import TextProcessor
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.qwen3_8 import build_model_config
from torchtitan.observability.metrics import MetricsProcessor

from . import assets
from .data import (
    FineWebSource,
    NullTokenizer,
    PooledGrainDataLoader,
    RandomTokenDataLoader,
)
from .layout import patch_fsdp_placements, use_row_stacked_linears
from torchtitan.trainer import Trainer

from . import instrument

OPTIONS_ENV = "TT_NATIVE_OPTIONS"

CompileMode = Literal["eager", "regions", "regions+graphs"]
ACMode = Literal["none", "selective", "full"]
Dataset = Literal["fineweb", "random"]
StackedLayout = Literal["rows", "torchtitan"]


@dataclass(kw_only=True)
class Options:
    # Model
    flavor: str = "27B"
    n_layers: int | None = None
    """Truncate to the first N layers (multiple of 4 keeps the 3:1 GDN/attention mix)."""
    attn_backend: str = "flex"
    stacked_layout: StackedLayout = "rows"
    """``rows``: store stacked W1/W3 as [F, 2, D] so FSDP shards rows (any GPU
    count). ``torchtitan``: upstream [2, F, D] layout, FSDP shards dim 1 and
    the shard group must divide F (see tt_native/layout.py)."""

    # Data
    dataset: Dataset = "fineweb"
    samples: int = assets.DEFAULT_FINEWEB_SAMPLES
    """FineWeb documents read (and downloaded by prepare.py)."""
    data_pool: int = 0
    """> 0: read this many FineWeb microbatches at startup and cycle them (no
    file reads or tokenization while training; A/B of the loader cost)."""
    max_documents: int = 64
    """Document segments per microbatch (fixed varlen/mask capacity). The packer
    pads the microbatch when it is reached; padding is not counted as tokens."""

    # Batch (per GPU, fixed across scales: weak scaling)
    seq_len: int = 4096
    microbatch_sequences: int = 3
    grad_accum: int = 1

    # Performance
    compile: CompileMode = "regions+graphs"
    ac: ACMode = "selective"
    dp_replicate: int = 1
    tp: int = 1
    fsdp_reshard_after_forward: Literal["default", "always", "never"] = "default"
    adam_moments: Literal["fp32", "bf16"] = "fp32"
    """AdamW moment dtype. bf16 saves 4 B/param (low-memory fallback)."""

    # Distributed timeouts. init covers process-group setup, model init and the
    # first step (compilation, CUDA graph capture); train covers later steps.
    init_timeout_seconds: int = 600
    train_timeout_seconds: int = 180

    # Logging
    log_freq: int = 10
    wandb: bool = False
    """TorchTitan's wandb logger on rank 0 (also on with ``BENCHMATE_WANDB=1``)."""
    wandb_all_ranks: bool = False
    """One wandb run per rank instead of rank 0 only (also ``BENCHMATE_WANDB_ALL_RANKS=1``)."""
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
    instrument.RUN_DESCRIPTION["n_layers_full"] = len(model.layers)
    if opts.n_layers is not None:
        if not 0 < opts.n_layers <= len(model.layers):
            raise ValueError(f"n_layers must be in [1, {len(model.layers)}]")
        model.layers = model.layers[: opts.n_layers]
    if opts.compile == "eager":
        model.local_compile_regions = []
    if opts.stacked_layout == "rows":
        if opts.tp > 1:
            raise ValueError("stacked_layout=rows does not support tensor parallelism")
        use_row_stacked_linears(model)
        patch_fsdp_placements()
    return model


def _not_none(sample) -> bool:
    return sample is not None


def _data(opts: Options, vocab_size: int):
    """(hf_assets_path, tokenizer, dataloader) configs for ``opts.dataset``."""
    if opts.dataset == "random":
        return (
            None,
            NullTokenizer.Config(vocab_size=vocab_size),
            RandomTokenDataLoader.Config(
                vocab_size=vocab_size,
                seed=opts.seed,
                max_num_documents=opts.microbatch_sequences,
            ),
        )
    # Documents (BOS/EOS added by the tokenizer) are concatenated and split
    # into microbatches; positions restart at each document, so attention and
    # the GatedDeltaNet recurrence stay within documents (segments of at most
    # seq_len tokens).
    dataset = SingleDatasetConfig(
        source=FineWebSource.Config(
            path=str(assets.fineweb_dir()),
            samples=opts.samples,
            # Set by scripts/slurm/tt_native.sh (TT_LOCAL_DATA=1).
            cache_dir=os.environ.get("TT_NATIVE_DATASETS_CACHE"),
        ),
        processor=TextProcessor.Config(),
        post_filters=(_not_none,),
    )
    return (
        str(assets.tokenizer_dir()),
        HuggingFaceTokenizer.Config(),
        PooledGrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=dataset),
            seed=opts.seed,
            max_num_documents=opts.max_documents,
            pool=opts.data_pool,
        ),
    )


def _activation_checkpoint(mode: ACMode):
    if mode == "none":
        return None
    if mode == "selective":
        return SelectiveAC.Config()
    if mode == "full":
        return FullAC.Config()
    raise ValueError(f"unknown activation checkpointing mode {mode!r}")


def qwen38() -> Trainer.Config:
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
        **instrument.RUN_DESCRIPTION,
        "model": "qwen3_8",
        "flavor": opts.flavor,
        "n_layers": len(model.layers),
        "text_only": True,
        "attn_backend": opts.attn_backend,
        "stacked_layout": opts.stacked_layout,
        "precision": "bf16 (fp32 master weights, FSDP mixed precision)",
        "compile": opts.compile,
        "cuda_graphs": opts.compile == "regions+graphs",
        "local_compile_regions": list(model.local_compile_regions),
        "activation_checkpointing": opts.ac,
        "adam_moments": opts.adam_moments,
        "seq_len": opts.seq_len,
        "microbatch_sequences_per_gpu": opts.microbatch_sequences,
        "grad_accum": opts.grad_accum,
        "data": (
            "random tokens (packed, fixed length)"
            if opts.dataset == "random"
            else f"FineWeb {assets.FINEWEB_SUBSET} (first {opts.samples} docs), "
            f"{assets.TOKENIZER_REPO} tokenizer, packed with document masking"
        ),
        "data_cache": os.environ.get("TT_NATIVE_DATASETS_CACHE") or "HF_HOME",
        "options": asdict(opts),
    }

    hf_assets_path, tokenizer, dataloader = _data(opts, vocab_size)

    # Recorded in the summary pushed by the voirfile (tt_native.instrument).
    instrument.RUN_DESCRIPTION = description

    return Trainer.Config(
        model=model,
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(global_vocab_size=vocab_size),
        ),
        hf_assets_path=hf_assets_path,
        tokenizer=tokenizer,
        dataloader=dataloader,
        optim=Optim.Config(
            optimizer=OptimizersContainer.Config(
                optimizers=[
                    AdamW.Config(
                        pattern=r".*",
                        lr=3e-4,
                        moment_dtype=(
                            "bfloat16" if opts.adam_moments == "bf16" else "parameter"
                        ),
                    )
                ]
            ),
            lr_scheduler=LRSchedulersContainer.Config(warmup_steps=10),
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=tokens_per_microbatch,
            num_tokens_per_train_step=tokens_per_train_step,
            max_context_length=opts.seq_len,
            # The voirfile stops the run (voir skip/stop); keep the LR schedule long.
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
        metrics=MetricsProcessor.Config(
            log_freq=opts.log_freq,
            enable_wandb=opts.wandb or os.environ.get("BENCHMATE_WANDB") == "1",
            save_for_all_ranks=opts.wandb_all_ranks
            or os.environ.get("BENCHMATE_WANDB_ALL_RANKS") == "1",
        ),
        checkpointer=None,
        activation_checkpoint=_activation_checkpoint(opts.ac),
        dump_folder=opts.dump_folder,
    )
