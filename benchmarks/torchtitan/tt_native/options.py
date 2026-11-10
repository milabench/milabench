"""Run options shared by every config of the pack (native Qwen3.8 and HF models).

TorchTitan config functions take no arguments, so run options are passed by
``main.py`` through the ``TT_NATIVE_OPTIONS`` environment variable (JSON) and
parsed into :class:`Options` here.

Kept free of model imports: the native models pull CUDA-only kernels
(attn-gym), which the HF configs must not import.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, fields
from typing import Literal

from torchtitan.components.data import (
    ConcatThenSplitPackingConfig,
    SingleDatasetConfig,
)
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.distributed.activation_checkpoint import FullAC, SelectiveAC
from torchtitan.hf_datasets.text_datasets import TextProcessor

from . import assets
from .data import (
    FineWebSource,
    NullTokenizer,
    PooledGrainDataLoader,
    RandomTokenDataLoader,
)

OPTIONS_ENV = "TT_NATIVE_OPTIONS"

CompileMode = Literal["eager", "regions", "regions+graphs"]
ACMode = Literal["none", "selective", "full"]
Dataset = Literal["fineweb", "random"]
StackedLayout = Literal["rows", "torchtitan"]


@dataclass(kw_only=True)
class Options:
    recipe: str = "qwen38"
    """Config function in ``tt_configs`` (``qwen38`` or an HF model, e.g.
    ``mistral_7b_pretrain``). Not ``--config``: voir takes that flag."""

    # Model (native Qwen3.8 only)
    flavor: str = "27B"
    n_layers: int | None = None
    """Truncate to the first N layers (multiple of 4 keeps the 3:1 GDN/attention mix)."""
    attn_backend: str = "flex"
    stacked_layout: StackedLayout = "rows"
    """``rows``: store stacked W1/W3 as [F, 2, D] so FSDP shards rows (any GPU
    count). ``torchtitan``: upstream [2, F, D] layout, FSDP shards dim 1 and
    the shard group must divide F (see tt_native/layout.py)."""

    # Data (pre-training; SFT configs read their chat dataset)
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
    ep: int = 1
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

    @property
    def tokens_per_microbatch(self) -> int:
        return self.microbatch_sequences * self.seq_len

    @property
    def tokens_per_train_step(self) -> int:
        """TorchTitan's global step size: -1 (one microbatch) unless accumulating."""
        if self.grad_accum <= 1:
            return -1
        dp_degree = int(os.environ["WORLD_SIZE"]) // self.tp
        return self.tokens_per_microbatch * dp_degree * self.grad_accum


def _not_none(sample) -> bool:
    return sample is not None


def pretrain_data(opts: Options, vocab_size: int, tokenizer=None):
    """(hf_assets_path, tokenizer, dataloader) configs for ``opts.dataset``.

    ``tokenizer``: (assets dir, tokenizer config) of the model; default: the
    Qwen3.8 tokenizer staged by prepare.py.
    """
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
    hf_assets_path, tokenizer_config = tokenizer or (
        str(assets.tokenizer_dir()),
        HuggingFaceTokenizer.Config(),
    )
    return (
        hf_assets_path,
        tokenizer_config,
        PooledGrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=dataset),
            seed=opts.seed,
            max_num_documents=opts.max_documents,
            pool=opts.data_pool,
        ),
    )


def activation_checkpoint(mode: ACMode):
    if mode == "none":
        return None
    if mode == "selective":
        return SelectiveAC.Config()
    if mode == "full":
        return FullAC.Config()
    raise ValueError(f"unknown activation checkpointing mode {mode!r}")
