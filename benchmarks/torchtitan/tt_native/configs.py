"""TorchTitan config function for the native Qwen3.8 benchmark.

Run options come from :mod:`tt_native.options` (``TT_NATIVE_OPTIONS``, set by
``main.py``). Imports Qwen3.8's Gated DeltaNet kernels (attn-gym): CUDA only.
"""

from __future__ import annotations

import os
from dataclasses import asdict

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
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.qwen3_8 import build_model_config
from torchtitan.observability.metrics import MetricsProcessor
from torchtitan.trainer import Trainer

from . import assets, instrument
from .layout import patch_fsdp_placements, use_row_stacked_linears
from .options import Options, activation_checkpoint, pretrain_data


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


def qwen38() -> Trainer.Config:
    opts = Options.from_env()
    model = _model_config(opts)
    vocab_size = decoder_vocab_size(model)

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

    hf_assets_path, tokenizer, dataloader = pretrain_data(opts, vocab_size)

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
            num_tokens_per_microbatch_per_dp_rank=opts.tokens_per_microbatch,
            num_tokens_per_train_step=opts.tokens_per_train_step,
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
        activation_checkpoint=activation_checkpoint(opts.ac),
        dump_folder=opts.dump_folder,
    )
