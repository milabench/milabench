"""TorchTitan config functions for the native Qwen3.8 benchmark.

TorchTitan config functions take no arguments, so run options are passed by
``main.py`` through the ``TT_NATIVE_OPTIONS`` environment variable (JSON) and
parsed into :class:`Options` here.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, fields
from typing import Literal

import torch
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
from .layout import patch_fsdp_placements, patch_tp_sharding, use_row_stacked_linears
from torchtitan.trainer import Trainer

from . import instrument

OPTIONS_ENV = "TT_NATIVE_OPTIONS"

CompileMode = Literal["eager", "regions", "regions+graphs"]
ACMode = Literal["none", "selective", "full"]
Dataset = Literal["fineweb", "random"]
StackedLayout = Literal["rows", "torchtitan"]
Precision = Literal["bf16", "mxfp8"]
CheckpointContent = Literal["off", "full", "model"]
CheckpointAsync = Literal["disabled", "async", "async_with_pinned_mem"]

# Linears converted to MXFP8 (FQN substrings). Not in_proj_a / in_proj_b (48
# outputs, not a multiple of the 32-wide MXFP8 blocks; a tiny share of the
# FLOPs), not the LM head. feed_forward.w13 only with the torchtitan stacked
# layout: MXFP8Linear needs [N, F, D] weights (rows stores [F, N, D]).
MXFP8_TARGETS = (
    "attention.wq",
    "attention.wk",
    "attention.wv",
    "attention.wo",
    "delta_net.in_proj_q",
    "delta_net.in_proj_k",
    "delta_net.in_proj_v",
    "delta_net.in_proj_z",
    "delta_net.out_proj",
    "feed_forward.w13",
    "feed_forward.w2",
)


@dataclass(kw_only=True)
class Options:
    # Model
    model: Literal["qwen3_8", "kimi_k3"] = "qwen3_8"
    """TorchTitan model package; ``flavor`` names one of its flavors (``27B``, ``Kimi-K3``)."""
    flavor: str = "27B"
    n_layers: int | None = None
    """Truncate to the first N layers (multiple of 4 keeps the 3:1 GDN/attention mix)."""
    attn_res_block_size: int | None = None
    """Kimi K3 only: layers per Attention Residuals block (the flavor's: 12). Smaller in a
    truncated stand-in, its later stages receive as deep a block stack as the full model's."""
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
    sac_save_scaled_mm: bool = False
    """Selective AC also saves every ``_scaled_mm_v2`` output (Trillium C-speed test)."""
    precision: Precision = "bf16"
    """``mxfp8``: MXFP8 linears (MXFP8_TARGETS, B200 / sm_100 only)."""
    grad_reduce: Literal["fp32", "bf16"] = "fp32"
    """Gradient reduce-scatter / all-reduce dtype (FSDP mixed precision)."""
    fsdp_reshard_after_forward: Literal["default", "always", "never"] = "default"
    adam_moments: Literal["fp32", "bf16"] = "fp32"
    """AdamW moment dtype. bf16 saves 4 B/param (low-memory fallback)."""
    training_dtype: Literal["float32", "bfloat16"] = "float32"
    """TorchTitan's ``training.dtype``. ``float32``: FP32 master weights, BF16
    compute through FSDP mixed precision. ``bfloat16``: parameters, gradients
    and optimizer state in BF16, no FP32 copy."""

    # Parallelism. ``layout`` names a layout and sets the degrees below:
    # ``fsdp`` (all GPUs), ``hsdp<k>`` (shard over k GPUs, replicate across),
    # ``tp<k>-fsdp``, ``pp<k>-fsdp``. The replicate degree of ``dp_shard > 0`` is
    # derived from the world size (entries do not know the node count).
    layout: str | None = None
    dp_replicate: int = 1
    dp_shard: int = -1
    tp: int = 1
    pp: int = 1
    pp_schedule: str = "1F1B"
    pp_microbatches: int | None = None
    """Pipeline microbatches per step and DP rank (default: ``pp``). Each has
    ``microbatch_sequences`` sequences."""
    pp_layers_per_stage: int | None = None
    pp_first_stage_less_layers: int = 1
    pp_last_stage_less_layers: int = 1
    """TorchTitan's split counts the embedding / the output as this many layers
    (0: the first / last stage holds as many decoder layers as the others)."""
    ep: int = 1
    """Expert-parallel degree (MoE models), taken from each stage's FSDP ranks."""

    # Checkpoint save / load timing, after the measurement (tt_native/checkpoint.py).
    checkpoint: CheckpointContent = "off"
    """``full``: model + optimizer + data loader state; ``model``: weights only (bf16)."""
    checkpoint_async: CheckpointAsync = "disabled"
    checkpoint_dir: str | None = None
    """Default: ``$TT_CKPT_ROOT/<milabench run name>``. Must not exist."""
    checkpoint_repeats: int = 1
    checkpoint_timeout_seconds: int = 1800
    """Process-group timeout while saving / loading (training uses train_timeout_seconds)."""

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
    profile_freq: int = 0
    """> 0: TorchTitan's profiler traces step ``profile_freq`` once, every rank,
    into ``$MILABENCH_DIR_RUN/traces``."""
    memory_snapshot_freq: int = 0
    """> 0: TorchTitan's CUDA memory snapshot every N steps and on out-of-memory, every
    rank, into ``$MILABENCH_DIR_RUN/memory_snapshot`` (view: pytorch.org/memory_viz)."""

    @classmethod
    def from_env(cls) -> "Options":
        raw = os.environ.get(OPTIONS_ENV, "{}")
        values = json.loads(raw)
        known = {f.name for f in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"Unknown {OPTIONS_ENV} keys: {sorted(unknown)}")
        return cls(**values)


def _patch_kimi_pp_stage() -> None:
    """Kimi K3's pipeline stage (TorchTitan 51b37c2) overrides ``forward_one_chunk``
    without torch's ``set_manual_backward_finalization(True)`` on an FSDP stage, so
    the schedule's gradient reduction (``finalize_backward``) fails at the first
    step with FSDP inside the stages (job 1261124). Set it as torch's stage does."""
    from torch.distributed.fsdp import FSDPModule
    from torchtitan.models.kimi_k3.pipeline_parallel.stage import AttnResPipelineStage

    forward_one_chunk = AttnResPipelineStage.forward_one_chunk

    def patched(self, *args, **kwargs):
        if isinstance(self.submod, FSDPModule) and self.has_backward:
            self.submod.set_manual_backward_finalization(True)
        return forward_one_chunk(self, *args, **kwargs)

    AttnResPipelineStage.forward_one_chunk = patched


def _patch_init_timeouts(seconds: int) -> None:
    """TorchTitan gives ``init_timeout_seconds`` to the default group only: the mesh
    groups, the pipeline's included, keep torch's 10-min NCCL default until its
    ``set_pg_timeouts`` lowers them and the default group to the train timeout after
    step 1. A 48-stage K3 pipeline infers its stage shapes one stage after another before
    step 1, ~21 s each: the last stages timed out at 600 s (job 1262377). Apply the init
    timeout to every group with that same function before training starts. The
    pipeline's 2-rank P2P child groups, created during step 1, copy it and keep it for
    the whole run (the lowering after step 1 does not reach them)."""
    from datetime import timedelta

    from torchtitan.distributed import utils as dist_utils
    from torchtitan.trainer import Trainer

    train = Trainer.train

    def patched(self, *args, **kwargs):
        dist_utils.set_pg_timeouts(timedelta(seconds=seconds), self.engine.parallelism_context)
        return train(self, *args, **kwargs)

    Trainer.train = patched


def _model_config(opts: Options):
    build = build_model_config
    if opts.model == "kimi_k3":
        from torchtitan.models.kimi_k3 import build_model_config as build

        _patch_kimi_pp_stage()
        _patch_init_timeouts(opts.init_timeout_seconds)
    model = build(opts.flavor, seq_len=opts.seq_len, attn_backend=opts.attn_backend)
    # Text-only pre-training: the 27B and Kimi-K3 flavors ship a vision encoder, which would
    # run on dummy inputs every step and is not part of the decoder FLOPs.
    model.vision_encoder = None
    instrument.RUN_DESCRIPTION["n_layers_full"] = len(model.layers)
    if opts.n_layers is not None:
        if not 0 < opts.n_layers <= len(model.layers):
            raise ValueError(f"n_layers must be in [1, {len(model.layers)}]")
        model.layers = model.layers[: opts.n_layers]
    if opts.attn_res_block_size is not None:
        if opts.model != "kimi_k3":
            raise ValueError("attn_res_block_size is a Kimi K3 option")
        # K3's pipeline reads the block layout from layer 0's config: set it on every layer.
        for layer in model.layers:
            layer.attn_res_block_size = opts.attn_res_block_size
    if opts.compile == "eager":
        model.local_compile_regions = []
    if opts.stacked_layout == "rows":
        use_row_stacked_linears(model)
        patch_fsdp_placements()
        patch_tp_sharding()
    if opts.precision == "mxfp8":
        model = _mxfp8(model, opts)
    return model


def _mxfp8(model, opts: Options):
    """Convert MXFP8_TARGETS to MXFP8Linear (after truncation and the row layout).

    Without sm_100 or the torchao kernels the run must still record its
    ``run_start``: the error is deferred to the trainer start
    (``instrument.FATAL``) and the model is left in BF16.
    """
    from torchtitan.models.common.linear import Linear
    from torchtitan.quantization import MXFP8Linear

    targets = [
        t for t in MXFP8_TARGETS if not (opts.stacked_layout == "rows" and t == "feed_forward.w13")
    ]
    try:
        from torchtitan.config.transform.quantization import MXFP8LinearConverter

        converter = MXFP8LinearConverter.Config(fqns=targets).build()
        # Conversion errors too (e.g. a dimension not a multiple of 32) are
        # raised after run_start; the run fails then, whatever the model holds.
        model = converter.convert(model)
    except (ImportError, ValueError, TypeError, RuntimeError) as exc:
        gpu = "no GPU"
        if torch.cuda.is_available():
            gpu = "GPU sm_{}{}".format(*torch.cuda.get_device_capability())
        instrument.FATAL = f"precision=mxfp8 unavailable ({gpu}): {exc}"
        return model
    instrument.RUN_DESCRIPTION["mxfp8_targets"] = targets
    instrument.RUN_DESCRIPTION["quantized_linears"] = sum(
        isinstance(cfg, MXFP8Linear.Config) for _, cfg, _, _ in model.traverse(Linear.Config)
    )
    return model


def parallel_degrees(opts: Options, world_size: int) -> dict[str, int | str]:
    """Degrees of ``opts.layout`` (or of the explicit options) for ``world_size`` GPUs."""
    tp, pp, dp_shard, dp_replicate = opts.tp, opts.pp, opts.dp_shard, opts.dp_replicate
    layout = opts.layout
    if layout is not None:
        match = re.fullmatch(r"fsdp|hsdp(\d+)|tp(\d+)-fsdp|pp(\d+)-fsdp", layout)
        if match is None:
            raise ValueError(f"unknown layout {layout!r}: fsdp, hsdp<k>, tp<k>-fsdp, pp<k>-fsdp")
        hsdp, tp_k, pp_k = (int(g) if g else None for g in match.groups())
        if (hsdp and dp_shard not in (-1, hsdp)) or (tp_k and tp not in (1, tp_k)) or (
            pp_k and pp not in (1, pp_k)
        ):
            raise ValueError(f"layout={layout} conflicts with dp_shard/tp/pp options")
        dp_shard, tp, pp = hsdp or dp_shard, tp_k or tp, pp_k or pp
    if dp_shard > 0:
        if world_size % (dp_shard * tp * pp):
            raise ValueError(
                f"{world_size} GPUs do not split into dp_shard={dp_shard} x tp={tp} x pp={pp}"
            )
        dp_replicate = world_size // (dp_shard * tp * pp)
    if layout is None:
        shard = world_size // (dp_replicate * tp * pp)
        layout = "-".join(
            [f"tp{tp}"] * (tp > 1)
            + [f"pp{pp}"] * (pp > 1)
            + [f"hsdp{shard}" if dp_replicate > 1 else "fsdp"]
        )
    return {"layout": layout, "dp_replicate": dp_replicate, "dp_shard": dp_shard, "tp": tp, "pp": pp}


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
        str(assets.tokenizer_dir(opts.model)),
        HuggingFaceTokenizer.Config(),
        PooledGrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=dataset),
            seed=opts.seed,
            max_num_documents=opts.max_documents,
            pool=opts.data_pool,
        ),
    )


class _SaveScaledMM(SelectiveAC):
    """Selective AC that also saves MXFP8's GEMMs. ``_scaled_mm_v2`` is outside the policy's
    every-second-mm count, so this saves all of them, where BF16 saves every second ``mm``."""

    @dataclass(kw_only=True, slots=True)
    class Config(SelectiveAC.Config):
        pass

    def get_save_ops(self) -> set:
        return super().get_save_ops() | {torch.ops.aten._scaled_mm_v2.default}


def _activation_checkpoint(mode: ACMode, save_scaled_mm: bool = False):
    if mode == "none":
        return None
    if mode == "selective":
        return _SaveScaledMM.Config() if save_scaled_mm else SelectiveAC.Config()
    if mode == "full":
        return FullAC.Config()
    raise ValueError(f"unknown activation checkpointing mode {mode!r}")


def _checkpointer(opts: Options):
    """CheckpointManager that never saves while training (``load_only``):
    tt_native/checkpoint.py saves and loads once the measurement is done."""
    if opts.checkpoint == "off":
        return None
    from torchtitan.components.checkpointer import CheckpointManager

    if opts.checkpoint == "model" and opts.checkpoint_async != "disabled":
        raise ValueError("checkpoint=model is saved synchronously only (TorchTitan last-step save)")
    folder = opts.checkpoint_dir
    if folder is None:
        root = os.environ.get("TT_CKPT_ROOT")
        if not root:
            raise ValueError("checkpoint needs checkpoint_dir or TT_CKPT_ROOT")
        run = json.loads(os.environ.get("MILABENCH_CONFIG") or "{}").get("run_name")
        folder = os.path.join(root, run or f"job-{os.environ.get('SLURM_JOB_ID', 'local')}")
    folder = os.path.abspath(folder)
    if os.path.exists(folder):
        raise FileExistsError(f"checkpoint folder {folder} exists: refusing to overwrite or load it")
    return CheckpointManager.Config(
        folder=folder,
        interval=1,
        load_only=True,
        keep_latest_k=0,
        async_mode=opts.checkpoint_async,
        last_save_model_only=opts.checkpoint == "model",
        export_dtype="bfloat16",
    )


def _profiler(opts: Options):
    """TorchTitan's profiler: step ``profile_freq`` traced once on every rank; memory
    snapshots every ``memory_snapshot_freq`` steps."""
    from torchtitan.observability.profiler import Profiler

    runs = os.path.abspath(
        os.environ.get("MILABENCH_DIR_RUN") or os.environ.get("MILABENCH_DIR_RUNS", ".")
    )
    trace = (
        dict(
            enable_profiling=True,
            profile_freq=opts.profile_freq,
            profiler_warmup=1,
            profiler_repeat=1,
            save_traces_folder=os.path.join(runs, "traces"),
        )
        if opts.profile_freq > 0
        else {}
    )
    memory = (
        dict(
            enable_memory_snapshot=True,
            memory_snapshot_freq=opts.memory_snapshot_freq,
            save_memory_snapshot_folder=os.path.join(runs, "memory_snapshot"),
            # The live blocks keep their allocation stacks; the history (TorchTitan: 1M
            # events) only adds the timeline, and its size is per rank.
            memory_snapshot_max_entries=100_000,
        )
        if opts.memory_snapshot_freq > 0
        else {}
    )
    return Profiler.Config(**trace, **memory)


def qwen38() -> Trainer.Config:
    """Trainer config of every model of the pack (``Options.model``)."""
    opts = Options.from_env()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    degrees = parallel_degrees(opts, world_size)
    model = _model_config(opts)
    vocab_size = decoder_vocab_size(model)
    pp_microbatches = opts.pp_microbatches or degrees["pp"]
    tokens_per_microbatch = opts.microbatch_sequences * opts.seq_len
    tokens_per_train_step = -1
    if opts.grad_accum > 1:
        # TorchTitan takes the global step size; derive it from the world size.
        # Under PP a step of a DP rank is pp_microbatches microbatches.
        dp_degree = world_size // (degrees["tp"] * degrees["pp"])
        microbatches = (pp_microbatches if degrees["pp"] > 1 else 1) * opts.grad_accum
        tokens_per_train_step = tokens_per_microbatch * dp_degree * microbatches
    checkpointer = _checkpointer(opts)
    master = (
        "fp32 master weights"
        if opts.training_dtype == "float32"
        else "no fp32 master weights (training.dtype bfloat16)"
    )
    moe = next((layer.moe for layer in model.layers if getattr(layer, "moe", None)), None)

    description = {
        **instrument.RUN_DESCRIPTION,
        "model": opts.model,
        "flavor": opts.flavor,
        "n_layers": len(model.layers),
        "text_only": True,
        "attn_backend": opts.attn_backend,
        "stacked_layout": opts.stacked_layout,
        "precision": opts.precision,
        "precision_detail": (
            f"bf16 ({master}, FSDP mixed precision)"
            if opts.precision == "bf16"
            else "MXFP8 linears (torchao kernels, 32x32 weight / 1x32 activation scales), "
            f"bf16 elsewhere, {master}"
        ),
        "training_dtype": opts.training_dtype,
        "ep": opts.ep,
        "moe": None if moe is None else {"experts": moe.num_experts, "top_k": moe.router.top_k},
        "flops_formula": None if opts.model == "qwen3_8" else (
            "TorchTitan KimiK3Model.get_nparams_and_flops: 6*active_params (routed experts "
            "counted top_k of num_experts, input embedding excluded) + per MLA layer "
            "quadratic_attention_flops_per_token + per KDA layer delta_rule_flops_per_token. "
            "mfu = flops_per_token * tokens_per_s / (peak * gpus)"
        ),
        "grad_reduce_dtype": opts.grad_reduce,
        "layout": degrees["layout"],
        "pp_schedule": opts.pp_schedule if degrees["pp"] > 1 else None,
        "pp_microbatches": pp_microbatches if degrees["pp"] > 1 else None,
        "checkpoint": None
        if checkpointer is None
        else {"content": opts.checkpoint, "async": opts.checkpoint_async, "folder": checkpointer.folder},
        "compile": opts.compile,
        "cuda_graphs": opts.compile == "regions+graphs",
        "local_compile_regions": list(model.local_compile_regions),
        "activation_checkpointing": opts.ac,
        "adam_moments": opts.adam_moments,
        "seq_len": opts.seq_len,
        "microbatch_sequences_per_gpu": opts.microbatch_sequences,
        "grad_accum": opts.grad_accum,
        # TP and PP ranks share their data-parallel rank's tokens; the summary recomputes it from the trainer.
        "tokens_per_gpu_per_step": opts.microbatch_sequences * opts.seq_len * opts.grad_accum
        * (pp_microbatches if degrees["pp"] > 1 else 1) // (degrees["tp"] * degrees["pp"]),
        "data": (
            "random tokens (packed, fixed length)"
            if opts.dataset == "random"
            else f"FineWeb {assets.FINEWEB_SUBSET} (first {opts.samples} docs), "
            f"{assets.TOKENIZERS[opts.model][0]} tokenizer, packed with document masking"
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
            dtype=opts.training_dtype,
            disable_cuda_graphs=opts.compile != "regions+graphs",
            mixed_precision_param="bfloat16",
            mixed_precision_reduce="bfloat16" if opts.grad_reduce == "bf16" else "float32",
        ),
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=degrees["dp_replicate"],
            data_parallel_shard_degree=degrees["dp_shard"],
            tensor_parallel_degree=degrees["tp"],
            pipeline_parallel_degree=degrees["pp"],
            pipeline_parallel_schedule=opts.pp_schedule,
            num_pp_microbatches=pp_microbatches,
            pipeline_parallel_layers_per_stage=opts.pp_layers_per_stage,
            pipeline_parallel_first_stage_less_layers=opts.pp_first_stage_less_layers,
            pipeline_parallel_last_stage_less_layers=opts.pp_last_stage_less_layers,
            expert_parallel_degree=opts.ep,
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
        checkpointer=checkpointer,
        activation_checkpoint=_activation_checkpoint(opts.ac, opts.sac_save_scaled_mm),
        dump_folder=opts.dump_folder,
        **(
            {"profiler": _profiler(opts)}
            if opts.profile_freq > 0 or opts.memory_snapshot_freq > 0
            else {}
        ),
    )
