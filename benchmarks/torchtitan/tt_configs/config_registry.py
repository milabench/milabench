"""TorchTitan configs for the HF transformers backend benches (pretrain + SFT).

One function per bench; run options (sequence length, batch, data,
parallelism) come from :mod:`tt_native.options`, like the native configs.
Assets (config, tokenizer, SFT weights) are staged by prepare.py under
``{data}/hf/<model name>``.
"""

from __future__ import annotations

import copy
import os
from dataclasses import asdict
from pathlib import Path

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.data import (
    FirstFitPackingConfig,
    GrainDataLoader,
    HuggingFaceRandomAccessSource,
    SingleDatasetConfig,
)
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.optim import (
    AdamW,
    LRSchedulersContainer,
    Optim,
    OptimizersContainer,
)
from torchtitan.config import TrainingConfig
from torchtitan.config.configs import CommConfig, DebugConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.experiments.transformers_modeling_backend import (
    HFTransformerModel,
    TitanModelConfig,
    TitanMoeModelConfig,
)
from torchtitan.experiments.transformers_modeling_backend.tokenizer import (
    HFBackendTokenizer,
)
from torchtitan.hf_datasets.text_datasets import ChatProcessor
from torchtitan.observability.metrics import MetricsProcessor
from torchtitan.trainer import Trainer
from transformers import AutoConfig

from tt_native import instrument
from tt_native.assets import data_dir
from tt_native.options import Options, activation_checkpoint, pretrain_data

from .state_dict_adapter import MilabenchHFStateDictAdapter

# SFT loads the HF weights; upstream's adapter cannot map MoE checkpoints.
HFTransformerModel.state_dict_adapter_cls = MilabenchHFStateDictAdapter


def hf_assets(hf_model: str) -> Path:
    """Local dir written by prepare.py: ``{data}/hf/<model name>``."""
    return data_dir() / "hf" / hf_model.split("/")[-1]


def _sft_json() -> str:
    return str(data_dir() / "sft_test" / "data.json")


def _process_sft_sample(sample):
    return [
        {"role": "user", "content": sample["question"]},
        {"role": "assistant", "content": sample["answer"]},
    ]


def _restore_per_layer_config(model, hf_model_id: str) -> None:
    """Give the model the HF config's per-layer overrides (heterogeneous configs).

    load_hf_config copies the HF config's public attributes only, so it drops
    ``_heterogeneity_spec``: every layer then gets the global values (Gemma4's
    full-attention layers lose head_dim 512 and 2 KV heads, and no longer
    match the checkpoint).
    """
    hf = AutoConfig.from_pretrained(hf_model_id)
    hf = getattr(hf, "text_config", hf)
    spec = hf.__dict__.get("_heterogeneity_spec")
    if spec is not None:
        model.__dict__["_heterogeneity_spec"] = copy.deepcopy(spec)


def _model_config(opts: Options, hf_model: str, *, moe: bool, sft: bool):
    # Empty overrides: the HF config supplies every dimension (expert counts too).
    # SFT packs several chats per sequence: attention must stay within each one.
    titan = (TitanMoeModelConfig if moe else TitanModelConfig)(
        attn_mask_type="block_causal" if sft else "causal",
    )
    model = HFTransformerModel.Config(model_config=titan)
    assets = hf_assets(hf_model)
    # Local copy when prepared: runs are offline (HF_HUB_OFFLINE).
    hf_model_id = str(assets) if (assets / "config.json").exists() else hf_model
    model.load_hf_config(
        hf_model_id=hf_model_id,
        max_context_length=opts.seq_len,
        deterministic=False,
    )
    _restore_per_layer_config(model, hf_model_id)
    if opts.compile == "eager":
        model.local_compile_regions = []
    elif opts.compile != "regions":
        raise ValueError(f"compile={opts.compile!r}: HF models support eager or regions")
    return model


def _sft_data(opts: Options):
    return GrainDataLoader.Config(
        dataset=FirstFitPackingConfig(
            dataset=SingleDatasetConfig(
                source=HuggingFaceRandomAccessSource.Config(
                    path="json",
                    split="train",
                    load_dataset_kwargs={"data_files": _sft_json()},
                ),
                processor=ChatProcessor.Config(messages_fn=_process_sft_sample),
                post_filters=(lambda sample: sample is not None,),
            ),
        ),
        seed=opts.seed,
        max_num_documents=opts.max_documents,
    )


def hf_trainer(
    hf_model: str,
    *,
    moe: bool = False,
    sft: bool = False,
    lr: float | None = None,
) -> Trainer.Config:
    """Trainer config for ``hf_model``: pre-training from scratch on FineWeb
    (or random tokens), or SFT from the HF weights on the staged chat set."""
    opts = Options.from_env()
    model = _model_config(opts, hf_model, moe=moe, sft=sft)
    assets = str(hf_assets(hf_model))
    tokenizer = HFBackendTokenizer.Config()

    if sft:
        hf_assets_path, dataloader = assets, _sft_data(opts)
        checkpointer = CheckpointManager.Config(
            initial_load_in_hf=True,
            initial_load_model_only=True,
            initial_load_path=assets,
            # The voirfile stops the run: never save.
            interval=10**9,
        )
    else:
        hf_assets_path, tokenizer, dataloader = pretrain_data(
            opts, model.vocab_size, tokenizer=(assets, tokenizer)
        )
        checkpointer = None

    instrument.RUN_DESCRIPTION = {
        **instrument.RUN_DESCRIPTION,
        "model": hf_model.split("/")[-1],
        "flavor": "sft" if sft else "pretrain",
        "hf_model": hf_model,
        "backend": "transformers_modeling_backend",
        "n_layers": getattr(model, "num_hidden_layers", None),
        "n_layers_full": getattr(model, "num_hidden_layers", None),
        "moe": moe,
        "precision": "bf16 (fp32 master weights, FSDP mixed precision)",
        "compile": opts.compile,
        "cuda_graphs": False,
        "local_compile_regions": list(model.local_compile_regions),
        "activation_checkpointing": opts.ac,
        "adam_moments": opts.adam_moments,
        "seq_len": opts.seq_len,
        "microbatch_sequences_per_gpu": opts.microbatch_sequences,
        "grad_accum": opts.grad_accum,
        "data": (
            "SFT chat samples (sft_test), packed with document masking"
            if sft
            else "random tokens (packed, fixed length)"
            if opts.dataset == "random"
            else f"FineWeb (first {opts.samples} docs), {hf_model} tokenizer, packed"
        ),
        "flops_formula": (
            "TorchTitan HFTransformerModel.get_nparams_and_flops: 6*active_params "
            "(embeddings excluded) + attention. "
            "mfu = flops_per_token * tokens_per_s / (peak * gpus)"
        ),
        "options": asdict(opts),
    }

    return Trainer.Config(
        model=model,
        loss=CrossEntropyLoss.Config(global_vocab_size=model.vocab_size),
        hf_assets_path=hf_assets_path,
        tokenizer=tokenizer,
        dataloader=dataloader,
        optim=Optim.Config(
            optimizer=OptimizersContainer.Config(
                optimizers=[
                    AdamW.Config(
                        pattern=r".*",
                        lr=lr if lr is not None else (2e-5 if sft else 8e-4),
                        moment_dtype=(
                            "bfloat16" if opts.adam_moments == "bf16" else "parameter"
                        ),
                    )
                ]
            ),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2,
                decay_ratio=0.8,
                decay_type="linear",
                min_lr_factor=0.0,
            ),
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=opts.tokens_per_microbatch,
            num_tokens_per_train_step=opts.tokens_per_train_step,
            max_context_length=opts.seq_len,
            # The voirfile stops the run (voir skip/stop); keep the LR schedule long.
            steps=1_000_000,
            disable_cuda_graphs=True,
            # fp32 master weights, bf16 compute (FSDP mixed precision). bf16
            # parameters break MoE: the swapped-in experts keep fp32 state and
            # FSDP needs one parameter dtype per group.
            mixed_precision_param="bfloat16",
        ),
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=opts.dp_replicate,
            data_parallel_shard_degree=-1,
            tensor_parallel_degree=opts.tp,
            expert_parallel_degree=opts.ep,
            fsdp_reshard_after_forward=opts.fsdp_reshard_after_forward,
            pipeline_parallel_schedule="1F1B",
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
        activation_checkpoint=activation_checkpoint(opts.ac),
        dump_folder=opts.dump_folder,
    )


def qwen3_4b_pretrain():
    return hf_trainer("Qwen/Qwen3-4B-Instruct-2507")


def qwen3_4b_sft():
    return hf_trainer("Qwen/Qwen3-4B-Instruct-2507", sft=True)


def mistral_7b_pretrain():
    return hf_trainer("mistralai/Mistral-7B-v0.1")


def mistral_7b_sft():
    return hf_trainer("mistralai/Mistral-7B-v0.1", sft=True)


def olmo_7b_pretrain():
    return hf_trainer("allenai/OLMo-7B-hf")


def olmo_7b_sft():
    return hf_trainer("allenai/OLMo-7B-hf", sft=True)


def qwen3_30b_pretrain():
    return hf_trainer("Qwen/Qwen3-30B-A3B", moe=True)


def qwen3_30b_sft():
    return hf_trainer("Qwen/Qwen3-30B-A3B", moe=True, sft=True)


def mixtral_8x7b_pretrain():
    return hf_trainer("mistralai/Mixtral-8x7B-v0.1", moe=True)


def mixtral_8x7b_sft():
    return hf_trainer("mistralai/Mixtral-8x7B-v0.1", moe=True, sft=True)


def olmoe_7b_pretrain():
    return hf_trainer("allenai/OLMoE-1B-7B-0924", moe=True)


def olmoe_7b_sft():
    return hf_trainer("allenai/OLMoE-1B-7B-0924", moe=True, sft=True)


def gemma4_26b_pretrain():
    return hf_trainer("google/gemma-4-26B-A4B", moe=True)


def gemma4_26b_sft():
    return hf_trainer("google/gemma-4-26B-A4B", moe=True, sft=True)


def deepseek_v2_lite_pretrain():
    return hf_trainer("deepseek-ai/DeepSeek-V2-Lite", moe=True)


def deepseek_v2_lite_sft():
    return hf_trainer("deepseek-ai/DeepSeek-V2-Lite", moe=True, sft=True)


def _glm5(cfg: Trainer.Config) -> Trainer.Config:
    # Upstream: TP/CP unsupported for GLM-5 DSA, FSDP+EP only (--ep 8 in dev.yaml).
    # Flight recorder off (trace_buf_size=0): long CPU-only init with no NCCL ops.
    cfg.comm.trace_buf_size = 0
    return cfg


def glm5_pretrain():
    return _glm5(hf_trainer("zai-org/GLM-5", moe=True))


def glm5_sft():
    return _glm5(hf_trainer("zai-org/GLM-5", moe=True, sft=True))

