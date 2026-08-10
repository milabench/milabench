"""Job configs for milabench torchtitan benches (pretrain + SFT).

Loaded by torchtitan ConfigManager via ``--module tt_configs --config <name>``.
Paths resolve from MILABENCH_DIR_* at parse time.
"""

from __future__ import annotations

import os
from pathlib import Path

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import (
    default_adamw,
    register_moe_load_balancing_hook,
)
from torchtitan.config import DebugConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.experiments.transformers_modeling_backend import (
    TitanModelConfig,
    TitanMoeModelConfig,
)
from torchtitan.experiments.transformers_modeling_backend.configs import (
    TransformersBackendConfig,
)
from torchtitan.experiments.transformers_modeling_backend.model import (
    HFTransformerModel,
)
from torchtitan.experiments.transformers_modeling_backend.parallelize import (
    parallelize_hf_transformers,
)
from torchtitan.experiments.transformers_modeling_backend.pipeline import (
    pipeline_hf_transformers,
)
from torchtitan.experiments.transformers_modeling_backend.state_dict_adapter import (
    HFTransformerStateDictAdapter,
)
from .state_dict_adapter import MilabenchMoeStateDictAdapter
from torchtitan.experiments.transformers_modeling_backend.tokenizer import (
    HFBackendTokenizer,
)
from torchtitan.hf_datasets.text_datasets import (
    ChatDataLoader,
    HuggingFaceTextDataLoader,
)
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.tools.profiler import Profiler


def _data_dir() -> Path:
    for key in ("MILABENCH_DIR_DATA",):
        if os.environ.get(key):
            return Path(os.environ[key])
    base = os.environ.get("MILABENCH_BASE")
    if base:
        return Path(base) / "data"
    # Fallback for this machine's shared milabench base.
    shared = Path("/data/results/data")
    if shared.exists():
        return shared
    return Path(os.environ.get("MILABENCH_DIR_DATA", "."))


def _code_dir() -> Path:
    return Path(
        os.environ.get("MILABENCH_DIR_CODE", Path(__file__).resolve().parents[1])
    )


def _extra_dir() -> Path:
    return Path(os.environ.get("MILABENCH_DIR_EXTRA", "./outputs"))


def _hf_assets(model_id: str) -> str:
    """Local dir written by prepare.py: ``{data}/hf/<model_name>``."""
    name = model_id.split("/")[-1]
    return str(_data_dir() / "hf" / name)


def _c4_path() -> str:
    bundled = _code_dir() / "assets" / "c4_test"
    staged = _data_dir() / "c4_test"
    return str(staged if staged.exists() else bundled)


def _sft_json() -> str:
    bundled = _code_dir() / "assets" / "sft_test" / "data.json"
    staged = _data_dir() / "sft_test" / "data.json"
    return str(staged if staged.exists() else bundled)


def _process_sft_sample(sample):
    return [
        {"role": "user", "content": sample["question"]},
        {"role": "assistant", "content": sample["answer"]},
    ]


def _model_spec(flavor: str) -> ModelSpec:
    flavors = {
        "dense": HFTransformerModel.Config(model_config=TitanModelConfig()),
        "dense_sft": HFTransformerModel.Config(
            model_config=TitanModelConfig(attn_mask_type="block_causal"),
        ),
        # Empty MoE overrides → HF AutoConfig supplies expert dims (Qwen3-MoE, GLM-5, …).
        "moe": HFTransformerModel.Config(model_config=TitanMoeModelConfig()),
        "moe_sft": HFTransformerModel.Config(
            model_config=TitanMoeModelConfig(attn_mask_type="block_causal"),
        ),
    }
    adapter = (
        MilabenchMoeStateDictAdapter
        if flavor in ("moe", "moe_sft")
        else HFTransformerStateDictAdapter
    )
    return ModelSpec(
        name="transformers_modeling_backend",
        flavor=flavor,
        model=flavors[flavor],
        parallelize_fn=parallelize_hf_transformers,
        pipelining_fn=pipeline_hf_transformers,
        post_optimizer_build_fn=register_moe_load_balancing_hook,
        state_dict_adapter=adapter,
    )


def _base_pretrain(
    *,
    hf_model: str,
    flavor: str,
    local_batch_size: int = 1,
    seq_len: int = 2048,
    steps: int = 40,
) -> TransformersBackendConfig:
    return TransformersBackendConfig(
        loss=CrossEntropyLoss.Config(),
        hf_assets_path=_hf_assets(hf_model),
        hf_model=hf_model,
        dump_folder=str(_extra_dir() / "torchtitan"),
        debug=DebugConfig(print_config=True),
        model_spec=_model_spec(flavor),
        profiler=Profiler.Config(enable_profiling=False),
        optimizer=default_adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=local_batch_size,
            seq_len=seq_len,
            steps=steps,
            dtype="bfloat16",
            mixed_precision_param="bfloat16",
        ),
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4_test",
            dataset_path=_c4_path(),
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        parallelism=ParallelismConfig(pipeline_parallel_schedule="1F1B"),
        checkpoint=CheckpointManager.Config(enable=False),
        activation_checkpoint=SelectiveAC.Config(),
        tokenizer=HFBackendTokenizer.Config(),
    )


def _base_sft(
    *,
    hf_model: str,
    flavor: str,
    local_batch_size: int = 1,
    seq_len: int = 1024,
    steps: int = 40,
) -> TransformersBackendConfig:
    assets = _hf_assets(hf_model)
    return TransformersBackendConfig(
        loss=CrossEntropyLoss.Config(),
        hf_assets_path=assets,
        hf_model=hf_model,
        dump_folder=str(_extra_dir() / "torchtitan"),
        debug=DebugConfig(print_config=True),
        model_spec=_model_spec(flavor),
        profiler=Profiler.Config(enable_profiling=False),
        optimizer=default_adamw(lr=2e-5),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=local_batch_size,
            seq_len=seq_len,
            steps=steps,
            dtype="bfloat16",
            mixed_precision_param="bfloat16",
        ),
        dataloader=ChatDataLoader.Config(
            dataset_path="json",
            load_dataset_kwargs={
                "data_files": _sft_json(),
                "split": "train",
            },
            sample_processor=_process_sft_sample,
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        parallelism=ParallelismConfig(pipeline_parallel_schedule="1F1B"),
        checkpoint=CheckpointManager.Config(
            enable=True,
            initial_load_in_hf=True,
            initial_load_model_only=True,
            initial_load_path=assets,
            interval=10_000,
            last_save_model_only=True,
        ),
        activation_checkpoint=SelectiveAC.Config(),
        tokenizer=HFBackendTokenizer.Config(),
    )


def qwen3_4b_pretrain() -> TransformersBackendConfig:
    return _base_pretrain(
        hf_model="Qwen/Qwen3-4B-Instruct-2507",
        flavor="dense",
    )


def qwen3_4b_sft() -> TransformersBackendConfig:
    return _base_sft(
        hf_model="Qwen/Qwen3-4B-Instruct-2507",
        flavor="dense_sft",
    )


def qwen3_30b_pretrain() -> TransformersBackendConfig:
    return _base_pretrain(
        hf_model="Qwen/Qwen3-30B-A3B",
        flavor="moe",
    )


def qwen3_30b_sft() -> TransformersBackendConfig:
    return _base_sft(
        hf_model="Qwen/Qwen3-30B-A3B",
        flavor="moe_sft",
    )


def _glm5_parallelism(cfg: TransformersBackendConfig) -> TransformersBackendConfig:
    # Upstream: TP/CP unsupported for GLM-5 DSA — FSDP+EP only.
    cfg.parallelism.tensor_parallel_degree = 1
    cfg.parallelism.context_parallel_degree = 1
    cfg.parallelism.expert_parallel_degree = 8
    # dp_shard stays -1 (auto → world_size on 8 GPUs); do not set to 1 with EP.
    # 744B MoE init/parallelize + SFT checkpoint load can take 30–60+ minutes.
    cfg.comm.init_timeout_seconds = 7200
    cfg.comm.train_timeout_seconds = 1800
    # Flight recorder defaults (trace_buf_size=20000) enable extra NCCL
    # monitoring; disable for long CPU-only GLM-5 init.
    cfg.comm.trace_buf_size = 0
    # MoE build uses CPU + per-layer EP shard (torch_compat); trainer still
    # materializes EP/FSDP-local shards on GPU after parallelize_fn returns.
    return cfg


def glm5_pretrain() -> TransformersBackendConfig:
    cfg = _base_pretrain(
        hf_model="zai-org/GLM-5",
        flavor="moe",
        seq_len=1024,
    )
    return _glm5_parallelism(cfg)


def glm5_sft() -> TransformersBackendConfig:
    cfg = _base_sft(
        hf_model="zai-org/GLM-5",
        flavor="moe_sft",
        seq_len=1024,
    )
    return _glm5_parallelism(cfg)
