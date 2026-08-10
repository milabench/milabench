"""PyTorch / torchtitan / voir shims for ROCm milabench runs."""

from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
import inspect
import os
import time
from typing import Callable, Optional, Tuple, Union


def ensure_voir_pipe_compat() -> None:
    """Avoid hung torchrun teardown when voir logger close hits broken pipes."""
    try:
        from voir.overseer import JsonlFileLogger
    except ImportError:
        return

    if getattr(JsonlFileLogger.close, "_milabench_compat", False):
        return

    _orig_close = JsonlFileLogger.close

    def close(self):
        try:
            _orig_close(self)
        except OSError as exc:
            if getattr(exc, "errno", None) not in (22, 32):
                raise

    close._milabench_compat = True  # type: ignore[attr-defined]
    JsonlFileLogger.close = close


def _patch_fsdp_uniform_dtype() -> None:
    """Cast floating params to FSDP param_dtype; mixed bf16/fp32 breaks fully_shard."""
    import torch

    import torchtitan.experiments.transformers_modeling_backend.parallelize as par

    if getattr(par.apply_fsdp, "_milabench_compat", False):
        return

    _orig = par.apply_fsdp

    @wraps(_orig)
    def apply_fsdp(model, dp_mesh, param_dtype, *args, **kwargs):
        for param in model.parameters():
            if param.dtype.is_floating_point and param.dtype != param_dtype:
                param.data = param.data.to(dtype=param_dtype)
        return _orig(model, dp_mesh, param_dtype, *args, **kwargs)

    apply_fsdp._milabench_compat = True  # type: ignore[attr-defined]
    par.apply_fsdp = apply_fsdp


def _patch_create_block_mask() -> None:
    """Strip kwargs torchtitan passes that older ROCm torch flex_attention lacks."""
    from torch.nn.attention import flex_attention

    orig = flex_attention.create_block_mask
    if getattr(orig, "_milabench_compat", False):
        return

    supported = set(inspect.signature(orig).parameters)

    @wraps(orig)
    def create_block_mask_compat(*args, **kwargs):
        filtered = {k: v for k, v in kwargs.items() if k in supported}
        return orig(*args, **filtered)

    create_block_mask_compat._milabench_compat = True  # type: ignore[attr-defined]
    flex_attention.create_block_mask = create_block_mask_compat


def _patch_torchtitan_create_attention_mask() -> None:
    """Rebind torchtitan's flex mask helper to the patched eager API."""
    import torchtitan.models.common.attention as attn

    if getattr(attn.create_attention_mask, "_milabench_compat", False):
        return

    from torch.nn.attention.flex_attention import create_block_mask

    attn._compiled_create_block_mask = create_block_mask

    @wraps(attn.create_attention_mask)
    def create_attention_mask(*args, **kwargs):
        kwargs.pop("separate_full_blocks", None)
        return attn._compiled_create_block_mask(*args, **kwargs)

    create_attention_mask._milabench_compat = True  # type: ignore[attr-defined]
    attn.create_attention_mask = create_attention_mask


def _patch_transformers_flex_attention() -> None:
    """ROCm triton/inductor compile often fails; use eager flex_attention."""
    try:
        from transformers.integrations import flex_attention as fa
        from torch.nn.attention.flex_attention import flex_attention
    except ImportError:
        return

    if getattr(fa.compile_friendly_flex_attention, "_milabench_compat", False):
        return

    @wraps(fa.compile_friendly_flex_attention)
    def compile_friendly_flex_attention(query, key, value, training=False, **kwargs):
        return flex_attention(query, key, value, **kwargs)

    compile_friendly_flex_attention._milabench_compat = True  # type: ignore[attr-defined]
    fa.compile_friendly_flex_attention = compile_friendly_flex_attention

    if hasattr(fa, "WrappedFlexAttention"):
        fa.WrappedFlexAttention._compiled_flex_attention = flex_attention
        fa.WrappedFlexAttention._is_flex_compiled = True


def _move_titan_module_to_cpu(module) -> None:
    """Keep EP-sharded MoE on CPU during build; trainer moves to GPU later."""
    import torch
    import torch.nn as nn
    from torch.distributed.tensor import DTensor

    for name, param in list(module.named_parameters(recurse=False)):
        if isinstance(param, DTensor):
            local = param.to_local()
            if local.device.type == "cpu":
                continue
            module.register_parameter(
                name,
                nn.Parameter(
                    DTensor.from_local(
                        local.cpu(),
                        param.device_mesh,
                        param.placements,
                        run_check=False,
                    ),
                    requires_grad=param.requires_grad,
                ),
            )
        elif param.device.type != "cpu":
            param.data = param.data.cpu()

    for name, buf in list(module.named_buffers(recurse=False)):
        if buf is None:
            continue
        if isinstance(buf, DTensor):
            local = buf.to_local()
            if local.device.type == "cpu":
                continue
            persistent = name not in module._non_persistent_buffers_set
            module.register_buffer(
                name,
                DTensor.from_local(
                    local.cpu(),
                    buf.device_mesh,
                    buf.placements,
                    run_check=False,
                ),
                persistent=persistent,
            )
        elif buf.device.type != "cpu":
            module._buffers[name] = buf.cpu()

    for child in module.children():
        _move_titan_module_to_cpu(child)


def _patch_module_parallelize_idempotent() -> None:
    """Allow model.parallelize() to skip MoE blocks already EP-sharded in build."""
    from torchtitan.protocols.module import Module

    if getattr(Module.parallelize, "_milabench_idempotent", False):
        return

    _orig = Module.parallelize

    @wraps(_orig)
    def parallelize(self, parallel_dims):
        if self._parallelized:
            return
        return _orig(self, parallel_dims)

    parallelize._milabench_idempotent = True  # type: ignore[attr-defined]
    Module.parallelize = parallelize


def _patch_moe_build_device() -> None:
    """GLM-5: CPU MoE init + immediate EP shard to cap peak memory.

    GPU init OOMs (~277GiB) stacking full expert sets before model.parallelize().
    Upstream CPU init keeps all 78 full-width layers until a single parallelize pass.
    Shard each layer right after build so peak stays ~O(one layer) not ~O(all layers).
    """
    import gc

    import torch
    import torch.distributed as dist

    import torchtitan.experiments.transformers_modeling_backend.moe_replacement as mr
    from torchtitan.models.common.feed_forward import SigmoidGatedFeedForward
    from torchtitan.protocols.sharding import ShardingConfig
    from torchtitan.tools.logging import logger

    if getattr(mr.build_and_swap_native_moe, "_milabench_incremental_ep_v2", False):
        return

    _get_expert_param_info = mr._get_expert_param_info
    _get_moe_attr_name = mr._get_moe_attr_name

    @wraps(mr.build_and_swap_native_moe)
    def build_and_swap_native_moe(model, parallel_dims):
        import spmd_types as spmd
        from torchtitan.models.common.decoder_sharding import (
            dense_activation_placement,
            dense_param_placement,
        )
        from torchtitan.models.common.moe_sharding import set_moe_sharding_config

        enable_ep = parallel_dims.ep_enabled
        enable_sp = parallel_dims.tp_enabled
        cpu = torch.device("cpu")
        rank = dist.get_rank() if dist.is_initialized() else 0

        moe_layers = [
            layer
            for layer in model.layers.values()
            if getattr(layer, "_native_moe_config", None) is not None
        ]
        total = len(moe_layers)
        print(
            f"[milabench][rank{rank}] build_and_swap_native_moe: "
            f"{total} MoE layers on CPU (incremental EP)",
            flush=True,
        )

        for layer_idx, layer in enumerate(moe_layers, start=1):
            moe_config = layer._native_moe_config
            t0 = time.time()

            _, expert_layout = _get_expert_param_info()
            set_moe_sharding_config(
                moe_config,
                enable_ep=enable_ep,
                enable_sp=enable_sp,
                expert_param_layout=expert_layout,
            )

            shared = moe_config.shared_experts
            if isinstance(shared, SigmoidGatedFeedForward.Config):
                shared.gate.sharding_config = ShardingConfig(
                    state_shardings={
                        "weight": dense_param_placement(tp=spmd.R),
                        "bias": dense_param_placement(tp=spmd.R),
                    },
                    out_dst_shardings=dense_activation_placement(tp=spmd.R),
                )

            with torch.device("meta"):
                native_moe = moe_config.build()

            native_moe.to_empty(device=cpu)
            native_moe.init_states(buffer_device=cpu)

            moe_attr = _get_moe_attr_name(layer)
            setattr(layer, moe_attr, native_moe)
            object.__setattr__(layer, "moe", native_moe)

            if getattr(layer, "_layer_level_moe", False):
                if hasattr(layer, "enable_moe_block"):
                    layer.enable_moe_block = False
                if hasattr(layer, "router"):
                    delattr(layer, "router")
                if hasattr(layer, "experts"):
                    delattr(layer, "experts")
                for norm_name in (
                    "pre_feedforward_layernorm_2",
                    "post_feedforward_layernorm_1",
                    "post_feedforward_layernorm_2",
                ):
                    if hasattr(layer, norm_name):
                        delattr(layer, norm_name)

            del layer._native_moe_config
            with torch.device("cpu"):
                native_moe.parallelize(parallel_dims)
            _move_titan_module_to_cpu(native_moe)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(
                f"[milabench][rank{rank}] MoE layer {layer_idx}/{total} "
                f"built+EP-sharded on CPU (+{time.time() - t0:.1f}s)",
                flush=True,
            )

        logger.info("Built and swapped Titan MoE modules into the model")

    build_and_swap_native_moe._milabench_incremental_ep_v2 = True  # type: ignore[attr-defined]
    mr.build_and_swap_native_moe = build_and_swap_native_moe


def _patch_parallelize_progress() -> None:
    """Log GLM-5 parallelize milestones; helps debug long CPU-only init hangs."""
    import torch.distributed as dist

    import torchtitan.experiments.transformers_modeling_backend.parallelize as par

    if getattr(par.parallelize_hf_transformers, "_milabench_progress", False):
        return

    _orig = par.parallelize_hf_transformers

    @wraps(_orig)
    def parallelize_hf_transformers(model, **kwargs):
        rank = dist.get_rank() if dist.is_initialized() else -1
        t0 = time.time()

        def log(step: str) -> None:
            print(
                f"[milabench][rank{rank}] parallelize {step} (+{time.time() - t0:.1f}s)",
                flush=True,
            )

        log("start")
        result = _orig(model, **kwargs)
        log("done")
        return result

    parallelize_hf_transformers._milabench_progress = True  # type: ignore[attr-defined]
    par.parallelize_hf_transformers = parallelize_hf_transformers


def _ensure_nccl_init_env() -> None:
    """Defaults for long meta-device parallelize on multi-rank jobs."""
    os.environ.setdefault("TORCH_NCCL_ENABLE_MONITORING", "0")
    os.environ.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "14400")


def _patch_distributed_set_timeout() -> None:
    """Backfill torch.distributed.set_timeout expected by torchtitan trainer."""
    import torch.distributed as dist
    from torch.distributed.distributed_c10d import _get_default_group

    if hasattr(dist, "set_timeout"):
        return

    def set_timeout(timeout, group=None):
        pg = _get_default_group() if group is None else group
        if pg is None:
            return
        try:
            pg.set_timeout(timeout)
        except RuntimeError as exc:
            msg = str(exc).lower()
            if "does not support" in msg and "timeout" in msg:
                return
            raise

    set_timeout._milabench_compat = True  # type: ignore[attr-defined]
    dist.set_timeout = set_timeout


def ensure_torchtitan_torch_compat() -> None:
    """Backfill torch.distributed.fsdp APIs expected by upstream torchtitan."""
    _ensure_nccl_init_env()
    ensure_voir_pipe_compat()
    _patch_distributed_set_timeout()

    import torch.distributed.fsdp as fsdp_mod

    if not hasattr(fsdp_mod, "DataParallelMeshDims"):

        @dataclass(frozen=True)
        class DataParallelMeshDims:
            shard: Union[str, Tuple[str, ...], None] = None
            replicate: Optional[str] = None

        fsdp_mod.DataParallelMeshDims = DataParallelMeshDims

    orig_fully_shard: Callable = fsdp_mod.fully_shard
    if not getattr(orig_fully_shard, "_milabench_compat", False):

        @wraps(orig_fully_shard)
        def fully_shard_compat(module, *args, **kwargs):
            kwargs.pop("dp_mesh_dims", None)
            return orig_fully_shard(module, *args, **kwargs)

        fully_shard_compat._milabench_compat = True  # type: ignore[attr-defined]
        fsdp_mod.fully_shard = fully_shard_compat

    _patch_create_block_mask()
    _patch_transformers_flex_attention()
    _patch_torchtitan_create_attention_mask()
    _patch_fsdp_uniform_dtype()
    _patch_module_parallelize_idempotent()
    _patch_moe_build_device()
    _patch_parallelize_progress()
