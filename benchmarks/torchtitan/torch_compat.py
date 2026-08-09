"""PyTorch / torchtitan / voir shims for ROCm milabench runs."""

from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
import inspect
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
