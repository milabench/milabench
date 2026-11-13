"""PyTorch / transformers / voir shims for milabench torchtitan runs."""

from __future__ import annotations

import inspect
import os
from functools import wraps


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


def _use_eager_flex() -> bool:
    """Eager flex attention: fallback for Triton compile failures (``MILABENCH_TT_EAGER_FLEX=1``).

    Much slower than the compiled kernel, so off unless forced.
    """
    return os.environ.get("MILABENCH_TT_EAGER_FLEX") == "1"


def _patch_transformers_flex_attention() -> None:
    """Make transformers' flex attention call the eager kernel."""
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


def _ensure_nccl_init_env() -> None:
    """Defaults for long meta-device parallelize on multi-rank jobs."""
    os.environ.setdefault("TORCH_NCCL_ENABLE_MONITORING", "0")
    os.environ.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "14400")


def _patch_heterogeneous_config_access() -> None:
    """Allow global reads of per-layer HF config attributes (e.g. Gemma4 num_key_value_heads).

    The transformers modeling backend copies every attribute of the HF config;
    heterogeneous configs raise on those reads. The HF modeling code still reads
    the per-layer values; the copied globals only feed torchtitan bookkeeping.
    """
    try:
        from transformers.integrations.heterogeneity.configuration_utils import (
            HeterogeneousConfigMixin,
        )
    except ImportError:
        return

    prop = HeterogeneousConfigMixin.allow_global_per_layer_attribute_access
    if getattr(prop.fget, "_milabench_compat", False):
        return

    def fget(self) -> bool:
        return self.__dict__.get("allow_global_per_layer_attribute_access", True)

    fget._milabench_compat = True  # type: ignore[attr-defined]
    HeterogeneousConfigMixin.allow_global_per_layer_attribute_access = prop.getter(fget)


# Attention projections hf_sharding inspects with hasattr().
_HF_ATTN_PROJECTIONS = (
    "q_proj", "q_a_proj", "q_a_layernorm", "q_b_proj",
    "kv_a_proj_with_mqa", "kv_a_layernorm", "kv_b_proj",
    "k_proj", "v_proj", "o_proj", "dense",
)


def _patch_hf_sharding_absent_projections() -> None:
    """Treat attention projections set to None as absent in TorchTitan's HF sharding.

    transformers >= 5.10 keeps absent projections as None attributes
    (DeepSeek-V2-Lite without low-rank Q: q_a_proj/q_b_proj; Gemma4 global
    layers, K=V: v_proj), while hf_sharding (written for 5.9) checks
    ``hasattr``. Hide them during the call; a K=V layer gets a stand-in
    v_proj so the k_proj/v_proj branch applies (its config is discarded).
    """
    import torch.nn as nn

    import torchtitan.experiments.transformers_modeling_backend.hf_sharding as hs

    orig = hs._set_layer_sharding_configs
    if getattr(orig, "_milabench_compat", False):
        return

    @wraps(orig)
    def _set_layer_sharding_configs(layer, *args, **kwargs):
        attn = getattr(layer, "self_attn", None)
        hidden = {}
        if attn is not None:
            for name in _HF_ATTN_PROJECTIONS:
                for store in (attn._modules, attn.__dict__):
                    if name in store and store[name] is None:
                        hidden[name] = (store, store.pop(name))
            if "v_proj" in hidden and getattr(attn, "k_proj", None) is not None:
                attn._modules["v_proj"] = nn.Identity()
        try:
            return orig(layer, *args, **kwargs)
        finally:
            if attn is not None:
                if "v_proj" in hidden:
                    attn._modules.pop("v_proj", None)
                for name, (store, value) in hidden.items():
                    store[name] = value

    _set_layer_sharding_configs._milabench_compat = True  # type: ignore[attr-defined]
    hs._set_layer_sharding_configs = _set_layer_sharding_configs

    # The embedding config names nn.Embedding.forward's argument ("input");
    # subclasses such as Gemma4TextScaledWordEmbedding name it "input_ids".
    orig_root = hs.set_hf_sharding_configs

    @wraps(orig_root)
    def set_hf_sharding_configs(model, *args, **kwargs):
        result = orig_root(model, *args, **kwargs)
        emb = getattr(model, "tok_embeddings", None)
        config = getattr(emb, "_sharding_config", None)
        if config is not None:
            arg = next(iter(inspect.signature(emb.forward).parameters))
            if arg != "input":
                for field in ("in_src_shardings", "in_dst_shardings"):
                    shardings = getattr(config, field)
                    if shardings and "input" in shardings:
                        shardings[arg] = shardings.pop("input")
        return result

    hs.set_hf_sharding_configs = set_hf_sharding_configs


def ensure_torchtitan_torch_compat() -> None:
    """Apply the shims; call before torchtitan builds the trainer."""
    _ensure_nccl_init_env()
    ensure_voir_pipe_compat()
    _patch_heterogeneous_config_access()
    _patch_hf_sharding_absent_projections()
    if _use_eager_flex():
        _patch_transformers_flex_attention()
