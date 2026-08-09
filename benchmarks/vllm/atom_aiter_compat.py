"""Patch amd_aiter gaps needed by ATOM main (Kimi-K3, MoE shuffles)."""

from __future__ import annotations

import sys
import types


def ensure_atom_aiter_mla_module() -> None:
    """Provide ``aiter.ops.triton.attention.mla`` when the wheel omits it."""
    name = "aiter.ops.triton.attention.mla"
    if name in sys.modules:
        return
    try:
        __import__(name)
        return
    except ModuleNotFoundError:
        pass

    from aiter.mla import mla_decode_fwd, mla_prefill_fwd

    mod = types.ModuleType(name)
    mod.mla_decode_fwd = mla_decode_fwd
    mod.mla_prefill_fwd = mla_prefill_fwd
    sys.modules[name] = mod


def ensure_atom_aiter_shuffle_compat() -> None:
    """Backfill shuffle helpers added after aiter 0.1.17-rc0."""
    import torch

    import aiter.ops.shuffle as shuffle

    if not hasattr(shuffle, "interleave_gate_up_rows"):

        def interleave_gate_up_rows(w: torch.Tensor) -> torch.Tensor:
            inter = w.shape[1] // 2
            return (
                torch.stack([w[:, :inter], w[:, inter:]], dim=2)
                .flatten(1, 2)
                .contiguous()
            )

        shuffle.interleave_gate_up_rows = interleave_gate_up_rows

    if not hasattr(shuffle, "moe_shuffle_weight"):
        from aiter.jit.utils.chip_info import get_gfx

        def moe_shuffle_weight(
            src: torch.Tensor,
            experts_cnt: int | None = None,
            is_guinterleave: bool = False,
            gate_up: bool = False,
            layout=(16, 16),
        ) -> torch.Tensor:
            del experts_cnt
            if get_gfx() == "gfx1250":
                if is_guinterleave and gate_up:
                    src = shuffle.interleave_gate_up_rows(src)
                return shuffle.shuffle_weight(src, layout=layout)
            return shuffle.shuffle_weight(
                src,
                layout=layout,
                is_guinterleave=is_guinterleave,
                gate_up=gate_up,
            )

        shuffle.moe_shuffle_weight = moe_shuffle_weight


def ensure_atom_aiter_compat() -> None:
    from patch_aiter_for_atom import patch_aiter_shuffle

    patch_aiter_shuffle()
    ensure_atom_aiter_mla_module()
    ensure_atom_aiter_shuffle_compat()
