"""Row-major layout for stacked linears, so FSDP can shard them unevenly.

TorchTitan stores a stacked projection (the FFN's fused W1/W3) as
``[num_linears, out_features, in_features]`` and tells FSDP to shard dim 1
(``out_features``) so every rank keeps a slice of both projections. FSDP2
only pads uneven shards on dim 0, so dim-1 sharding requires the shard group
size to divide ``out_features`` (17408 = 2**10 * 17 for Qwen3.8-27B): it
fails at 12, 28, 112, 280, 288 or 296 GPUs.

:class:`RowStackedLinear` stores the same weights as
``[out_features, num_linears, in_features]``. FSDP's default ``Shard(0)`` then
splits the 17408 rows (padded when uneven), each rank still holds a slice of
both projections, and the all-gather writes in place. The forward is the same
single fused matmul; only the order of the output features changes (gate/up
interleaved), and the output is returned with TorchTitan's
``[..., num_linears, out_features]`` shape as a strided view.

Tensor parallelism: TorchTitan's plan shards the stacked weight on F (dim 1
of ``[N, F, D]``); :func:`patch_tp_sharding` moves that to dim 0 of
``[F, N, D]``, so each TP rank holds a contiguous F slice of both projections,
the input rows of its ``w2`` (row-parallel) shard.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from functools import wraps

import torch
import torch.nn as nn

from torchtitan.models.common.linear import ColumnParallelLinear


class RowStackedLinear(ColumnParallelLinear):
    @dataclass(kw_only=True, slots=True)
    class Config(ColumnParallelLinear.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)
        if self.num_linears > 1:
            # [N, F, D] -> [F, N, D]
            self.weight = nn.Parameter(
                self.weight.detach().transpose(0, 1).contiguous(),
                requires_grad=self.weight.requires_grad,
            )
            if self.bias is not None:
                self.bias = nn.Parameter(
                    self.bias.detach().transpose(0, 1).contiguous(),
                    requires_grad=self.bias.requires_grad,
                )

    def _unflatten_output(self, output: torch.Tensor) -> torch.Tensor:
        if self.num_linears == 1:
            return output
        # [..., F * N] -> [..., F, N] -> [..., N, F] (view). F is the local
        # (TP-sharded) row count, as in Linear._unflatten_output.
        return output.unflatten(-1, self.weight.shape[:-1]).transpose(-1, -2)


def _transposed_init(init):
    """Run a ``[N, F, D]`` stacked-weight initializer on a ``[F, N, D]`` weight."""

    @wraps(init)
    def wrapped(weight: torch.Tensor) -> None:
        init(weight.transpose(0, 1))

    return wrapped


def use_row_stacked_linears(model_config) -> int:
    """Replace stacked ColumnParallelLinear configs in-place. Returns the count."""
    replaced = 0
    for _, cfg, parent, attr in list(model_config.traverse(ColumnParallelLinear.Config)):
        if cfg.num_linears <= 1 or isinstance(cfg, RowStackedLinear.Config):
            continue
        values = {f.name: getattr(cfg, f.name) for f in fields(cfg) if f.init}
        if values.get("param_init") and "weight" in values["param_init"]:
            values["param_init"] = {
                **values["param_init"],
                "weight": _transposed_init(values["param_init"]["weight"]),
            }
        new = RowStackedLinear.Config(**values)
        if isinstance(parent, list):
            parent[attr] = new
        else:
            setattr(parent, attr, new)
        replaced += 1
    return replaced


def patch_fsdp_placements() -> None:
    """Let FSDP use its default ``Shard(0)`` for row-stacked weights."""
    import torchtitan.distributed.fsdp as fsdp

    original = fsdp.linear_param_shard_placements
    if getattr(original, "_tt_native_row_stacked", False):
        return

    @wraps(original)
    def linear_param_shard_placements(module, **kwargs):
        placements = original(module, **kwargs)
        for child in module.modules():
            if isinstance(child, RowStackedLinear):
                placements.pop(child.weight, None)
                if child.bias is not None:
                    placements.pop(child.bias, None)
        return placements

    linear_param_shard_placements._tt_native_row_stacked = True
    fsdp.linear_param_shard_placements = linear_param_shard_placements


def patch_tp_sharding() -> None:
    """TP-shard row-stacked weights on dim 0 (``[F, N, D]``), not dim 1.

    Wraps ``Qwen35Model.Config.set_sharding_`` (called by the Trainer with the
    parallelism config): the stacked column-parallel plan declares the weight
    ``Shard(1)`` on TP, which for ``[F, N, D]`` would split the N projections.
    Activations keep the plan's layout: the output is ``[..., N, F_local]``.
    """
    import dataclasses

    import spmd_types as spmd
    from torchtitan.models.common.decoder_sharding import dense_param_placement
    from torchtitan.models.qwen3_5.model import Qwen35Model

    original = Qwen35Model.Config.set_sharding_
    if getattr(original, "_tt_native_row_stacked", False):
        return

    @wraps(original)
    def set_sharding_(self, parallelism) -> None:
        original(self, parallelism)
        for _, cfg, _, _ in self.traverse(RowStackedLinear.Config):
            sharding = cfg.sharding_config
            if sharding is None:
                continue
            states = dict(sharding.state_shardings)
            for name in ("weight", "bias"):
                if name in states:
                    states[name] = dense_param_placement(tp=spmd.S(0))
            cfg.sharding_config = dataclasses.replace(sharding, state_shardings=states)

    set_sharding_._tt_native_row_stacked = True
    Qwen35Model.Config.set_sharding_ = set_sharding_
