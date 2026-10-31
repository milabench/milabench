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

Not compatible with tensor parallelism (the TP plan shards dim -2).
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
        # [..., F * N] -> [..., F, N] -> [..., N, F] (view)
        return output.unflatten(-1, (self.out_features, self.num_linears)).transpose(
            -1, -2
        )


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
