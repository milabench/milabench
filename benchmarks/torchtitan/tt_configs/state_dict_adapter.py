"""Load Hugging Face checkpoints into transformers-backend models, MoE included.

TorchTitan's HFTransformerStateDictAdapter only strips the ``model.`` prefix:
it cannot load MoE checkpoints, since the backend swaps HF MoE blocks for
TorchTitan's grouped experts (``routed_experts.w13`` [E, 2, F, D],
``routed_experts.w2`` [E, D, F], ``router.gate``, ``shared_experts.w13``
[2, F, D] / ``w2``), and checkpoints spell experts per model:

* per expert: ``mlp.experts.<e>.{gate,up,down}_proj`` (OLMoE, DeepSeek, Qwen3-MoE)
  or ``block_sparse_moe.experts.<e>.{w1,w3,w2}`` (Mixtral);
* packed: ``experts.gate_up_proj`` [E, 2F, D] / ``experts.down_proj`` (Gemma4).

``to_hf`` maps every model tensor to the key the checkpoint stores (from its
``model.safetensors.index.json``) and returns load targets that are views of
the rank's own parameter shards: a rank requests only the experts it holds,
and DCP writes them in place (no gathered copy of the model). ``from_hf``
returns the model state dict those views alias.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import torch
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

from torchtitan.experiments.transformers_modeling_backend.state_dict_adapter import (
    HFTransformerStateDictAdapter,
)

# TorchTitan-only state with no checkpoint counterpart: initialized by the model.
_TITAN_ONLY = re.compile(r"\.(expert_bias_E|tokens_per_expert_E)$")

_MOE_KEY = re.compile(
    r"^(?P<pre>.*layers\.\d+\.)(?P<attr>[^.]+)\."
    r"(?P<part>routed_experts\.w13|routed_experts\.w2|shared_experts\.w13|shared_experts\.w2|router\.gate)"
    r"\.weight$"
)

# Checkpoint spellings, tried in order; {p}: layer prefix, {a}: MoE module name, {e}: expert.
_ROUTED_PER_EXPERT = [
    ("{p}{a}.experts.{e}.gate_proj.weight", "{p}{a}.experts.{e}.up_proj.weight", "{p}{a}.experts.{e}.down_proj.weight"),
    ("{p}block_sparse_moe.experts.{e}.w1.weight", "{p}block_sparse_moe.experts.{e}.w3.weight", "{p}block_sparse_moe.experts.{e}.w2.weight"),
]
_ROUTED_PACKED = [
    ("{p}{a}.experts.gate_up_proj", "{p}{a}.experts.down_proj"),
    ("{p}experts.gate_up_proj", "{p}experts.down_proj"),
]
_SHARED = [
    ("{p}{a}.shared_experts.gate_proj.weight", "{p}{a}.shared_experts.up_proj.weight", "{p}{a}.shared_experts.down_proj.weight"),
    ("{p}{a}.shared_expert.gate_proj.weight", "{p}{a}.shared_expert.up_proj.weight", "{p}{a}.shared_expert.down_proj.weight"),
    # Gemma4: the dense MLP beside the experts becomes TorchTitan's shared expert.
    ("{p}{a}.gate_proj.weight", "{p}{a}.up_proj.weight", "{p}{a}.down_proj.weight"),
]
_ROUTER = ["{p}{a}.gate.weight", "{p}block_sparse_moe.gate.weight", "{p}router.proj.weight", "{p}{a}.router.weight"]

# Where the checkpoint keeps the decoder (composite VL models: Gemma4).
_PREFIXES = ("model.", "model.language_model.", "language_model.model.")


def _local_rows(value: torch.Tensor) -> tuple[torch.Tensor, int]:
    """(local tensor, global index of its first row) for a tensor sharded on dim 0 only."""
    if not isinstance(value, DTensor):
        return value, 0
    local = value.to_local()
    shape, offset = compute_local_shape_and_global_offset(
        value.shape, value.device_mesh, value.placements
    )
    if any(o != 0 for o in offset[1:]) or tuple(shape[1:]) != tuple(value.shape[1:]):
        raise NotImplementedError(
            f"HF load of a tensor sharded beyond dim 0 ({value.placements}): "
            "not supported by the milabench adapter"
        )
    return local, offset[0]


class MilabenchHFStateDictAdapter(HFTransformerStateDictAdapter):
    def __init__(self, model_config, hf_assets_path: str | None):
        super().__init__(model_config, hf_assets_path)
        self._model_state: dict[str, Any] | None = None
        self._disk_keys: set[str] = set()
        index = os.path.join(hf_assets_path or "", "model.safetensors.index.json")
        if hf_assets_path and os.path.isfile(index):
            with open(index, encoding="utf-8") as f:
                self._disk_keys = set(json.load(f)["weight_map"])

    def _resolve(self, key: str) -> str | None:
        """Checkpoint spelling of an HF-style key (``model.layers...``, ``lm_head...``), or None."""
        bare = key.removeprefix("model.")
        for candidate in (*(prefix + bare for prefix in _PREFIXES), bare):
            if candidate in self._disk_keys:
                return candidate
        return None

    def _first(self, templates, **fmt) -> tuple[str, ...] | None:
        for group in templates:
            keys = tuple(self._resolve(t.format(**fmt)) for t in group)
            if all(keys):
                return keys
        return None

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        if not self._disk_keys:
            return super().to_hf(state_dict)
        self._model_state = state_dict
        targets: dict[str, Any] = {}
        missing = []
        for key, value in state_dict.items():
            if _TITAN_ONLY.search(key):
                continue
            hf_key = key.removeprefix("model.")
            moe = _MOE_KEY.match(hf_key)
            if moe is None:
                disk = self._resolve(hf_key)
                if disk is None and hf_key == "lm_head.weight" and self._tie_word_embeddings:
                    continue  # tied: loaded through the embedding
                if disk is None:
                    missing.append(key)
                else:
                    targets[disk] = value
                continue
            if not self._moe_targets(moe, value, targets):
                missing.append(key)
        if missing:
            raise KeyError(
                f"{len(missing)} model tensors have no counterpart in the checkpoint "
                f"(first: {missing[:5]})"
            )
        return targets

    def _moe_targets(self, moe: re.Match, value: torch.Tensor, targets: dict[str, Any]) -> bool:
        fmt = {"p": moe["pre"], "a": moe["attr"]}
        part = moe["part"]

        if part == "router.gate":
            disk = self._first([[t] for t in _ROUTER], **fmt)
            if disk:
                targets[disk[0]] = value
            return disk is not None

        if part.startswith("shared_experts"):
            disk = self._first(_SHARED, **fmt)
            if disk is None:
                return False
            gate, up, down = disk
            if part.endswith("w2"):
                targets[down] = value
                return True
            # [2, F, D]: one row per projection (FSDP may give a rank one, both or none).
            local, start = _local_rows(value)
            for i in range(local.shape[0]):
                targets[(gate, up)[start + i]] = local[i]
            return True

        packed = self._first(_ROUTED_PACKED, **fmt)
        if packed is not None:
            gate_up, down = packed
            if part.endswith("w2"):
                targets[down] = value
            elif isinstance(value, DTensor):
                # [E, 2, F, D] and the checkpoint's [E, 2F, D] share one memory layout.
                local = value.to_local()
                shape = (value.shape[0], value.shape[1] * value.shape[2], value.shape[3])
                targets[gate_up] = DTensor.from_local(
                    local.flatten(1, 2), value.device_mesh, value.placements,
                    run_check=False, shape=torch.Size(shape),
                    stride=torch.empty(shape, device="meta").stride(),
                )
            else:
                targets[gate_up] = value.flatten(1, 2)
            return True

        local, start = _local_rows(value)
        for i in range(local.shape[0]):
            disk = self._first(_ROUTED_PER_EXPERT, e=start + i, **fmt)
            if disk is None:
                return False
            gate, up, down = disk
            if part.endswith("w2"):
                targets[down] = local[i]
            else:
                targets[gate], targets[up] = local[i, 0], local[i, 1]
        return True

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        if self._model_state is None:
            return super().from_hf(hf_state_dict)
        # The load targets alias the model's tensors: they already hold the weights.
        state, self._model_state = self._model_state, None
        return state
