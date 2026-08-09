"""HF checkpoint adapter for transformers-backend MoE models on milabench."""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from typing import Any

import torch

from torchtitan.experiments.transformers_modeling_backend.state_dict_adapter import (
    HFTransformerStateDictAdapter,
    hf_to_titan_moe_state_dict,
    titan_to_hf_moe_state_dict,
    _expert_names,
)

_TITAN_ONLY_KEY_PARTS = (
    "expert_bias_E",
    "tokens_per_expert_E",
)

_NUMBERED_EXPERT_RE = re.compile(
    r"^layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"
)

_GROUPED_EXPERT_RE = re.compile(
    r"^layers\.(\d+)\.mlp\.routed_experts\.inner_experts\.([^.]+)$"
)

_STACKED_HF_GATE_UP_RE = re.compile(
    r"^layers\.(\d+)\.mlp\.experts\.gate_up_proj(?:\.weight)?$"
)
_STACKED_HF_PROJ_RE = re.compile(
    r"^layers\.(\d+)\.mlp\.experts\.(gate_proj|up_proj|down_proj)(?:\.weight)?$"
)


def _with_model_prefix(hf_state_dict: dict[str, Any]) -> dict[str, Any]:
    """Safetensors on disk use a leading ``model.``; DCP load plans must match."""
    return {
        k if k.startswith("model.") else f"model.{k}": v
        for k, v in hf_state_dict.items()
    }


def _as_dense_tensor(value: Any) -> torch.Tensor | None:
    if not isinstance(value, torch.Tensor):
        return None
    tensor = value
    if hasattr(tensor, "to_local"):
        try:
            tensor = tensor.to_local()
        except Exception:
            pass
    if hasattr(tensor, "full_tensor"):
        try:
            tensor = tensor.full_tensor()
        except Exception:
            pass
    return tensor


class MilabenchMoeStateDictAdapter(HFTransformerStateDictAdapter):
    """Map HF MoE safetensors to Titan MoE modules for initial_load_in_hf."""

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        filtered = {
            k: v
            for k, v in state_dict.items()
            if not any(part in k for part in _TITAN_ONLY_KEY_PARTS)
        }
        hf = super().to_hf(filtered)
        hf = titan_to_hf_moe_state_dict(hf)
        hf = self._split_grouped_experts(hf)
        hf = self._rekey_for_hf_safetensors(hf)
        weight_map = self._load_hf_weight_map()
        if weight_map:
            return self._align_to_weight_map(hf, weight_map)
        return _with_model_prefix(hf)

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        normalized = {
            k.removeprefix("model."): v for k, v in hf_state_dict.items()
        }
        stacked, remaining = self._stack_numbered_experts(normalized)
        converted = hf_to_titan_moe_state_dict(remaining)
        converted.update(stacked)
        renamed = {
            re.sub(r"\.expert_bias$", ".expert_bias_E", k): v
            for k, v in converted.items()
        }
        return super().from_hf(renamed)

    def _load_hf_weight_map(self) -> dict[str, str] | None:
        if not self.hf_assets_path:
            return None
        mapping_path = os.path.join(self.hf_assets_path, "model.safetensors.index.json")
        if not os.path.isfile(mapping_path):
            return None
        with open(mapping_path, encoding="utf-8") as handle:
            return json.load(handle).get("weight_map")

    def _rekey_for_hf_safetensors(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        """Expand stacked MoE tensors into numbered HF keys expected on disk."""
        weight_map = self._load_hf_weight_map()
        out: dict[str, Any] = {}
        stacked: dict[tuple[str, str], torch.Tensor] = {}

        for key, value in hf_state_dict.items():
            norm = key.removeprefix("model.")
            gate_up = _STACKED_HF_GATE_UP_RE.match(norm)
            if gate_up:
                dense = _as_dense_tensor(value)
                if dense is not None and dense.ndim == 3:
                    stacked[(gate_up.group(1), "gate_up_proj")] = dense
                continue

            proj = _STACKED_HF_PROJ_RE.match(norm)
            if proj:
                dense = _as_dense_tensor(value)
                if dense is not None and dense.ndim == 3:
                    stacked[(proj.group(1), proj.group(2))] = dense
                continue

            out[norm] = value

        if not weight_map:
            return self._split_stacked_hf_experts(out)

        for ck in weight_map:
            hf_key = ck.removeprefix("model.")
            if hf_key in out:
                continue
            match = _NUMBERED_EXPERT_RE.match(hf_key)
            if not match:
                continue
            layer, expert, proj = match.groups()
            expert_idx = int(expert)

            if proj in ("gate_proj", "up_proj"):
                src = stacked.get((layer, "gate_up_proj"))
                if src is None or expert_idx >= src.shape[0]:
                    continue
                half = src.shape[1] // 2
                if proj == "gate_proj":
                    out[hf_key] = src[expert_idx, :half, :].clone()
                else:
                    out[hf_key] = src[expert_idx, half:, :].clone()
                continue

            src = stacked.get((layer, "down_proj"))
            if src is None or expert_idx >= src.shape[0]:
                continue
            out[hf_key] = src[expert_idx].clone()

        return out

    def _align_to_weight_map(
        self, hf_state_dict: dict[str, Any], weight_map: dict[str, str]
    ) -> dict[str, Any]:
        """Use on-disk safetensor key spellings in the DCP load plan."""
        by_bare = {k.removeprefix("model."): v for k, v in hf_state_dict.items()}
        aligned: dict[str, Any] = {}
        for ck in weight_map:
            bare = ck.removeprefix("model.")
            if bare in by_bare:
                aligned[ck] = by_bare[bare]
        return aligned

    def _stack_numbered_experts(
        self, hf_state_dict: dict[str, Any]
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        gate_name, down_name, up_name = _expert_names()
        proj_map = {
            "gate_proj": gate_name,
            "up_proj": up_name,
            "down_proj": down_name,
        }
        buckets: dict[tuple[str, str], dict[int, torch.Tensor]] = defaultdict(dict)
        remaining: dict[str, Any] = {}

        for key, value in hf_state_dict.items():
            match = _NUMBERED_EXPERT_RE.match(key)
            if not match:
                remaining[key] = value
                continue
            layer, expert, proj = match.groups()
            titan_key = (
                f"layers.{layer}.mlp.routed_experts.inner_experts.{proj_map[proj]}"
            )
            buckets[(titan_key, proj)][int(expert)] = value

        stacked: dict[str, torch.Tensor] = {}
        for (titan_key, _), experts in buckets.items():
            ordered = [experts[i] for i in sorted(experts)]
            stacked[titan_key] = torch.stack(ordered, dim=0)

        return stacked, remaining

    def _split_stacked_hf_experts(
        self, hf_state_dict: dict[str, Any]
    ) -> dict[str, Any]:
        """Split stacked HF MoE tensors into per-expert keys (fallback path)."""
        split: dict[str, torch.Tensor] = {}
        passthrough: dict[str, Any] = {}

        for key, value in hf_state_dict.items():
            dense = _as_dense_tensor(value)
            if dense is None:
                passthrough[key] = value
                continue

            norm = key.removeprefix("model.")
            gate_up = _STACKED_HF_GATE_UP_RE.match(norm)
            if gate_up and dense.ndim == 3:
                layer = gate_up.group(1)
                half = dense.shape[1] // 2
                for expert_idx in range(dense.shape[0]):
                    prefix = f"layers.{layer}.mlp.experts.{expert_idx}"
                    split[f"{prefix}.gate_proj.weight"] = dense[expert_idx, :half, :]
                    split[f"{prefix}.up_proj.weight"] = dense[expert_idx, half:, :]
                continue

            proj = _STACKED_HF_PROJ_RE.match(norm)
            if proj and dense.ndim == 3:
                layer, proj_name = proj.groups()
                for expert_idx in range(dense.shape[0]):
                    split_key = (
                        f"layers.{layer}.mlp.experts.{expert_idx}.{proj_name}.weight"
                    )
                    split[split_key] = dense[expert_idx]
                continue

            passthrough[key] = value

        passthrough.update(split)
        return passthrough

    def _split_grouped_experts(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        split: dict[str, torch.Tensor] = {}
        passthrough: dict[str, Any] = {}

        for key, value in hf_state_dict.items():
            match = _GROUPED_EXPERT_RE.match(key.removeprefix("model."))
            if not match:
                passthrough[key] = value
                continue
            dense = _as_dense_tensor(value)
            if dense is None or dense.ndim != 3:
                passthrough[key] = value
                continue

            layer, param_name = match.groups()
            gate_name, down_name, up_name = _expert_names()
            if param_name == gate_name:
                proj = "gate_proj"
            elif param_name == up_name:
                proj = "up_proj"
            elif param_name == down_name:
                proj = "down_proj"
            else:
                passthrough[key] = value
                continue

            for expert_idx in range(dense.shape[0]):
                split_key = (
                    f"layers.{layer}.mlp.experts.{expert_idx}.{proj}.weight"
                )
                split[split_key] = dense[expert_idx]

        passthrough.update(split)
        return passthrough
