#!/usr/bin/env python
"""How many long-context sessions fit next to each model: KV cache per session and sessions per context length.

    <milabench venv>/bin/python benchmarks/longctx/kv_capacity.py config/homelab/text.yaml [--fraction 0.65] [--json out.json]

No GPU, no server: the model's config.json and weights size (HF cache, else config.json from the Hub) and the
machine's memory. The budget is what a vLLM server gets on unified memory (milabench.commands.serving):
fraction x MemTotal (MILABENCH_GPU_MEMORY_FRACTION_MAX, 0.65 on the Spark run) - weights - runtime overhead.
A session at length L holds the full-attention layers' KV for L tokens, the sliding-window layers' KV for
min(L, window) tokens and, for hybrid models, a fixed recurrent state (Mamba / Gated DeltaNet / KDA), estimated
here from the config. The benchmark's own vLLM KV pool is sized for its passes (MILABENCH_KV_CACHE_FIT), so the
server log's "maximum concurrency" is not this number: this is the machine-wide capacity.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

LENGTHS = [8192, 32768, 65536, 131072, 262144]


def _text_config(hf: dict) -> dict:
    for key in ("text_config", "llm_config", "language_config"):
        if isinstance(hf.get(key), dict) and hf[key].get("num_hidden_layers"):
            return hf[key]
    return hf


def _config(model: str) -> dict | None:
    from milabench.commands.serving import _hf_config

    hf = _hf_config(model)
    if hf is None:  # not downloaded yet: config.json from the Hub (HF_TOKEN for gated repos)
        import urllib.request

        request = urllib.request.Request(f"https://huggingface.co/{model}/resolve/main/config.json")
        if os.environ.get("HF_TOKEN"):
            request.add_header("Authorization", f"Bearer {os.environ['HF_TOKEN']}")
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                hf = json.load(response)
        except Exception as exc:  # noqa: BLE001
            print(f"{model}: no config.json ({exc})", file=sys.stderr)
    return hf


def layout(hf: dict, kv_dtype: str | None) -> dict | None:
    """milabench's kv_layout, corrected for the hybrids it counts as all-attention (Nemotron 3.5 puzzle configs
    with layers_block_type, Kimi-Linear's linear_attn_config), plus the recurrent state per sequence."""
    from milabench.commands.serving import kv_layout

    cfg = _text_config(hf)
    for key in ("llm_config", "text_config"):  # Nemotron 3.5 puzzle configs: layers_block_type, no layer count
        if isinstance(hf.get(key), dict) and isinstance(hf[key].get("layers_block_type"), list):
            cfg = hf[key]
    blocks = cfg.get("layers_block_type")
    if isinstance(blocks, list) and "attention" in blocks:
        elem = 1 if str(kv_dtype or "").startswith("fp8") else 2
        heads = cfg.get("num_key_value_heads") or cfg.get("num_attention_heads")
        head_dim = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
        out = {"full": 2 * blocks.count("attention") * heads * head_dim * elem, "sliding": 0, "window": None}
    else:
        out = kv_layout(hf, kv_dtype)
    if out is None:
        return None
    layers = cfg.get("num_hidden_layers") or 1
    lin = cfg.get("linear_attn_config") or {}
    if lin.get("full_attn_layers"):
        out["full"] = out["full"] * len(lin["full_attn_layers"]) // layers
    out["state"] = recurrent_state(cfg)
    return out


def recurrent_state(cfg: dict) -> int:
    """Bytes of fixed per-sequence state of the linear-attention / Mamba layers (rough: SSM / delta-rule
    matrices only, conv buffers ignored)."""
    kinds = [str(t) for t in cfg.get("layer_types") or []]
    if kinds and "linear_attention" in kinds:  # Qwen3.5/3.6/3.8, Qwen3-Next: Gated DeltaNet, fp32 state
        n = kinds.count("linear_attention")
        heads = cfg.get("linear_num_value_heads") or 32
        return n * heads * (cfg.get("linear_key_head_dim") or 128) * (cfg.get("linear_value_head_dim") or 128) * 4
    pattern = cfg.get("hybrid_override_pattern")
    blocks = cfg.get("layers_block_type")
    mamba = pattern.count("M") if isinstance(pattern, str) else (blocks.count("mamba") if isinstance(blocks, list) else 0)
    if mamba:  # Nemotron-H Mamba-2: heads x head_dim x state
        return mamba * (cfg.get("mamba_num_heads") or 128) * (cfg.get("mamba_head_dim") or 64) * (cfg.get("ssm_state_size") or 128) * 2
    lin = cfg.get("linear_attn_config") or {}
    if lin.get("kda_layers"):  # Kimi Delta Attention
        return len(lin["kda_layers"]) * lin.get("num_heads", 32) * lin.get("head_dim", 128) ** 2 * 4
    return 0


def hub_weights_bytes(model: str) -> int:
    """Top-level safetensors of the repo on the Hub (a model not, or not fully, downloaded yet); 0 offline."""
    import urllib.request

    request = urllib.request.Request(f"https://huggingface.co/api/models/{model}?blobs=true")
    if os.environ.get("HF_TOKEN"):
        request.add_header("Authorization", f"Bearer {os.environ['HF_TOKEN']}")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            files = json.load(response).get("siblings") or []
    except Exception:  # noqa: BLE001
        return 0
    return sum(f.get("size") or 0 for f in files if "/" not in f["rfilename"] and f["rfilename"].endswith(".safetensors"))


def session_bytes(lay: dict, length: int) -> int:
    kept = min(length, lay["window"]) if lay.get("window") else length
    return int(length * lay["full"] + kept * lay["sliding"] + lay["state"])


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--fraction", type=float,
                        default=float(os.environ.get("MILABENCH_GPU_MEMORY_FRACTION_MAX") or 0.65))
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)

    from milabench.commands.serving import _flag, _server_model, memory_total, runtime_overhead, weights_bytes
    from milabench.config import build_config
    from milabench.utils import assemble_options

    total = memory_total()
    rows = []
    for name, cfg in build_config(args.config).items():
        if name.startswith("_") or not cfg.get("enabled") or "longctx" not in (cfg.get("tags") or []):
            continue
        argv = [str(a) for a in assemble_options((cfg.get("server") or {}).get("argv") or {})]
        model = _server_model(argv)
        hf = _config(model) if model else None
        lay = layout(hf, _flag(argv, "--kv-cache-dtype")) if hf else None
        if not lay:
            continue
        weights = max(weights_bytes(model), hub_weights_bytes(model))
        budget = args.fraction * total - weights - runtime_overhead(weights)
        row = {"bench": name, "model": model, "weights_gb": round(weights / 1e9, 1),
               "kv_per_token_kb": round((lay["full"] + lay["sliding"]) / 1024, 1),
               "state_mb": round(lay["state"] / 2**20, 1), "budget_gb": round(budget / 1e9, 1), "sessions": {}}
        for length in LENGTHS:
            per = session_bytes(lay, length)
            row["sessions"][length] = {"gb": round(per / 1e9, 2), "fit": max(0, int(budget // per))}
        rows.append(row)

    head = f"{'bench':58} {'weights':>7} {'KV/tok':>7} {'state':>6} {'budget':>7}  " + "  ".join(f"{n // 1024:>4}k GB/fit" for n in LENGTHS)
    print(head)
    for r in rows:
        cells = "  ".join(f"{r['sessions'][n]['gb']:>6.2f}/{r['sessions'][n]['fit']:<4}" for n in LENGTHS)
        weights = f"{r['weights_gb']}G" if r["weights_gb"] else "n/c"
        print(f"{r['bench']:58} {weights:>7} {r['kv_per_token_kb']:>5}KB {r['state_mb']:>4.0f}MB {r['budget_gb']:>6}G  {cells}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
