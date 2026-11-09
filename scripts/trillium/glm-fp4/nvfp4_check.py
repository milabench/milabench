"""GLM-FP4, disposable: TransformerEngine's GEMM modules on one B200 in bf16 / mxfp8 / nvfp4, at a model's shapes.

    python nvfp4_check.py glm5     GLM-5, one micro-batch of pp13-ep8: 8192 tokens; 32 local experts of 256, 2048
                                   tokens each at top-8 (glm5.py get_glm5_spec, dev.yaml model.argv)
    python nvfp4_check.py qwen3    Qwen3 32B (hidden 5120, 64 x 128 query heads, 8 x 128 KV heads, FFN 25600, SwiGLU)
                                   at 4096 and 8192 tokens

Each module as Megatron's TE spec builds it (te.Linear, te.LayerNormLinear with RMSNorm where the norm is fused,
te.GroupedLinear for the experts), under te.fp8_autocast with the recipe Megatron builds (MXFP8BlockScaling(),
NVFP4BlockScaling() with TE's defaults: RHT on the weight-gradient inputs, 2D weight blocks, stochastic rounding).
Times per call, in ms: `fwd` (no grad, the first forward under full recompute) and `step` (forward + dgrad + wgrad),
with the weights quantized once and cached across micro-batches as Megatron does (is_first_microbatch), the
activation and gradient quantization (and the Hadamard transforms) in every call; `first` is `step` with the weights
quantized again in every call (a step of one micro-batch). Also the error against bf16 (random data, same weights).
"""

import sys

import torch
import transformer_engine
import transformer_engine.pytorch as te
from transformer_engine.common import recipe as R

EXPERTS, TOK = 32, 2048
MODELS = {
    # name: (module, in, out); GLM-5: hidden 6144, q_lora 2048, 64 x (192 + 64) q / 256 v, kv_lora 512 + rope 64,
    # indexer 32 x 128, expert / shared FFN 2048 (gate + up fused), dense FFN 12288
    "glm5": {
        "q_down": ("linear", 6144, 2048), "q_up": ("ln_linear", 2048, 16384), "kv_down": ("linear", 6144, 576),
        "o_proj": ("linear", 16384, 6144), "idx_wq_b": ("linear", 2048, 4096), "idx_wk": ("linear", 6144, 128),
        "shared_fc1": ("linear", 6144, 4096), "shared_fc2": ("linear", 2048, 6144),
        "expert_fc1": ("grouped", 6144, 4096), "expert_fc2": ("grouped", 2048, 6144),
        "dense_fc1": ("ln_linear", 6144, 24576), "dense_fc2": ("linear", 12288, 6144),
    },
    # Qwen3 32B: QKV 64 x 128 + 2 x 8 x 128, O 64 x 128 -> 5120, gate + up 2 x 25600, down 25600 -> 5120
    "qwen3": {
        "qkv": ("ln_linear", 5120, 10240), "o": ("linear", 8192, 5120),
        "gate_up": ("ln_linear", 5120, 51200), "down": ("linear", 25600, 5120),
    },
}
TOKENS = {"glm5": [8192], "qwen3": [4096, 8192]}
# per layer and micro-batch: GLM-5 runs full recompute (fwd + step), Qwen3 is counted without recompute (step)
LAYERS = {
    "glm5": {"moe": ["q_down", "q_up", "kv_down", "o_proj", "idx_wq_b", "idx_wk", "shared_fc1", "shared_fc2", "expert_fc1", "expert_fc2"],
             "dense": ["q_down", "q_up", "kv_down", "o_proj", "idx_wq_b", "idx_wk", "dense_fc1", "dense_fc2"]},
    "qwen3": {"layer": ["qkv", "o", "gate_up", "down"]},
}
RECIPES = {"bf16": None, "mxfp8": R.MXFP8BlockScaling(), "nvfp4": R.NVFP4BlockScaling()}


def build(kind, n_in, n_out):
    kw = dict(bias=False, params_dtype=torch.bfloat16, device="cuda")
    if kind == "grouped":
        return te.GroupedLinear(EXPERTS, n_in, n_out, **kw)
    if kind == "ln_linear":
        return te.LayerNormLinear(n_in, n_out, normalization="RMSNorm", **kw)
    return te.Linear(n_in, n_out, **kw)


def call(mod, kind, x, recipe, first):
    with te.fp8_autocast(enabled=recipe is not None, fp8_recipe=recipe):
        if kind == "grouped":
            return mod(x, [TOK] * EXPERTS, is_first_microbatch=first)
        return mod(x, is_first_microbatch=first)


def quantizers(mod):
    out = []
    for key, roles in (("scaling_fwd", 3), ("scaling_bwd", 2)):
        for i, q in enumerate(getattr(mod, "quantizers", {}).get(key, [])[:roles]):
            if q is not None:
                attrs = {a: getattr(q, a) for a in ("with_rht", "with_2d_quantization", "stochastic_rounding") if hasattr(q, a)}
                out.append((key, i, type(q).__name__, attrs))
    return out


def timed(fn, n=10):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(3):
        fn()
    start.record()
    for _ in range(n):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n


def bench(name, kind, n_in, n_out, tokens, results):
    rows = TOK * EXPERTS if kind == "grouped" else tokens
    g = torch.Generator(device="cuda").manual_seed(0)
    x0 = torch.randn(rows, n_in, device="cuda", dtype=torch.bfloat16, generator=g)
    dy = torch.randn(rows, n_out, device="cuda", dtype=torch.bfloat16, generator=g) * 1e-2
    ref = None
    for rname, recipe in RECIPES.items():
        torch.manual_seed(0)  # the same weights for every recipe
        mod = build(kind, n_in, n_out)
        x = x0.detach().requires_grad_(True)
        y = call(mod, kind, x, recipe, True)
        y.backward(dy)
        out = (y.detach().float(), next(p for p in mod.parameters() if p.dim() == 2).grad.detach().float())
        nan = bool(torch.isnan(out[0]).any() or torch.isnan(out[1]).any())
        if ref is None:
            ref, err = out, (0.0, 0.0)
        else:
            err = tuple(((a - b).norm() / b.norm()).item() for a, b in zip(out, ref))
        del y, x, out

        def fwd():
            with torch.no_grad():
                call(mod, kind, x0, recipe, False)

        def step(first=False):
            call(mod, kind, x0.detach().requires_grad_(True), recipe, first).backward(dy)

        t = dict(fwd=timed(fwd), step=timed(step), first=timed(lambda: step(True)))
        results[(name, tokens, rname)] = t
        print(f"{name:11s} {tokens:5d} {rname:6s} fwd {t['fwd']:7.3f}  step {t['step']:7.3f}  first {t['first']:7.3f} ms"
              f"  rel.err y {err[0]:.4f} wgrad {err[1]:.4f}  nan {nan}", flush=True)
        if rname == "nvfp4" and name in ("q_down", "expert_fc1", "qkv"):
            for q in quantizers(mod):
                print("    ", q, flush=True)
        del mod


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "glm5"
    print(torch.__version__, transformer_engine.__version__, torch.cuda.get_device_name(), torch.cuda.get_device_capability(), flush=True)
    print("NVFP4BlockScaling():", RECIPES["nvfp4"], flush=True)
    results = {}
    for tokens in TOKENS[model]:
        for name, (kind, n_in, n_out) in MODELS[model].items():
            try:
                bench(name, kind, n_in, n_out, tokens, results)
            except Exception as exc:  # a shape TE refuses is a finding: report it, go on with the others
                print(f"{name:11s} {tokens:5d} FAILED: {type(exc).__name__}: {str(exc)[:300]}", flush=True)
        for name in MODELS[model]:
            if all((name, tokens, r) in results for r in RECIPES):
                b, m, f = (results[(name, tokens, r)] for r in RECIPES)
                print(f"speedup {name:11s} {tokens:5d}  step mxfp8 {b['step'] / m['step']:5.2f}x nvfp4 {b['step'] / f['step']:5.2f}x"
                      f" (nvfp4 / mxfp8 {m['step'] / f['step']:5.2f}x)  first mxfp8 {b['first'] / m['first']:5.2f}x"
                      f" nvfp4 {b['first'] / f['first']:5.2f}x", flush=True)
        for layer, names in LAYERS[model].items():
            if any((n, tokens, r) not in results for n in names for r in RECIPES):
                continue
            keys = ("fwd", "step") if model == "glm5" else ("step",)
            t = {r: sum(results[(n, tokens, r)][k] for n in names for k in keys) for r in RECIPES}
            print(f"layer {model} {layer:5s} {tokens:5d}  per micro-batch ({' + '.join(keys)}) bf16 {t['bf16']:7.2f} mxfp8 "
                  f"{t['mxfp8']:7.2f} nvfp4 {t['nvfp4']:7.2f} ms: mxfp8 {t['bf16'] / t['mxfp8']:5.2f}x nvfp4 "
                  f"{t['bf16'] / t['nvfp4']:5.2f}x", flush=True)


if __name__ == "__main__":
    main()
