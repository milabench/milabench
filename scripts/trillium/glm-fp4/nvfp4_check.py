"""GLM-FP4 step 1, disposable: TransformerEngine's NVFP4 recipe on one B200, at GLM-5's GEMM shapes.

One micro-batch of pp13-ep8 (8192 tokens; 32 local experts of 256, about 2048 tokens each at top-8). For each TE module
GLM-5 calls under Megatron's per-layer autocast (glm5.py get_glm5_spec, dev.yaml model.argv), in bf16 / mxfp8 / nvfp4:
- the quantizers TE builds (RHT, 2D weight blocks, stochastic rounding),
- fwd and fwd+bwd time per micro-batch, weights cached across micro-batches as Megatron does (is_first_microbatch),
- the memory TE keeps after the first micro-batch (the quantized weight copies),
- the output and weight-gradient error against bf16 (same weights, same input).
Run: python nvfp4_check.py (one GPU).
"""

import torch
import transformer_engine
import transformer_engine.pytorch as te
from transformer_engine.common import recipe as R

T, EXPERTS, TOK = 8192, 32, 2048
# name: (module, in, out); GLM-5: hidden 6144, q_lora 2048, heads 64 x (192 + 64) q / 256 v, kv_lora 512 + rope 64,
# indexer 32 x 128 heads, expert / shared ffn 2048 (gate + up fused), dense ffn 12288
MODULES = {
    "q_down": ("linear", 6144, 2048),
    "q_up": ("ln_linear", 2048, 16384),
    "kv_down": ("linear", 6144, 576),
    "o_proj": ("linear", 16384, 6144),
    "idx_wq_b": ("linear", 2048, 4096),
    "idx_wk": ("linear", 6144, 128),
    "shared_fc1": ("linear", 6144, 4096),
    "shared_fc2": ("linear", 2048, 6144),
    "expert_fc1": ("grouped", 6144, 4096),
    "expert_fc2": ("grouped", 2048, 6144),
    "dense_fc1": ("ln_linear", 6144, 24576),
    "dense_fc2": ("linear", 12288, 6144),
}
MOE_LAYER = ["q_down", "q_up", "kv_down", "o_proj", "idx_wq_b", "idx_wk", "shared_fc1", "shared_fc2", "expert_fc1", "expert_fc2"]
DENSE_LAYER = ["q_down", "q_up", "kv_down", "o_proj", "idx_wq_b", "idx_wk", "dense_fc1", "dense_fc2"]
RECIPES = {"bf16": None, "mxfp8": R.MXFP8BlockScaling(), "nvfp4": R.NVFP4BlockScaling()}
GIB = 2**30


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


def quantizer_summary(mod):
    out = []
    for key in ("scaling_fwd", "scaling_bwd"):
        seen = set()
        for i, q in enumerate(getattr(mod, "quantizers", {}).get(key, [])):
            if q is None:
                continue
            attrs = {a: getattr(q, a) for a in ("with_rht", "with_2d_quantization", "stochastic_rounding") if hasattr(q, a)}
            item = (key, i % (3 if key == "scaling_fwd" else 2), type(q).__name__, tuple(sorted(attrs.items())))
            if item not in seen:
                seen.add(item)
                out.append(item)
    return out


def bench(name, ref_state, results):
    kind, n_in, n_out = MODULES[name]
    rows = T if kind != "grouped" else TOK * EXPERTS
    g = torch.Generator(device="cuda").manual_seed(0)
    x0 = torch.randn(rows, n_in, device="cuda", dtype=torch.bfloat16, generator=g)
    dy = torch.randn(rows, n_out, device="cuda", dtype=torch.bfloat16, generator=g) * 1e-2
    for rname, recipe in RECIPES.items():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        mod = build(kind, n_in, n_out)
        if ref_state.get(name) is None:
            ref_state[name] = {k: v.detach().clone() for k, v in mod.state_dict().items() if torch.is_tensor(v) and v.is_floating_point()}
        else:
            with torch.no_grad():
                for k, v in mod.state_dict().items():
                    if k in ref_state[name]:
                        v.copy_(ref_state[name][k])
        torch.cuda.synchronize()
        m0 = torch.cuda.memory_allocated()
        x = x0.detach().requires_grad_(True)
        y = call(mod, kind, x, recipe, True)
        y.backward(dy)
        y_out = y.detach().float()
        wgrad = next(p for p in mod.parameters() if p.dim() == 2).grad.detach().float()
        nan = bool(torch.isnan(y_out).any() or torch.isnan(wgrad).any())
        if rname == "bf16":
            ref_bytes = y_out.nbytes + wgrad.nbytes  # kept as the reference, not counted below
            ey = ew = 0.0
        else:
            ref = results[(name, "bf16")]
            ey = ((y_out - ref["y"]).norm() / ref["y"].norm()).item()
            ew = ((wgrad - ref["wgrad"]).norm() / ref["wgrad"].norm()).item()
            ref_bytes = 0
            del y_out, wgrad
        del y, x
        for p in mod.parameters():
            p.grad = None
        torch.cuda.synchronize()
        kept = torch.cuda.memory_allocated() - m0 - ref_bytes
        params = sum(p.numel() for p in mod.parameters() if p.dim() == 2)
        quant = quantizer_summary(mod)
        # steady state: weights already quantized (micro-batches 2..32 of a step)
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        for _ in range(3):
            x = x0.detach().requires_grad_(True)
            call(mod, kind, x, recipe, False).backward(dy)
        n = 10
        ev[0].record()
        for _ in range(n):
            x = x0.detach().requires_grad_(True)
            with torch.no_grad():
                call(mod, kind, x, recipe, False)
        ev[1].record()
        for _ in range(n):
            x = x0.detach().requires_grad_(True)
            call(mod, kind, x, recipe, False).backward(dy)
        ev[2].record()
        torch.cuda.synchronize()
        fwd = ev[0].elapsed_time(ev[1]) / n
        fwdbwd = ev[1].elapsed_time(ev[2]) / n
        results[(name, rname)] = dict(fwd=fwd, fwdbwd=fwdbwd, kept=kept, params=params, quant=quant)
        if rname == "bf16":
            results[(name, rname)].update(y=y_out, wgrad=wgrad)
        print(f"{name:11s} {rname:6s} fwd {fwd:7.3f} ms  fwd+bwd {fwdbwd:7.3f} ms  kept {kept / GIB:6.3f} GiB "
              f"({kept / params:5.3f} B/param of {params / 1e6:7.1f} M)  rel.err y {ey:.4f} wgrad {ew:.4f}  nan {nan}", flush=True)
        del mod
    for rname in RECIPES:
        results.get((name, rname), {}).pop("y", None)
        results.get((name, rname), {}).pop("wgrad", None)


def main():
    print(torch.__version__, transformer_engine.__version__, torch.cuda.get_device_name(), torch.cuda.get_device_capability(), flush=True)
    print("NVFP4BlockScaling():", RECIPES["nvfp4"], flush=True)
    results, ref_state = {}, {}
    for name in MODULES:
        try:
            bench(name, ref_state, results)
        except Exception as exc:  # a shape TE refuses is a finding: report it, go on with the others
            print(f"{name:11s} FAILED: {type(exc).__name__}: {str(exc)[:400]}", flush=True)
        ref_state[name] = None
    for name in ("q_down", "expert_fc1"):
        if (name, "nvfp4") not in results:
            continue
        print(f"quantizers {name} (nvfp4): (list, index mod 3 fwd [input, weight, output] / mod 2 bwd [grad_out, grad_in], type, settings)")
        for q in results[(name, "nvfp4")]["quant"]:
            print("   ", q)
    # per micro-batch and layer, full recompute (dev.yaml): forward twice (forward, recompute) + backward
    for layer, names in (("moe", MOE_LAYER), ("dense", DENSE_LAYER)):
        for rname in RECIPES:
            if any((n, rname) not in results for n in names):
                continue
            t = sum(results[(n, rname)]["fwd"] + results[(n, rname)]["fwdbwd"] for n in names)
            kept = sum(results[(n, rname)]["kept"] for n in names)
            params = sum(results[(n, rname)]["params"] for n in names)
            print(f"layer {layer:5s} {rname:6s} GEMM modules per micro-batch {t:7.2f} ms; middle stage step (32 x 6 layers) "
                  f"{t * 192 / 1000:6.2f} s; kept {kept / GIB:6.2f} GiB per layer ({kept / params:5.3f} B/param)", flush=True)


if __name__ == "__main__":
    main()
