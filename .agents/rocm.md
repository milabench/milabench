# AMD / ROCm specifics

Read this when running on AMD GPUs (`MILABENCH_GPU_ARCH=rocm`).

## Pitfalls

- **`env.sh` may export `ROCM_PATH=` (empty).** Use `os.environ.get("ROCM_PATH") or "/opt/rocm"`,
  never `.get("ROCM_PATH", "/opt/rocm")`.
- **amdsmi must match the host ROCm, not the torch build**: it is a Python wrapper over the host's
  `libamd_smi.so`. 27.x needs `libamd_smi.so.27` (host ROCm ≥ 7.14), 26.x needs `.so.26`.
  A missing or mismatched amdsmi makes vLLM fail with `Failed to infer device type`.
- voir's GPU monitor (`voir/instruments/gpu/rocm.py`) also uses amdsmi; devices are numbered by
  `hip_id`, which matches `ROCR_VISIBLE_DEVICES`.
- `ROCR_VISIBLE_DEVICES=` does not hide a process from the GPU warden: it still opens `/dev/kfd`.

## Serving: vLLM vs ATOM

The `serving-*` benchmarks pick their server per arch (`variants.rocm.server.backend`):
ATOM on ROCm for Kimi-K3 and GLM-5.2, vLLM for the rest; the `vllm-*` benchmarks always use vLLM.

- **ATOM installs vLLM plugins** that swap vLLM's models for ATOM's. vLLM-backend benchmarks set
  `ATOM_DISABLE_VLLM_PLUGIN=1` (`benchmarks/vllm/benchfile.py`); over Ray it must reach the workers too.
- The ATOM backend sets `ROCM_PATH=/opt/rocm`: flydsl's MLIR serializer needs `amdgcn/bitcode` and
  `llvm/bin/ld.lld`, which the rocm-sdk wheels lay out differently.
- ATOM pins `transformers` exactly (`==5.16.1` for ATOM 0.1.8), which the whole shared venv then uses.
- Don't force `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`: it switches every server to flash-attn's Triton
  backend, whose autotune keys Triton ≥ 3.8 rejects. vLLM only wants it on RDNA.
- ATOM's API server doesn't exit when its engine dies, see [debugging.md](debugging.md).
- Weights are large (Kimi-K3 1.5 TB, GLM-5.2 FP8 704 GB, MXFP4 408 GB): pre-fetch them, see [running.md](running.md).

Building ROCm wheels (aiter, mori, ATOM, vLLM, ...) is in [wheels.md](wheels.md).
