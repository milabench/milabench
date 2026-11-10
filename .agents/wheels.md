# Dependencies, pins and wheels

Read this when adding a dependency, pinning a new torch/accelerator combo, or building GPU extension wheels.

## Pins

- Each benchmark lists its dependencies in `benchmarks/<bench>/requirements.toml` (`[common]` plus one
  section per accelerator). Pins live in `.pin/constraints.<arch><ver>.torch<ver>.txt`.
- Pin one combo per call: `milabench tools pin --set rocm=7.2 torch=2.14.1`
  - `--from-scratch`: stop uv reusing old versions from the lockfile.
  - `--pip-compile=--upgrade-package=xformers`: refresh one package only.
- For vLLM, add a `[vllm.<arch>]` entry `"<torch>,<ver>"` in `platforms.toml` before pinning.
- Locally built wheels are found through `find-links` (`$MILABENCH_WHEELS_DIR/torch{torch_short}-rocm{rocm}`).
- **uv picks the extra index first.** For a hand-made test venv use `--index-url https://pypi.org/simple
  --extra-index-url https://download.pytorch.org/whl/<arch>`, otherwise you silently get the wrong torch.
- Watch for packages that resolve to an unrelated PyPI project (`atom` must be `atom>=0.1.8,<0.2`) and for
  CUDA-only wheels in `[common]` that pin torch exactly (they make every other torch unresolvable).

## Building wheels (milabench/wheels repo)

Upload a wheel only after a smoke test passes. Wheels go to release tag `torch<X.Y>-<arch><ver>`.

- **Never build on a host where milabench runs**: the GPU warden kills anything holding a GPU.
  Build in a container, or in a systemd unit with `PrivateDevices=yes`:
  ```bash
  sudo systemd-run --unit wheels-X --property=PrivateDevices=yes --uid=$(id -u) --gid=$(id -g) \
    --setenv=HOME=$HOME --setenv=CHAIN_STEPS="aiter mori atom" \
    --setenv=WHEELS_DIR=$PWD/wheels/torch2.15-rocm10.0-vllm \
    --working-directory=$PWD/build-logs bash -c "bash $PWD/build-logs/run-vllm-chain.sh > $PWD/build-logs/X.log 2>&1"
  ```
- ROCm 7.x (apt toolkit), same as CI:
  `PYTORCH_VERSION=2.14.1 ROCM_VERSION=7.2 VLLM_VERSION=0.32.0 bash scripts/container-build-rocm.sh`
  (`STEPS="aiter vllm"` to rebuild some packages only). Runs `ubuntu:22.04`, so wheels are glibc 2.35 compatible.
- ROCm 10+ (rocm-sdk wheels), in the venv:
  `make GPU_BACKEND=rocm ROCM_VERSION=10.0 PYTORCH_VERSION=2.15.0 WHEELS_DIR=$PWD/wheels/torch2.15-rocm10.0 pyg torchao xformers`.
  `common.sh` builds against `rocm-sdk path --root`, not `/opt/rocm`.
- Upload: `gh release upload <tag> -R milabench/wheels wheels/<dir>/*.whl` (`gh release create` first if new).

### Versions (`wheels/.env`)
- vLLM 0.32.0: pins `torch==2.13.0`; `build-vllm.sh` strips it with `tools/use_existing_torch.py --prefix`
  (CUDA: only when the pin differs from the build torch). Building a vLLM against a newer torch than it
  pins (GCC 13, libdw headers, nvcc parallelism, wheel checks): see "Building vLLM against a newer torch"
  in `wheels/AGENTS.md` before dispatching CI.
- aiter `v0.1.26` for torch 2.15 (needs `flydsl==0.4.0`); ATOM 0.1.8's Kimi-K3/GLM-5.2 need aiter ≥ 0.1.25.
- mori `v1.2.3.post1`; ATOM 0.1.8 (built `0.1.8.post1`); xformers 0.0.35; flash-attn 2.8.3;
  torchao 0.18.0 (built `0.18.0.post1`); torch-cluster 1.6.3, torch-scatter 2.1.2, torch-sparse 0.6.18.

### Patches in the build scripts (know why)
- **C++20**: torch ≥ 2.15 headers use `std::strong_ordering`; flash-attn, xformers and mori hardcode `-std=c++17`.
  For mori, setup.py otherwise silently drops the SymmetricMemory extension: check `mori_torch_symm*.so` is in the wheel.
- **xformers on ROCm 10**: `-Wno-error=deprecated-attributes`; stamped `+rocmX.Y` so installs prefer it over
  PyPI's CUDA-only build (no ROCm kernels, breaks dinov2).
- **aiter**: `AITER_USE_SYSTEM_TRITON=1`, otherwise it replaces triton-rocm with PyPI's CUDA triton.
- **vLLM**: `env_override.py` symbol promotion made opt-in (`VLLM_ROCM_PROMOTE_TORCH_SYMBOLS=1`), otherwise
  `import vllm` segfaults in libtriton.
- **mori/amdsmi on rocm-sdk**: `common.sh` fixes the SDK's leaked build paths; mori needs `setuptools<80` and
  `BUILD_UMBP=OFF` (UMBP links system gRPC/protobuf).
- **torchao**: backports pytorch/ao#4880 (autotune key Triton ≥ 3.8 rejects, hit by every torchtitan model).
- **ATOM**: applies `wheels/patches/atom/<PR>-*.patch` (e.g. #2626, Kimi-K3 one-token prefill crash).
- **Bump the version whenever you patch a release** (`.postN` before the local tag), otherwise install keeps
  the already-installed wheel.
- **Check portability**: unzip and `ldd` every `.so`. Only torch, ROCm and rocm-sdk libraries may be
  unresolved; system libraries like protobuf or gRPC must not appear.
