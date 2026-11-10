#!/bin/bash
# Build GPU extension wheels (xformers, PyG, torchao) for one torch + CUDA combo.
#
# The canonical build lives in milabench/wheels (CI: build-cuda.yml); this is a
# quick local fallback. Every version can be overridden from the environment:
#
#   PYTORCH_VERSION=2.15.0 CUDA_VERSION=132 bash wheel.sh
#   CUDA_VERSION=13.0 UPLOAD=0 bash wheel.sh            # build only, no upload
#
# CUDA_VERSION accepts 132, 13.2, 13.2.0 or cu132.
set -euo pipefail

PYTORCH_VERSION="${PYTORCH_VERSION:-2.15.0}"
CUDA_VERSION="${CUDA_VERSION:-132}"

CLUSTER_VERSION="${CLUSTER_VERSION:-1.6.3}"
SPARSE_VERSION="${SPARSE_VERSION:-0.6.18}"
SCATTER_VERSION="${SCATTER_VERSION:-2.1.2}"
TORCH_AO_VERSION="${TORCH_AO_VERSION:-0.18.0}"
XFORMER_VERSION="${XFORMER_VERSION:-0.0.35}"

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.5;8.0;8.6;8.9;9.0;10.0;12.0;12.1}"

WHEELS_DIR="${WHEELS_DIR:-$(pwd)/wheels}"
UPLOAD="${UPLOAD:-1}"
WHEELS_REPO="${WHEELS_REPO:-milabench/wheels}"

# Derived names: cu132, pt215, release tag torch2.15-cu132 (wheels repo convention)
_cuda="${CUDA_VERSION#cu}"
if [[ "$_cuda" == *.* ]]; then
    IFS=. read -r _cmaj _cmin _ <<< "$_cuda"
    _cuda="${_cmaj}${_cmin}"
fi
CUDA="cu${_cuda}"
IFS=. read -r TORCH_MAJOR TORCH_MINOR _ <<< "$PYTORCH_VERSION"
PT_VER="pt${TORCH_MAJOR}${TORCH_MINOR}"
RELEASE_TAG="${RELEASE_TAG:-torch${TORCH_MAJOR}.${TORCH_MINOR}-${CUDA}}"

echo "==> torch ${PYTORCH_VERSION} + ${CUDA} -> ${WHEELS_DIR} (release ${RELEASE_TAG})"
mkdir -p "$WHEELS_DIR"

pip install --extra-index-url "https://download.pytorch.org/whl/${CUDA}" "torch==${PYTORCH_VERSION}+${CUDA}"
pip install wheel setuptools cmake ninja

(
    git clone --recurse-submodules --branch "v${XFORMER_VERSION}" https://github.com/facebookresearch/xformers.git
    cd xformers
    # No +cuXXX local version for this one
    BUILD_VERSION="$XFORMER_VERSION" FORCE_CUDA=1 pip wheel . --no-cache-dir --no-deps --no-build-isolation -w "$WHEELS_DIR/"
    cd ..
    rm -rf xformers
)

# PyG extensions: version patched to X.Y.Z+pt215cu132
build_pyg() {
    local repo=$1 version=$2
    git clone --recurse-submodules --branch "$version" --depth 1 "https://github.com/rusty1s/${repo}.git"
    (
        cd "$repo"
        sed -i "s/__version__ = '${version}'/__version__ = '${version}+${PT_VER}${CUDA}'/" setup.py
        FORCE_CUDA=1 pip wheel . --no-cache-dir --no-deps --no-build-isolation -w "$WHEELS_DIR/"
    )
    rm -rf "$repo"
}

build_pyg pytorch_cluster "$CLUSTER_VERSION"
build_pyg pytorch_sparse "$SPARSE_VERSION"
build_pyg pytorch_scatter "$SCATTER_VERSION"

(
    git clone --recurse-submodules --branch "v${TORCH_AO_VERSION}" --depth 1 https://github.com/pytorch/ao.git
    cd ao
    FORCE_CUDA=1 VERSION_SUFFIX="+${CUDA}" pip wheel . --no-cache-dir --no-deps --no-build-isolation -w "$WHEELS_DIR/"
    cd ..
    rm -rf ao
)

if [ "$UPLOAD" = "1" ]; then
    gh release upload "$RELEASE_TAG" "$WHEELS_DIR"/*.whl --repo "$WHEELS_REPO"
fi
