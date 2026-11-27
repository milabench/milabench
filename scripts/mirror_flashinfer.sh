#!/bin/bash
# Mirror the flashinfer wheels pinned in a constraints file into a milabench/wheels release,
# so installs only depend on our own wheel cache (see platforms.toml [cuda.indexes]).
#
#   scripts/mirror_flashinfer.sh .pin/constraints.cuda132.torch2150.txt torch2.15-cu132
#
# Takes the exact flashinfer pins (cubin, python, jit-cache and jit-cache-sm*), downloads the
# matching x86_64 + aarch64 wheels from the upstream flashinfer GitHub release, and uploads
# the ones the target release does not already have.

set -euo pipefail

CONSTRAINTS=${1:?constraints file}
RELEASE=${2:?milabench/wheels release tag, e.g. torch2.15-cu132}
UPSTREAM=flashinfer-ai/flashinfer
TARGET=milabench/wheels

# flashinfer-jit-cache-sm90a==0.7.0.post1+cu130 -> flashinfer_jit_cache_sm90a-0.7.0.post1+cu130-*
mapfile -t PATTERNS < <(
    grep -E '^flashinfer-[a-z0-9-]+==' "$CONSTRAINTS" \
        | sed -E 's/[[:space:]].*//; s/==/ /' \
        | awk '{ gsub("-", "_", $1); print $1 "-" $2 "-*.whl" }'
)
if [ ${#PATTERNS[@]} -eq 0 ]; then
    echo "no flashinfer pins in $CONSTRAINTS" >&2
    exit 1
fi

VERSION=$(grep -E '^flashinfer-python==' "$CONSTRAINTS" | sed -E 's/.*==([^[:space:]+]+).*/\1/')
TAG="v$VERSION"

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

ARGS=()
for p in "${PATTERNS[@]}"; do ARGS+=(-p "$p"); done
gh release download "$TAG" -R "$UPSTREAM" -D "$WORK" "${ARGS[@]}"

EXISTING=$(gh release view "$RELEASE" -R "$TARGET" --json assets --jq '.assets[].name')
TODO=()
for f in "$WORK"/*.whl; do
    name=$(basename "$f")
    if grep -qxF "$name" <<< "$EXISTING"; then
        echo "already in $RELEASE: $name"
    else
        TODO+=("$f")
    fi
done

if [ ${#TODO[@]} -eq 0 ]; then
    echo "nothing to upload"
    exit 0
fi

echo "uploading ${#TODO[@]} wheel(s) to $TARGET@$RELEASE"
gh release upload "$RELEASE" -R "$TARGET" "${TODO[@]}"
