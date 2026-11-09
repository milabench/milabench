#!/usr/bin/env bash
# Build flux.sif, the FLUX.1 MLPerf image (flux.def: NVIDIA's b200_nemo Dockerfile), without Docker, on a Trillium
# data-transfer node (internet; the build is over 10 GB). Run on tri-dm1 in the background:
#   nohup bash scripts/trillium/flux/build.sh > $SCRATCH/trillium/new/flux/logs/build.log 2>&1 &
# Needs: images/pytorch-25.09-py3.sif (apptainer pull docker://nvcr.io/nvidia/pytorch:25.09-py3) and Florida's code
# (florida/, sparse checkout of mlcommons/training_results_v5.1 at 441d65f), both under $F (README.md).
set -euo pipefail
F=${SCRATCH:?}/trillium/new/flux
export APPTAINER_CACHEDIR=$F/apptainer/cache APPTAINER_TMPDIR=$F/apptainer/tmp
CODE=$F/florida/University-of-Florida/benchmarks/flux1/implementations/b200_nemo
# the login environment points pip at Compute Canada's wheelhouse and sets PYTHONPATH: neither belongs in the image
unset PIP_CONFIG_FILE PYTHONPATH
# --fakeroot: no /etc/subuid entry here, so Apptainer uses a root-mapped user namespace with the fakeroot wrapper
exec nice -n 19 apptainer build --fakeroot --build-arg BASE="$F/images/pytorch-25.09-py3.sif" --build-arg CODE="$CODE" \
    "$F/images/flux.sif" "$(dirname "$(readlink -f "$0")")/flux.def"
