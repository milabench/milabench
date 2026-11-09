#!/bin/bash
# FLUX.1 data conversion on one compute node, in the image, as NVIDIA's download.sub --preprocess (README.md, "Data"):
#   SET=val    the validation set only (3 shards of 10,000 samples, one per worker): the subset that times a shard
#   SET=train  the training set (110 shards) with WORKERS processes (NVIDIA's: 8), then energon_prepare.py, the empty
#              encodings, and data.sh check-conversion (needs the val shards in place)
#   SCRIPT=convert.sbatch scripts/trillium/flux/submit.sh convert-val 1 0:30:00 SET=val -- --nodelist=trig0145
#SBATCH --exclusive
#SBATCH --mem=0

set -euxo pipefail
ulimit -c 0  # no core dumps (one K3 abort wrote 681 GB, 2026-11-07); no srun steps here
for kv in "$@"; do export "$kv"; done
F=${SCRATCH}/trillium/new/flux
D=$F/data CONT=$F/images/flux.sif
unset PYTHONPATH PIP_CONFIG_FILE  # the login environment's, not the image's
export HF_HOME=/tmp/$USER-hf  # $HOME is not writable from jobs (job 1271975)
rm -f "$D/conversion-checked"  # the shards are about to change: cleanup must wait for a new check
# the image's NCCL from the host, as flux.sub binds it: the converter imports torch, which needs it
in_image() { apptainer exec -B "$D:/dataset,$F/nccl/lib" --env LD_LIBRARY_PATH="$F/nccl/lib" --pwd "$1" "$CONT" "${@:2}"; }
case ${SET:?val or train} in
val)
    time in_image /dataset python /workspace/flux/scripts/to_webdataset.py --input_path /dataset/coco_preprocessed \
        --output_path /dataset/energon/val --num_workers 8 ;;
train)
    time in_image /dataset python /workspace/flux/scripts/to_webdataset.py --input_path /dataset/cc12m_preprocessed \
        --output_path /dataset/energon/train --num_workers "${WORKERS:-8}"
    time in_image /dataset/energon python /workspace/flux/scripts/energon_prepare.py ./ --num-workers 8 \
        --template-dir /workspace/flux/scripts/dataset_template/
    mkdir -p "$D/energon/empty_encodings"  # the two encodings, not the downloader's md5 list beside them
    cp "$D/empty_encodings/t5_empty.npy" "$D/empty_encodings/clip_empty.npy" "$D/energon/empty_encodings/"
    bash "$F/src/scripts/trillium/flux/data.sh" check-conversion ;;
esac
