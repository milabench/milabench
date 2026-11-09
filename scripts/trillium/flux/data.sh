#!/usr/bin/env bash
# FLUX.1 data: NVIDIA's scripts/download.sub (--download, --preprocess, --cleanup) on a Trillium data-transfer node,
# the image through apptainer exec instead of srun with pyxis (compute nodes have no internet). On tri-dm1, one step
# at a time, in the background:
#   nohup bash scripts/trillium/flux/data.sh download > $SCRATCH/trillium/new/flux/logs/data-download.log 2>&1 &
# download: MLCommons' R2 downloader at 330c47a (wget --continue per file, then md5sum -c; rerun to resume) into data/,
#   JOBS files at a time (default: the downloader's, 4).
# preprocess: NVIDIA's to_webdataset.py (8 workers) and energon_prepare.py in the image, then the job's mount check
#   (mlperf_common.mountcheck against expected-mounts.csv: 230 files, 2,450,225,588,745 bytes).
# cleanup: deletes the raw sets, only after preprocess's mount check passed.
set -euo pipefail
F=${SCRATCH:?}/trillium/new/flux
D=$F/data CONT=$F/images/flux.sif
R2=https://raw.githubusercontent.com/mlcommons/r2-downloader/330c47a280cf1f87a52400ca3dc19bec9a77f8b8/mlc-r2-downloader.sh
unset PYTHONPATH PIP_CONFIG_FILE  # the login environment's, not the image's
mkdir -p "$D" && cd "$D"
case ${1:?download, preprocess or cleanup} in
download)
    for s in cc12m-preprocessed coco-preprocessed empty-encodings; do
        bash <(curl -fsS "$R2") ${JOBS:+-j "$JOBS"} "https://training.mlcommons-storage.org/metadata/flux-1-$s.uri"
    done ;;
preprocess)
    in_image() { apptainer exec -B "$D:/dataset" --pwd "$1" "$CONT" "${@:2}"; }
    mkdir -p energon
    in_image /dataset python /workspace/flux/scripts/to_webdataset.py --input_path /dataset/cc12m_preprocessed \
        --output_path /dataset/energon/train --num_workers 8
    in_image /dataset python /workspace/flux/scripts/to_webdataset.py --input_path /dataset/coco_preprocessed \
        --output_path /dataset/energon/val --num_workers 8
    in_image /dataset/energon python /workspace/flux/scripts/energon_prepare.py ./ --num-workers 8 \
        --template-dir /workspace/flux/scripts/dataset_template/
    cp -r empty_encodings energon/
    apptainer exec -B "$D/energon:/dataset/energon" --pwd /workspace/flux "$CONT" python3 -m mlperf_common.mountcheck \
        --expected_mounts_csv expected-mounts.csv --mounts_to_verify DATAROOT:/dataset/energon
    touch "$D/mountcheck-passed" ;;
cleanup)
    [ -e "$D/mountcheck-passed" ] || { echo "preprocess's mount check has not passed" >&2; exit 1; }
    rm -r "$D/cc12m_preprocessed" "$D/coco_preprocessed" ;;
esac
