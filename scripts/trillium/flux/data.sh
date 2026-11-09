#!/usr/bin/env bash
# FLUX.1 data: NVIDIA's scripts/download.sub steps, each where it can run (README.md, "Data"):
#   tri-dm1 (internet):  data.sh download, then data.sh check-download
#   one compute node:    convert.sbatch (to_webdataset.py, energon_prepare.py), which ends with data.sh check-conversion
#   login node:          data.sh cleanup, which deletes the raw sets only after both checks passed
# download: MLCommons' R2 downloader at 330c47a (wget --continue per file, then md5sum -c; rerun to resume) into data/,
#   JOBS files at a time (default: the downloader's, 4). In the background:
#   nohup bash scripts/trillium/flux/data.sh download > $SCRATCH/trillium/new/flux/logs/data-download.log 2>&1 &
# check-download: per set, the files present against the downloader's md5 list (count), every file's size against the
#   bucket's (HTTP HEAD), and the downloader's own md5 verification in its log (3 sets).
# check-conversion: every file of NVIDIA's expected-mounts.csv by size (230 files, 2,450,225,588,745 bytes), and
#   energon's metadata read back: 110 train shards with 1,099,776 samples, 3 val shards with 29,696 (the bucket's counts).
# cleanup: rm -r data/cc12m_preprocessed data/coco_preprocessed, diskusage_report before and after.
set -euo pipefail
F=${SCRATCH:?}/trillium/new/flux
D=$F/data CONT=$F/images/flux.sif
R2=https://raw.githubusercontent.com/mlcommons/r2-downloader/330c47a280cf1f87a52400ca3dc19bec9a77f8b8/mlc-r2-downloader.sh
unset PYTHONPATH PIP_CONFIG_FILE  # the login environment's, not the image's
mkdir -p "$D" && cd "$D"
case ${1:?download, check-download, check-conversion or cleanup} in
download)
    for s in cc12m-preprocessed coco-preprocessed empty-encodings; do
        bash <(curl -fsS "$R2") ${JOBS:+-j "$JOBS"} "https://training.mlcommons-storage.org/metadata/flux-1-$s.uri"
    done ;;
check-download)
    rm -f "$D/download-checked"
    B=https://training.mlcommons-storage.org/flux_1/datasets
    for s in cc12m_preprocessed coco_preprocessed empty_encodings; do
        m=$s/flux-1-${s//_/-}.md5  # the downloader's list, copied into the set's directory
        listed=$(grep -c . "$m")
        present=$(find "$s" -type f ! -name '*.md5' | wc -l)
        bad=$(sed 's/^[a-f0-9]\{32\}[[:space:]]*//; s,^\./,,' "$m" | xargs -P 8 -I{} bash -c '
            r=$(curl -sfI "$1/$3" | tr -d "\r" | awk "tolower(\$1) == \"content-length:\" {print \$2}")
            l=$(stat -c %s "$2/$3" 2>/dev/null || echo missing)
            if [ "$r" != "$l" ]; then echo "$2/$3: bucket $r, local $l" >&2; echo bad; fi' _ "$B/$s" "$s" {} | wc -l)
        echo "$s: $listed files listed, $present present, $bad size mismatches"
        [ "$listed" -eq "$present" ] || exit 1
        [ "$bad" -eq 0 ] || exit 1
    done
    verified=$(grep -c "Checksum verification completed successfully" "$F/logs/data-download.log" || true)
    echo "downloader md5 verification passed for $verified of 3 sets"
    [ "$verified" -eq 3 ]
    date > "$D/download-checked" ;;
check-conversion)
    rm -f "$D/conversion-checked"
    # energon imports torch, which needs the image's NCCL from the host (as flux.sub binds it)
    apptainer exec -B "$D/energon:/dataset/energon,$F/nccl/lib" --env LD_LIBRARY_PATH="$F/nccl/lib" --pwd /workspace/flux \
        "$CONT" python3 - <<'PY'
import csv, os, sys
from megatron.energon.epathlib import EPath
from megatron.energon.flavors.webdataset.metadata import WebdatasetMeta

root, bad, files, size = "/dataset/energon", 0, 0, 0
for row in csv.DictReader(open("expected-mounts.csv")):
    if row["type"] != "file":
        continue
    path = os.path.join(root, row["relative_path"])
    found = os.path.getsize(path) if os.path.isfile(path) else None
    files, size = files + 1, size + (found or 0)
    if found != int(row["num_bytes"]):
        bad += 1
        print(f"{row['relative_path']}: expected {row['num_bytes']} bytes, found {found}")
print(f"expected-mounts.csv: {files} files, {size} bytes found, {bad} mismatches")
for split, shards, samples in (("train", 110, 1099776), ("val", 3, 29696)):
    meta = WebdatasetMeta.from_config(EPath(root), split_part=split)
    count = sum(shard.count for shard in meta.shards)
    print(f"energon {split}: {len(meta.shards)} shards, {count} samples (expected {shards}, {samples})")
    bad += (len(meta.shards), count) != (shards, samples)
sys.exit(1 if bad else 0)
PY
    date > "$D/conversion-checked" ;;
cleanup)
    [ -e "$D/download-checked" ] && [ -e "$D/conversion-checked" ] \
        || { echo "check-download and check-conversion have not both passed" >&2; exit 1; }
    diskusage_report | grep -E "Description|/scratch \(user"
    rm -r "$D/cc12m_preprocessed" "$D/coco_preprocessed"
    diskusage_report | grep -E "Description|/scratch \(user" ;;
esac
