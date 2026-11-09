#!/bin/bash
# OF3 step-1 training-data subset, under 10 GB (OF3_BRIEF.md "Space and downloads"): 150 PDB entries drawn by their
# training weight, the first 800 long distillation monomers, 400 RNA distillation monomers (subset.py), with the files
# they read and all reference molecules, from the public bucket s3://openfold3-data into $D/s3 (same layout as the
# bucket). The full PDB training cache (1.68 GB) is deleted once the subset caches are written. Rerunning resumes.
# Run on a data-transfer node: ssh tri-dm2 bash $SCRATCH/trillium/new/of3/src/scripts/trillium/of3/data.sh
set -euo pipefail
O=$SCRATCH/trillium/new/of3
D=$O/data
S=$O/src/scripts/trillium/of3
B=https://openfold3-data.s3.amazonaws.com
mkdir -p "$D/caches" "$D/s3"
cd "$D/caches"
if [ ! -s "$D/subset/keys.txt" ]; then
    curl -sS -C - -o rna_cache.json $B/rna_distillation_set/training_cache_RNA_monomer.json
    curl -sS -r 0-16777215 -o long_head.json $B/monomer_distillation_sets_v2/long_monomers/cache.json
    curl -sS -C - -o pdb_cache.json $B/pdb_training_set/dataset_caches/training_cache_with_templates.json
    "$O/venv/bin/python" -I "$S/subset.py" "$D/caches" "$D/subset" 150 800 400
    rm pdb_cache.json
fi
for p in pdb_training_set/preprocessed_pdb_data/standard/reference_mols monomer_distillation_sets_v2/shared/reference_mols \
        rna_distillation_set/reference_mols; do
    "$O/venv/bin/aws" s3 sync --no-sign-request --only-show-errors "s3://openfold3-data/$p/" "$D/s3/$p/"
done
"$O/venv/bin/python" -I "$S/fetch.py" "$D/subset/keys.txt" "$D/s3"
find "$D/s3" -type f -printf '%s\n' | awk '{n++; s+=$1} END {printf "s3: %d files, %.2f GB\n", n, s/1e9}'
ls -l "$D/caches" "$D/subset"
