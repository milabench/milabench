#!/bin/bash
# OF3 training-data subset (OF3_BRIEF.md "Space and downloads"): N_PDB PDB entries (DRAW uniform or weighted,
# subset.py), the first N_MONO long distillation monomers (read in order: monomers.py), 400 RNA distillation monomers, with the files they read
# and all reference molecules, from the public bucket s3://openfold3-data into $D/s3 (same layout as the bucket); the
# subset's caches and key list in $D/NAME. With OLD (an earlier subset's directory), the files only OLD read are deleted
# from $D/s3 afterwards. Rerunning resumes.
# Step 1: data.sh subset weighted 150 800. Step 2: data.sh subset-u uniform 700 800 subset. OF3-01: data.sh subset-u2
# uniform 700 4330 (adds 44,805 files, 9.98 GB).
# Run on a data-transfer node: ssh tri-dm2 bash $SCRATCH/trillium/new/of3/src/scripts/trillium/of3/data.sh NAME DRAW N_PDB N_MONO [OLD]
set -euo pipefail
NAME=$1 DRAW=$2 N_PDB=$3 N_MONO=$4 OLD=${5:-}
O=$SCRATCH/trillium/new/of3
D=$O/data
S=$O/src/scripts/trillium/of3
B=https://openfold3-data.s3.amazonaws.com
mkdir -p "$D/caches" "$D/s3"
cd "$D/caches"
if [ ! -s "$D/$NAME/keys.txt" ]; then
    curl -sS -C - -o rna_cache.json $B/rna_distillation_set/training_cache_RNA_monomer.json
    curl -sS -r 0-16777215 -o long_head.json $B/monomer_distillation_sets_v2/long_monomers/cache.json
    "$O/venv/bin/python" -I "$S/subset.py" "$D/caches" "$D/$NAME" "$N_PDB" "$N_MONO" 400 0 "$DRAW"
fi
for p in pdb_training_set/preprocessed_pdb_data/standard/reference_mols monomer_distillation_sets_v2/shared/reference_mols \
        rna_distillation_set/reference_mols; do
    "$O/venv/bin/aws" s3 sync --no-sign-request --only-show-errors "s3://openfold3-data/$p/" "$D/s3/$p/"
done
"$O/venv/bin/python" -I "$S/fetch.py" "$D/$NAME/keys.txt" "$D/s3"
if [ -n "$OLD" ]; then
    comm -23 <(sort "$D/$OLD/keys.txt") <(sort "$D/$NAME/keys.txt") | sed "s|^|$D/s3/|" > "$D/$NAME/deleted.txt"
    xargs -a "$D/$NAME/deleted.txt" rm -f
    echo "deleted $(wc -l < "$D/$NAME/deleted.txt") files only $OLD read"
fi
find "$D/s3" -type f -printf '%s\n' | awk '{n++; s+=$1} END {printf "s3: %d files, %.2f GB\n", n, s/1e9}'
ls -l "$D/caches" "$D/$NAME"
