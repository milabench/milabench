#!/usr/bin/env bash
# Step 04: G3, EP8 mixed, same 50-node set as G1, 6 / 25 expert groups cross (3 swaps). Same NODELIST / EXCLUDE as G1.
#   ./04_g3_mixed.sh [pass1|pass2] [-- extra sbatch args]
# Full training if 01_check_full.sh found it fits on this node set, else frozen experts. See _g.sh / _common.sh for the environment.
set -euo pipefail
exec "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/_g.sh" G3 "$@"
