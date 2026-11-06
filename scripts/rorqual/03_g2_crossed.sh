#!/usr/bin/env bash
# Step 03: G2, EP8 crossed, same 50-node set as G1, 25 / 25 expert groups cross. Same NODELIST / EXCLUDE as G1.
#   ./03_g2_crossed.sh [pass1|pass2] [-- extra sbatch args]
# Full training if 01_check_full.sh found it fits on this node set, else frozen experts. See _g.sh / _common.sh for the environment.
set -euo pipefail
exec "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/_g.sh" G2 "$@"
