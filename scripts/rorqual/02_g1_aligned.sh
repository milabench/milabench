#!/usr/bin/env bash
# Step 02: G1, EP8 aligned, 50-node set, 0 / 25 expert groups cross a switch. Runs first: the reference.
#   ./02_g1_aligned.sh [pass1|pass2] [-- extra sbatch args]
# Full training if 01_check_full.sh found it fits on this node set, else frozen experts. See _g.sh / _common.sh for the environment.
set -euo pipefail
exec "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/_g.sh" G1 "$@"
