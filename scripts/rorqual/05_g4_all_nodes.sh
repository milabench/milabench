#!/usr/bin/env bash
# Step 05: G4, EP8 x PP16 on 64 nodes (every healthy node minus the last partial group). Record the crossing count.
#   ./05_g4_all_nodes.sh [pass1|pass2] [-- extra sbatch args]
# Full training if 01_check_full.sh found it fits on this node set, else frozen experts. See _g.sh / _common.sh for the environment.
set -euo pipefail
exec "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/_g.sh" G4 "$@"
