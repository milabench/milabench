#!/usr/bin/env bash
# Step 06: G5, G1 in FP8 (same set and layout). Only if there is time; the full rung was measured in BF16.
#   ./06_g5_fp8.sh [pass1|pass2] [-- extra sbatch args]
# Full training if 01_check_full.sh found it fits on this node set, else frozen experts. See _g.sh / _common.sh for the environment.
set -euo pipefail
exec "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/_g.sh" G5 "$@"
