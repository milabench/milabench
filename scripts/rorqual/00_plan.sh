#!/usr/bin/env bash
# Step 0, login node, no GPU: the node lists the G runs would use (aligned / crossed / mixed / leaf) and how many expert
# groups of each cross a switch. Check them against the table (G1 0, G2 all, G3 about 1 in 4) and the set sizes (50 for
# G1-G3 / G5, 64 for G4) before submitting anything: a 52-node aligned set means rg21701 is back (see 01_check_full.sh).
#   ./00_plan.sh "<healthy full-GPU hostlist, main node first>" [--group-nodes 2]
set -euo pipefail
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/_common.sh"
hosts=${1:?usage: $0 HOSTLIST [--group-nodes N]}
shift
exec "$MILABENCH_BASE/venv/torch/bin/python" "$WORKDIR/benchmarks/megatron/placement.py" --hosts "$hosts" "$@"
