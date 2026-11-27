#!/usr/bin/env bash
# Submit the G blocks one after the other (afterany: a failed block does not stop the next), after 01_check_full.sh
# has finished (its rung files must exist). Order of the window: pass 1 of G1-G4, pass 2 of G1-G4, then G5 pass 1 / 2.
# Use the same NODELIST / EXCLUDE for all of them.
#
#   ./all.sh [first [last]]     block numbers in that order, default 1 10   (e.g. ./all.sh 1 4 = pass 1 of G1-G4)
#   ./all.sh --list             what would be submitted
#
# NOTE: the window interleaves other experiments between these blocks: chained like this the GLM blocks run back to
# back. To follow the queue, submit the numbered scripts one at a time at their turn instead.
set -euo pipefail
here=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")
source "$here/_common.sh"

blocks=("02_g1_aligned.sh pass1" "03_g2_crossed.sh pass1" "04_g3_mixed.sh pass1" "05_g4_all_nodes.sh pass1"
        "02_g1_aligned.sh pass2" "03_g2_crossed.sh pass2" "04_g3_mixed.sh pass2" "05_g4_all_nodes.sh pass2"
        "06_g5_fp8.sh pass1" "06_g5_fp8.sh pass2")
if [ "${1:-}" = --list ]; then
    for i in "${!blocks[@]}"; do echo "$((i + 1)) ${blocks[$i]}"; done
    exit 0
fi
first=${1:-1}; last=${2:-${#blocks[@]}}

after=${AFTER:-}
for i in $(seq "$first" "$last"); do
    read -r script pass <<< "${blocks[$((i - 1))]}"
    echo "== $i: $script $pass"
    out=$(AFTER=$after NO_TAIL=1 "$here/$script" "$pass" 2>&1 | tee /dev/stderr)
    id=$(job_id "$out")
    [ -n "$id" ] || [ -n "${TEST_ONLY:-}" ] || { echo "no job id from $script $pass, stopping" >&2; exit 1; }
    after=${id:-$after}
done
