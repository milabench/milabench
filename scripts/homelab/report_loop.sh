#!/usr/bin/env bash
# Keep a run's HTML report up to date while it runs:  report_loop.sh <run dir> <report.html> [interval s]
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUN_DIR=$1 OUT=$2 EVERY=${3:-120}
RUN=$(basename "$RUN_DIR")
export HF_HUB_CACHE=${HF_HUB_CACHE:-${MILABENCH_BASE:-/opt/milabench}/data/hub}
update() {
    python3 -I "$HERE/report.py" "$RUN_DIR" > /dev/null 2>&1
    python3 -I "$HERE/report_html.py" "$RUN_DIR" --out "$OUT" > /dev/null 2>&1
}
while pgrep -f "run-name $RUN|homelab-queue" > /dev/null; do
    update
    sleep "$EVERY"
done
update  # the final state
