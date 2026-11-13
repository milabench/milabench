#!/bin/bash
# Trillium B200: fail if a GPU of this node has an enforced power limit below 1000 W. Run on every node when a job starts:
# on 2026-11-05, 8 nodes had all GPUs at 200 W (120 MHz; every rank waits for them, 26.7 s per step against 3.0 s). Other GPU models pass.
set -o pipefail
nvidia-smi --query-gpu=name,enforced.power.limit --format=csv,noheader,nounits | awk -F', ' -v host="$(hostname -s)" '
    /B200/ && $2 < 1000 {capped = capped " GPU" NR-1 "=" $2 "W"}
    END {if (capped) {print "[power-check] CAPPED " host ":" capped; exit 1}}'
