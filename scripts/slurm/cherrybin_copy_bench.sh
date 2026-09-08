#!/bin/bash
# Benchmark cherrybin archive copy speed: shared (network) FS -> local scratch.
#
# Copies both halves of the archive from CHERRYBIN_DB down to node-local
# /tmp and times each: the sqlite ledger (archive.db, small) and the
# append-only payload (archive.db.blobs, where the actual file bytes
# live -- usually the dominant part by far). Reported separately so a
# slow ledger copy doesn't hide a missing/slow blobs copy or vice versa.
#
# Override the archive location with CHERRYBIN_DB, same as shared_cherrybin.sh.

set -ex

export PATH="$HOME/.bin/:$PATH"

# ===
OUTPUT_DIRECTORY=$(scontrol show job "$SLURM_JOB_ID" --json | jq -r '.jobs[0].standard_output' | xargs dirname)
mkdir -p "$OUTPUT_DIRECTORY/meta"
scontrol show job --json "$SLURM_JOB_ID" | jq '.jobs[0]' > "$OUTPUT_DIRECTORY/meta/info.json"
touch "$SLURM_SUBMIT_DIR/.no_report"
# ===

export MILABENCH_SHARED="$HOME/scratch/shared"
export CHERRYBIN_DB="${CHERRYBIN_DB:-$MILABENCH_SHARED/archive.db}"

LOCAL_DIR="/tmp/$SLURM_JOB_ID/cherrybin_bench"
mkdir -p "$LOCAL_DIR"

# MB/s = bytes / 1e6 / seconds, no bc dependency.
report() {
    local label="$1" bytes="$2" seconds="$3"
    awk -v label="$label" -v bytes="$bytes" -v seconds="$seconds" 'BEGIN {
        if (bytes <= 0 || seconds <= 0) {
            printf "[%s] skipped (nothing to copy)\n", label
        } else {
            printf "[%s] %.2f GB in %.1fs -> %.0f MB/s\n", label, bytes/1e9, seconds, bytes/1e6/seconds
        }
    }'
}

timed_copy() {
    local label="$1" src="$2" dest="$3"
    if [ ! -f "$src" ]; then
        echo "[$label] MISSING: $src does not exist -- not copied"
        return 0
    fi
    local bytes
    bytes=$(stat --format=%s "$src")
    local start end
    start=$(date +%s.%N)
    rsync --inplace "$src" "$dest"
    end=$(date +%s.%N)
    report "$label" "$bytes" "$(awk -v a="$start" -v b="$end" 'BEGIN { print b - a }')"
}

echo "=== shared -> local: $CHERRYBIN_DB ==="
timed_copy "ledger " "$CHERRYBIN_DB" "$LOCAL_DIR/archive.db"
timed_copy "payload" "${CHERRYBIN_DB}.blobs" "$LOCAL_DIR/archive.db.blobs"

rm -rf "$LOCAL_DIR"
