#!/bin/bash
#
# Setup milabench on a new machine
#
#   1. Verify that there is at least 2 TB of free space for milabench
#   2. Look for unused high capacity storage drives
#   3. Make a big RAID0 partition with the unused drives
#   4. Use it as the root where all milabench folders will be
#   5. Install this milabench checkout in a virtual environment in that root
#
# The script runs in dry-run mode by default; pass --yes to actually format the drives.
#
# Usage:
#
#   bash scripts/setup_new_machine.sh                  # dry-run, show what would be done
#   bash scripts/setup_new_machine.sh --yes            # do it
#   bash scripts/setup_new_machine.sh --root /data --min-disk-size 1T --yes
#
# Layout:
#
#   $ROOT/env           virtual environment with milabench installed
#   $ROOT/results       MILABENCH_BASE (data/, cache/, runs/, venv/, extra/)
#   $ROOT/env.sh        environment to source before running milabench
#

set -euo pipefail

ROOT=/mnt/milabench
MIN_FREE=2T
MIN_DISK_SIZE=500G
FS=ext4
MD_NAME=milabench
MILABENCH_SOURCE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DRY_RUN=1
SKIP_STORAGE=0
SKIP_INSTALL=0

usage() {
    sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'
    cat <<EOF
Options:
    --root PATH           Where milabench folders will live (default: $ROOT)
    --min-free SIZE       Minimum free space required (default: $MIN_FREE)
    --min-disk-size SIZE  Ignore drives smaller than SIZE (default: $MIN_DISK_SIZE)
    --fs ext4|xfs         Filesystem of the RAID partition (default: $FS)
    --skip-storage        Do not touch the drives, only check free space under --root
    --skip-install        Only setup the storage
    --yes                 Actually do it (default is dry-run)
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --root)          ROOT="$2"; shift 2 ;;
        --min-free)      MIN_FREE="$2"; shift 2 ;;
        --min-disk-size) MIN_DISK_SIZE="$2"; shift 2 ;;
        --fs)            FS="$2"; shift 2 ;;
        --skip-storage)  SKIP_STORAGE=1; shift ;;
        --skip-install)  SKIP_INSTALL=1; shift ;;
        --yes)           DRY_RUN=0; shift ;;
        -h|--help)       usage; exit 0 ;;
        *)               echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

USER_NAME="${SUDO_USER:-$USER}"
SUDO=""
if [[ $EUID -ne 0 ]]; then
    SUDO="sudo"
fi

log()  { echo -e "\033[1;34m==>\033[0m $*"; }
warn() { echo -e "\033[1;33mWARN:\033[0m $*" >&2; }
die()  { echo -e "\033[1;31mERROR:\033[0m $*" >&2; exit 1; }

# Run a command, or only print it in dry-run mode
run() {
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "    [dry-run] $*"
    else
        echo "    + $*"
        "$@"
    fi
}

to_bytes() {
    numfmt --from=iec "${1%B}"
}

human() {
    numfmt --to=iec --suffix=B "$1"
}

# Free bytes available on the filesystem holding PATH (or its closest existing parent)
free_bytes() {
    local path="$1"
    while [[ ! -e "$path" ]]; do
        path=$(dirname "$path")
    done
    df --output=avail -B1 "$path" | tail -1 | tr -d ' '
}

# A disk is unused if it has no partitions, no filesystem/RAID/LVM signature,
# is not mounted, is not used as swap and is not held by another device (md, dm)
is_unused_disk() {
    local dev="/dev/$1"

    # has partitions or is used by md/dm
    [[ $(lsblk -nro NAME "$dev" | wc -l) -eq 1 ]] || return 1
    [[ -z "$(ls -A /sys/block/$1/holders 2>/dev/null)" ]] || return 1

    # has a signature (filesystem, raid member, LVM PV, ...)
    [[ -z "$($SUDO blkid -p -o value -s TYPE "$dev" 2>/dev/null)" ]] || return 1
    [[ -z "$($SUDO wipefs -n "$dev" 2>/dev/null | tail -n +2)" ]] || return 1

    # mounted or swap
    grep -q "^$dev " /proc/mounts && return 1
    grep -q "^$dev " /proc/swaps && return 1

    # read-only or removable
    [[ "$(cat /sys/block/$1/ro)" == "0" ]] || return 1
    [[ "$(cat /sys/block/$1/removable)" == "0" ]] || return 1

    return 0
}

# RAID0 is limited by its smallest/slowest member, so only keep the group of
# identically sized drives that provides the most capacity
find_unused_disks() {
    local min_bytes
    min_bytes=$(to_bytes "$MIN_DISK_SIZE")

    lsblk -dnbo NAME,SIZE,TYPE | while read -r name size type; do
        [[ "$type" == "disk" ]] || continue
        [[ "$size" -ge "$min_bytes" ]] || continue
        if is_unused_disk "$name"; then
            echo "$size $name"
        fi
    done | awk '
        { names[$1] = names[$1] " " $2; total[$1] += $1 }
        END {
            for (s in total) if (total[s] > best_total) { best_total = total[s]; best = s }
            n = split(names[best], out, " ")
            for (i = 1; i <= n; i++) print out[i]
        }'
}

setup_storage() {
    local min_free
    min_free=$(to_bytes "$MIN_FREE")

    if mountpoint -q "$ROOT" 2>/dev/null; then
        log "$ROOT is already mounted ($(human "$(free_bytes "$ROOT")") free), skipping RAID setup"
        return
    fi

    log "Looking for unused drives larger than $MIN_DISK_SIZE"
    mapfile -t DISKS < <(find_unused_disks)

    if [[ ${#DISKS[@]} -eq 0 ]]; then
        local avail
        avail=$(free_bytes "$ROOT")
        if [[ "$avail" -ge "$min_free" ]]; then
            warn "No unused drives found; using existing filesystem for $ROOT ($(human "$avail") free)"
            run $SUDO mkdir -p "$ROOT"
            run $SUDO chown "$USER_NAME:" "$ROOT"
            return
        fi
        die "No unused drives found and only $(human "$avail") free for $ROOT (need $MIN_FREE)"
    fi

    local total=0
    for d in "${DISKS[@]}"; do
        local size
        size=$(lsblk -dnbo SIZE "/dev/$d")
        total=$((total + size))
        echo "    /dev/$d  $(human "$size")  $(lsblk -dno MODEL "/dev/$d" | xargs)"
    done
    log "Found ${#DISKS[@]} unused drives, total $(human "$total")"

    if [[ "$total" -lt "$min_free" ]]; then
        die "Unused drives only provide $(human "$total"), need $MIN_FREE"
    fi

    local packages=(mdadm)
    [[ "$FS" == "xfs" ]] && packages+=(xfsprogs)
    for p in "${packages[@]}"; do
        if ! dpkg -s "$p" >/dev/null 2>&1; then
            run $SUDO apt-get install -y "$p"
        fi
    done

    local devices=()
    for d in "${DISKS[@]}"; do
        devices+=("/dev/$d")
    done

    local target
    if [[ ${#devices[@]} -eq 1 ]]; then
        log "Single drive, formatting it directly"
        target="${devices[0]}"
    else
        log "Creating RAID0 /dev/md/$MD_NAME with ${devices[*]}"
        target="/dev/md/$MD_NAME"
        [[ -e "$target" ]] && die "$target already exists"
        run $SUDO mdadm --create "$target" --level=0 --raid-devices=${#devices[@]} \
            --chunk=512 --name="$MD_NAME" --run "${devices[@]}"

        # Persist the array across reboots
        if [[ $DRY_RUN -eq 0 ]]; then
            $SUDO mdadm --detail --brief "$target" | $SUDO tee -a /etc/mdadm/mdadm.conf
        else
            echo "    [dry-run] mdadm --detail --scan >> /etc/mdadm/mdadm.conf"
        fi
        run $SUDO update-initramfs -u
    fi

    log "Formatting $target as $FS"
    case "$FS" in
        ext4) run $SUDO mkfs.ext4 -F -m 0 -L milabench -E lazy_itable_init=1,lazy_journal_init=1 "$target" ;;
        xfs)  run $SUDO mkfs.xfs -f -L milabench "$target" ;;
        *)    die "Unsupported filesystem $FS" ;;
    esac

    log "Mounting $target on $ROOT"
    run $SUDO mkdir -p "$ROOT"
    if [[ $DRY_RUN -eq 0 ]]; then
        local uuid
        uuid=$($SUDO blkid -o value -s UUID "$target")
        echo "UUID=$uuid $ROOT $FS defaults,noatime,nofail 0 2" | $SUDO tee -a /etc/fstab
        $SUDO systemctl daemon-reload || true
    else
        echo "    [dry-run] echo 'UUID=<uuid> $ROOT $FS defaults,noatime,nofail 0 2' >> /etc/fstab"
    fi
    run $SUDO mount "$ROOT"
    run $SUDO chown "$USER_NAME:" "$ROOT"
}

check_free_space() {
    local min_free avail
    min_free=$(to_bytes "$MIN_FREE")
    avail=$(free_bytes "$ROOT")

    if [[ "$avail" -lt "$min_free" ]]; then
        if [[ $DRY_RUN -eq 1 ]]; then
            warn "$ROOT has $(human "$avail") free (need $MIN_FREE); expected in dry-run before the RAID is created"
            return
        fi
        die "$ROOT has $(human "$avail") free, need $MIN_FREE"
    fi
    log "$ROOT has $(human "$avail") free"
}

detect_gpu_arch() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo cuda
    elif command -v rocm-smi >/dev/null 2>&1 || command -v amd-smi >/dev/null 2>&1; then
        echo rocm
    elif command -v xpu-smi >/dev/null 2>&1; then
        echo xpu
    elif command -v hl-smi >/dev/null 2>&1; then
        echo hpu
    else
        echo cpu
    fi
}

install_milabench() {
    local arch
    arch=$(detect_gpu_arch)
    log "Detected accelerator: $arch"

    log "Installing system dependencies"
    run $SUDO apt-get update -y
    run $SUDO apt-get install -y --no-install-recommends \
        git build-essential curl python3 python3-venv python3-pip python-is-python3 libglib2.0-0 gcc g++

    if ! command -v uv >/dev/null 2>&1 && [[ ! -x "$HOME/.local/bin/uv" ]]; then
        log "Installing uv"
        if [[ $DRY_RUN -eq 0 ]]; then
            curl -LsSf https://astral.sh/uv/install.sh | sh
        else
            echo "    [dry-run] curl -LsSf https://astral.sh/uv/install.sh | sh"
        fi
    fi
    export PATH="$HOME/.local/bin:$PATH"

    log "Creating milabench folders in $ROOT"
    run mkdir -p "$ROOT/results"/{data,cache,runs,extra}
    log "Using milabench from $MILABENCH_SOURCE"

    if [[ ! -d "$ROOT/env" ]]; then
        run uv venv --python 3.12 "$ROOT/env"
    fi
    if [[ "$arch" == "cuda" ]]; then
        run uv pip install --python "$ROOT/env/bin/python" -e "$MILABENCH_SOURCE[cuda]"
    else
        run uv pip install --python "$ROOT/env/bin/python" -e "$MILABENCH_SOURCE"
    fi

    log "Writing $ROOT/env.sh"
    local envfile
    envfile=$(cat <<EOF
# Source this file before running milabench
export MILABENCH_ROOT=$ROOT
export MILABENCH_BASE=$ROOT/results
export MILABENCH_CONFIG=$MILABENCH_SOURCE/config/standard.yaml
export MILABENCH_GPU_ARCH=$arch
# export MILABENCH_HF_TOKEN=

# Keep 3rd party caches away from ~/.cache
export XDG_CACHE_HOME=\$MILABENCH_BASE/cache
export HF_HOME=\$MILABENCH_BASE/cache/huggingface
export TORCH_HOME=\$MILABENCH_BASE/cache/torch
export UV_CACHE_DIR=\$MILABENCH_BASE/cache/uv
export PIP_CACHE_DIR=\$MILABENCH_BASE/cache/pip

export PATH=\$HOME/.local/bin:\$PATH
source $ROOT/env/bin/activate
EOF
)
    if [[ $DRY_RUN -eq 0 ]]; then
        echo "$envfile" > "$ROOT/env.sh"
    else
        echo "$envfile" | sed 's/^/    [dry-run] /'
    fi
}

main() {
    if [[ $DRY_RUN -eq 1 ]]; then
        warn "Dry-run mode, nothing will be modified. Pass --yes to apply."
    fi

    if [[ $SKIP_STORAGE -eq 0 ]]; then
        setup_storage
    fi
    check_free_space

    if [[ $SKIP_INSTALL -eq 0 ]]; then
        install_milabench
    fi

    log "Done"
    cat <<EOF

Next steps:

    source $ROOT/env.sh
    export MILABENCH_HF_TOKEN=<token>
    milabench install
    milabench prepare
    milabench run

EOF
}

main
