#!/bin/bash
# One-off follow-up to job 1258790 (2026-11-07), on its 50 nodes, one step after another:
#   d3-50       collectives.py at 50 nodes only, size sweep at 50 (1258790 never reached 50)
#   d3-ladder   collectives.py 1, 2, 4, 8, 13, 26, 50, sweep at 2: its 26 is the subgroup that hung in 1258790
#   d3-26-x/y   collectives.py on two fixed 26-node sets, each as a whole step
#   pair-*      p2p_probe2.py on one pair of 1258790's D2, alone (per-GPU spread in the log)
#   together-*  p2p_probe2.py on 2 and 3 slow pairs of 1258790's D2 round 1; nodes nested in host order,
#               so the probe's round 1 is that set of pairs
#   d3-26-ya/yb Y with only trig0122 or only trig0123 swapped, run only if one of x, y is slow and the other not
# Each step has an srun --time, so a hang costs minutes and the next step runs. IB counters of the
# compute ports on the step's nodes before and after each step: logs/ib/<step>-{before,after}.txt.
#
#   /opt/slurm/bin/.sbatch --export=NONE --get-user-env --nodes=50 --nodelist=<50 nodes> \
#       --output=$SCRATCH/trillium/nccl/%x-%j.out scripts/trillium/net/followup.sbatch
#
#SBATCH --job-name=tri-net-followup
#SBATCH --account=rrg-bengioy-ad
#SBATCH --partition=compute_b200_full_node
#SBATCH --reservation=b200-friendly-users
#SBATCH --qos=bump
#SBATCH --gpus-per-node=b200:8
#SBATCH --exclusive
#SBATCH --time=0:35:00

set -uo pipefail

NET_DIR=${NET_DIR:-$SLURM_SUBMIT_DIR/scripts/trillium/net}
NET_OUT=${NET_OUT:-$SCRATCH/trillium/nccl/$SLURM_JOB_ID}
NET_PY=${NET_PY:-$SCRATCH/tt38/output/venv/torchtitan_native/bin/python}
export NET_DIR
mkdir -p "$NET_OUT/logs/ib"
unset PYTHONPATH PIP_CONFIG_FILE
set +u; module load ${NET_MODULES-cuda/12.9} || echo "module load failed; continuing"; set -u
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
GPUS=$(nvidia-smi -L | wc -l)
CPUS=$(( SLURM_CPUS_ON_NODE / GPUS ))
ALL=$SLURM_JOB_NODELIST
echo "job $SLURM_JOB_ID: $SLURM_NNODES nodes x $GPUS GPUs, out $NET_OUT"
scontrol show hostnames "$ALL" > "$NET_OUT/logs/nodes.txt"

count() { scontrol show hostnames "$1" | wc -l; }

snap() {  # snap <file> <nodes>: IB counters of the four compute ports
    srun --nodes="$(count "$2")" --nodelist="$2" --ntasks-per-node=1 --time=1 bash -c '
        for d in mlx5_0 mlx5_1 mlx5_4 mlx5_9; do
            for c in port_xmit_wait symbol_error link_downed port_rcv_errors link_error_recovery port_xmit_data port_rcv_data; do
                echo "$(hostname -s) $d $c $(cat /sys/class/infiniband/$d/ports/1/counters/$c 2>&1)"
            done
        done' > "$NET_OUT/logs/ib/$1" 2>&1
}

port=29600
declare -A RC
step() {  # step <tag> <nodes> <minutes> <script> [args]: one process per GPU on <nodes>, output in $NET_OUT/<tag>/
    local tag=$1 nodes=$2 limit=$3 script=$4 n
    shift 4
    n=$(count "$nodes")
    port=$((port + 1))
    mkdir -p "$NET_OUT/$tag"
    snap "$tag-before.txt" "$nodes"
    echo "[$(date +%T)] $tag: $n nodes, $script $* (limit $limit min)"
    MASTER_ADDR=$(scontrol show hostnames "$nodes" | head -1) MASTER_PORT=$port NET_OUT=$NET_OUT/$tag \
        srun --nodes="$n" --nodelist="$nodes" --ntasks-per-node="$GPUS" --cpus-per-task="$CPUS" \
        --kill-on-bad-exit=1 --time="$limit" "$NET_PY" "$NET_DIR/$script" "$@" > "$NET_OUT/$tag/log.txt" 2>&1
    RC[$tag]=$?
    snap "$tag-after.txt" "$nodes"
    echo "[$(date +%T)] $tag: exit ${RC[$tag]}"
}

slow() {  # slow <tag>: the step failed, or its 1 GiB all-gather is missing or under 100 GB/s
    [[ ${RC[$1]} -ne 0 ]] && return 0
    "$NET_PY" -I -c '
import json, sys
try:
    rows = [json.loads(line) for line in open(sys.argv[1])]
except OSError:
    sys.exit(0)
bw = [r["busbw_gbs"] for r in rows if r["collective"] == "all_gather" and r["size_bytes"] == 2**30]
sys.exit(0 if len(bw) < 2 or min(bw) < 100 else 1)' "$NET_OUT/$1/collectives.jsonl"
}

X="trig[0101-0113,0115-0121,0124-0129]"   # 1250124's 26 nodes; trig0114 (maintenance) replaced by trig0129
Y="trig[0101-0113,0115-0127]"             # 1258790's ladder at 26, which hung
YA="trig[0101-0113,0115-0121,0123-0128]"  # Y with trig0122 replaced by trig0128
YB="trig[0101-0113,0115-0122,0124-0128]"  # Y with trig0123 replaced by trig0128

step d3-50 "$ALL" 4 collectives.py --nodes 50
step d3-ladder "$ALL" 5 collectives.py --sweep-nodes 2
step d3-26-x "$X" 3 collectives.py --nodes 26 --sweep-nodes 0
step d3-26-y "$Y" 3 collectives.py --nodes 26 --sweep-nodes 0

# 1258790's D2: slow (~96 GB/s) trig0103-0150, 0107-0146, 0109-0144, 0118-0135, 0120-0133 (round 1), 0151-0147;
# fast (~189) trig0101-0152 (round 1), 0125-0126, 0112-0139.
for p in trig0103,trig0150 trig0107,trig0146 trig0109,trig0144 trig0118,trig0135 trig0120,trig0133 trig0147,trig0151 \
         trig0101,trig0152 trig0125,trig0126 trig0112,trig0139; do
    step "pair-${p//,/-}" "$p" 2 p2p_probe2.py
done
step together-2 "trig[0103,0107,0146,0150]" 2 p2p_probe2.py
step together-3 "trig[0109,0118,0120,0133,0135,0144]" 3 p2p_probe2.py

slow d3-26-x; sx=$?
slow d3-26-y; sy=$?
if [[ $sx -ne $sy ]]; then
    step d3-26-ya "$YA" 3 collectives.py --nodes 26 --sweep-nodes 0
    step d3-26-yb "$YB" 3 collectives.py --nodes 26 --sweep-nodes 0
else
    echo "[$(date +%T)] d3-26-x slow: $((1 - sx)), d3-26-y slow: $((1 - sy)); swap sets not run"
fi
echo "[$(date +%T)] done"
