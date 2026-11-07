#!/bin/bash
# One-off (2026-11-07): D3 collectives slow at 26 and 50 nodes in 1258790 and 1259019 (~36 GB/s against 170-192).
# Records the environment on every node, then D3 at 26 (the first 26 nodes, a subgroup) and 50 on the job's
# nodes, 1 GiB, with NCCL_DEBUG=INFO (INIT, ENV, TUNING) to per-process files, under NCCL variants:
#   base      as before (also 13 nodes, and all-to-all)
#   ring      NCCL_ALGO=Ring
#   tree      NCCL_ALGO=Tree, all-reduce only (the only one of the three with a tree algorithm)
#   nvls0     NCCL_NVLS_ENABLE=0
#   collnet0  NCCL_COLLNET_ENABLE=0
#   base-end  base again, to see whether it changes over the job
# Each step has an srun --time (a hang costs 3 minutes, then the next step runs); IB counters of the compute
# ports before and after each step: logs/ib/<step>-{before,after}.txt. Environment: env/<host>.txt.
#
#   /opt/slurm/bin/.sbatch --export=NONE --get-user-env --nodes=50 --nodelist=<50 nodes> \
#       --output=$SCRATCH/trillium/nccl/%x-%j.out scripts/trillium/net/nccl_variants.sbatch
#
#SBATCH --job-name=tri-net-nccl-variants
#SBATCH --account=rrg-bengioy-ad
#SBATCH --partition=compute_b200_full_node
#SBATCH --reservation=b200-friendly-users
#SBATCH --qos=bump
#SBATCH --gpus-per-node=b200:8
#SBATCH --exclusive
#SBATCH --time=0:25:00

set -uo pipefail

NET_DIR=${NET_DIR:-$SLURM_SUBMIT_DIR/scripts/trillium/net}
NET_OUT=${NET_OUT:-$SCRATCH/trillium/nccl/$SLURM_JOB_ID}
NET_PY=${NET_PY:-$SCRATCH/tt38/output/venv/torchtitan_native/bin/python}
export NET_DIR NET_OUT NET_PY
mkdir -p "$NET_OUT/logs/ib" "$NET_OUT/env"
unset PYTHONPATH PIP_CONFIG_FILE
set +u; module load ${NET_MODULES-cuda/12.9} || echo "module load failed; continuing"; set -u
GPUS=$(nvidia-smi -L | wc -l)
CPUS=$(( SLURM_CPUS_ON_NODE / GPUS ))
N=$SLURM_NNODES
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
echo "job $SLURM_JOB_ID: $N nodes x $GPUS GPUs, out $NET_OUT"
scontrol show hostnames "$SLURM_JOB_NODELIST" > "$NET_OUT/logs/nodes.txt"

srun --nodes="$N" --ntasks-per-node=1 --time=2 bash -c '{
    echo "== nccl.conf"; ls -la /etc/nccl* 2>&1; cat /etc/nccl.conf 2>&1
    echo "== env"; env | grep -E "^(NCCL|UCX|OMPI|PMIX|CUDA|LD_LIBRARY_PATH|LD_PRELOAD|TORCH)" | sort
    echo "== modules"; echo "${LOADEDMODULES:-none}"
    echo "== torch, NCCL"; "$NET_PY" -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.nccl.version())"
    "$NET_PY" -c "import glob, os, nvidia; [print(p) for d in nvidia.__path__ for p in sorted(glob.glob(os.path.join(d, \"nccl\", \"lib\", \"*\")))]"
    echo "== NCCL plugins on the system"; ls -la /usr/lib64/libnccl* /usr/lib/libnccl* /usr/local/lib/libnccl* 2>&1 | head
    echo "== HCA firmware, rate"; for d in /sys/class/infiniband/mlx5_*; do echo "${d##*/} $(cat $d/fw_ver) $(cat $d/ports/1/rate)"; done
    echo "== kernel"; uname -r; cat /proc/cmdline
    echo "== GPUs"; nvidia-smi --query-gpu=index,driver_version,power.limit,enforced.power.limit,clocks.sm,clocks.max.sm --format=csv,noheader
} > "$NET_OUT/env/$(hostname -s).txt" 2>&1'

snap() {  # snap <file>: IB counters of the four compute ports on every node
    srun --nodes="$N" --ntasks-per-node=1 --time=1 bash -c '
        for d in mlx5_0 mlx5_1 mlx5_4 mlx5_9; do
            for c in port_xmit_wait symbol_error link_downed port_rcv_errors link_error_recovery port_xmit_data port_rcv_data; do
                echo "$(hostname -s) $d $c $(cat /sys/class/infiniband/$d/ports/1/counters/$c 2>&1)"
            done
        done' > "$NET_OUT/logs/ib/$1" 2>&1
}

port=29700
step() {  # step <tag> <minutes> [VAR=value ...] -- <collectives.py args>
    local tag=$1 limit=$2 envs=()
    shift 2
    while [[ $1 != -- ]]; do envs+=("$1"); shift; done
    shift
    port=$((port + 1))
    mkdir -p "$NET_OUT/$tag/nccl"
    snap "$tag-before.txt"
    echo "[$(date +%T)] $tag: ${envs[*]:-defaults} collectives.py $* (limit $limit min)"
    env ${envs[@]+"${envs[@]}"} NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,ENV,TUNING NCCL_DEBUG_FILE="$NET_OUT/$tag/nccl/%h.%p.log" \
        MASTER_PORT=$port NET_OUT="$NET_OUT/$tag" \
        srun --nodes="$N" --ntasks-per-node="$GPUS" --cpus-per-task="$CPUS" --kill-on-bad-exit=1 --time="$limit" \
        "$NET_PY" "$NET_DIR/collectives.py" "$@" > "$NET_OUT/$tag/log.txt" 2>&1
    echo "[$(date +%T)] $tag: exit $?"
    snap "$tag-after.txt"
}

THREE=all_reduce,all_gather,reduce_scatter
step base 3 -- --nodes 13,26 --sweep-nodes 0
step ring 3 NCCL_ALGO=Ring -- --nodes 26 --sweep-nodes 0 --collectives $THREE
step tree 3 NCCL_ALGO=Tree -- --nodes 26 --sweep-nodes 0 --collectives all_reduce
step nvls0 3 NCCL_NVLS_ENABLE=0 -- --nodes 26 --sweep-nodes 0 --collectives $THREE
step collnet0 3 NCCL_COLLNET_ENABLE=0 -- --nodes 26 --sweep-nodes 0 --collectives $THREE
step base-end 3 -- --nodes 13,26 --sweep-nodes 0
echo "[$(date +%T)] done"
