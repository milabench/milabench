#!/bin/bash
# D2-D4 network sweeps on every node of the job (README.md here for the commands).
#
#   sbatch --nodes=52 [--time=..] scripts/trillium/net/net.sbatch [d2] [d3] [d4] [-- args for every script]
#
# Submit from the repository root (the scripts are found under $SLURM_SUBMIT_DIR, or $NET_DIR).
# Defaults are for Trillium's B200 partition; on Rorqual override account, partition and GPUs:
#   --account=rrg-bengioy-ad_gpu --partition=gpubase_bynode_b1 --gpus-per-node=h100:4 --mem=0 --reservation= --qos=normal
# Results: $NET_OUT (default ${NET_ROOT:-$SCRATCH/trillium}/nccl/<job>/): pairs.jsonl, collectives.jsonl, logs/.
# Per-experiment arguments: NET_D2_ARGS, NET_D3_ARGS, NET_D4_ARGS (e.g. NET_D2_ARGS=--sequential).
#
#SBATCH --job-name=tri-net
#SBATCH --account=rrg-bengioy-ad
#SBATCH --partition=compute_b200_full_node
#SBATCH --reservation=b200-friendly-users
#SBATCH --qos=bump
#SBATCH --exclude=trig[0122-0123,0133-0135,0138,0140,0143,0145,0148,0152]
#SBATCH --gpus-per-node=b200:8
#SBATCH --exclusive
#SBATCH --time=1:00:00

set -euo pipefail

EXPERIMENTS=()
while [[ $# -gt 0 && "$1" != "--" ]]; do EXPERIMENTS+=("$1"); shift; done
[[ "${1:-}" == "--" ]] && shift
[[ ${#EXPERIMENTS[@]} -gt 0 ]] || EXPERIMENTS=(d2 d3 d4)

NET_DIR=${NET_DIR:-$SLURM_SUBMIT_DIR/scripts/trillium/net}
NET_OUT=${NET_OUT:-${NET_ROOT:-$SCRATCH/trillium}/nccl/$SLURM_JOB_ID}
NET_PY=${NET_PY:-$SCRATCH/tt38/output/venv/torchtitan_native/bin/python}
export NET_OUT
mkdir -p "$NET_OUT/logs/topo"

# DRAC login profile points Python at cvmfs site-packages; torch brings its own CUDA and NCCL.
unset PYTHONPATH PIP_CONFIG_FILE
set +u; module load ${NET_MODULES-cuda/12.9} || echo "module load failed; continuing"; set -u

GPUS=$(nvidia-smi -L | wc -l)
CPUS=$(( ${SLURM_CPUS_ON_NODE:-$(nproc)} / GPUS ))
NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_ADDR=$(head -1 <<< "$NODES")
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
echo "job $SLURM_JOB_ID: $SLURM_NNODES nodes x $GPUS GPUs, $CPUS CPUs per task, out $NET_OUT, ${EXPERIMENTS[*]}"
echo "$NODES" > "$NET_OUT/logs/nodes.txt"

# GPU <-> HCA affinity and HCA link rates, once per node.
srun --ntasks-per-node=1 bash -c '{ nvidia-smi topo -m; echo; for p in /sys/class/infiniband/*/ports/*; do
    echo "$p $(cat $p/rate) $(cat $p/state) $(cat $p/link_layer)"; done; } > "$NET_OUT/logs/topo/$(hostname -s).txt" 2>&1' || true

port=29500
run() {  # run <name> <script> [args]: one process per GPU, log beside the JSONL
    local name=$1 script=$2; shift 2
    port=$((port + 1))
    echo "[$(date +%T)] $name: $script $*"
    MASTER_PORT=$port srun --ntasks-per-node="$GPUS" --cpus-per-task="$CPUS" --kill-on-bad-exit=1 \
        "$NET_PY" "$NET_DIR/$script" "$@" > "$NET_OUT/logs/$name.log" 2>&1 \
        && echo "[$(date +%T)] $name done" || echo "[$(date +%T)] $name FAILED (exit $?), see logs/$name.log"
}

for e in "${EXPERIMENTS[@]}"; do
    case $e in
        d2) run d2 pairs.py ${NET_D2_ARGS:-} "$@" ;;
        d3) run d3 collectives.py ${NET_D3_ARGS:-} "$@" ;;
        d4) run d4 bisection.py ${NET_D4_ARGS:-} "$@" ;;
        *) echo "unknown experiment $e (d2 | d3 | d4)"; exit 2 ;;
    esac
done
