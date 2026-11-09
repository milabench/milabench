#!/usr/bin/env bash
# Job script of submit.sh: slime entries (benchmarks/slime/dev.yaml) on Trillium's B200 nodes,
# `job.sbatch [NAME=value...] ENTRY...` (the NAME=value are exported first; SLIME_CONVERT=1: README, step 3).
#   1. power check on every node
#   2. `milabench install`, offline: the first job builds TransformerEngine (wheelhouse) and torch_memory_saver (git
#      source in the uv cache) against the venv's torch, later jobs return at once (venv/torch/.slime_installed)
#   3. `milabench prepare` on the first node: the HF -> torch_dist conversion (--ref-load), once (an entry's
#      model.convert_nodes > 1: over that many nodes); a SLIME_CONVERT=1 job stops there
#   4. one `milabench run` per entry, in the order given; run data in $B/runs/<job name>.<job id>.<entry>
# Everything lives in B = $SCRATCH/trillium/new/glm-rl (install.sh); $HOME is read-only from the compute nodes.
set -uo pipefail
# no core files: the compute nodes allow them and Slurm does not pass on the login node's limit (a K3 crash left
# 681 GB of them, 2026-11-07); srun steps get it only with --propagate=CORE (PropagateResourceLimits=NONE): this
# script's `nodes`, milabench's srun (milabench/cli/slurm/srun.py: Ray's bring-up, `rayrun`) and the conversion's
# (benchmarks/slime/prepare.py) pass it; each node's limit is in the log at the start
ulimit -c 0
while [[ ${1:-} == [A-Za-z_]*=* ]]; do export "$1"; shift; done
B=${SCRATCH:?}/trillium/new/glm-rl
C=$B/src/benchmarks/slime/dev.yaml
P=[$SLURM_JOB_NAME]
unset PYTHONPATH PIP_CONFIG_FILE
export MILABENCH_BASE=$B MILABENCH_GPU_ARCH=cuda PYTHONUNBUFFERED=1
export PATH=$B/src/.venv/bin:$HOME/.local/bin:$PATH
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline
export UV_CACHE_DIR=$B/uv-cache UV_LINK_MODE=copy UV_OFFLINE=1 UV_NO_INDEX=1 UV_FIND_LINKS=$B/wheelhouse
# no internet: build TE from its sdist, for B200 only
export NVTE_PYTORCH_FORCE_BUILD=TRUE NVTE_CUDA_ARCHS=100
# torch_memory_saver links -lcuda, which cvmfs's ld does not find in /usr/lib64 (job 1261365); /usr/lib64/nvidia holds
# the driver's libraries only
export LIBRARY_PATH=/usr/lib64/nvidia${LIBRARY_PATH:+:$LIBRARY_PATH}
export XDG_CACHE_HOME=$B/cache
# node-local files in RAM (/tmp holds 2 GB): TMPDIR (Ray's session directory among them) and the kernel caches
L=/dev/shm/$USER/glm-rl
nodes() {
    srun --overlap --propagate=CORE --nodes="$SLURM_JOB_NUM_NODES" --ntasks="$SLURM_JOB_NUM_NODES" \
        --ntasks-per-node=1 "$@"
}
# `NVRM: knvlinkSendInbandData_IMPL` lines in each node's kernel log at the start and the end, for the hardware checks
# (the epilog's kernel messages are the first node's only)
nvrm() { nodes bash -c "echo \"$P nvrm-inband $1 \$(hostname -s) \$(dmesg | grep -c knvlinkSendInbandData)\""; }
nodes mkdir -p "$L/tmp"
# FlashInfer's JIT modules (the *-trtllm entries' MoE, GEMM and FMHA launchers: 17 min of nvcc per node, job 1274132),
# built once and copied to every node, where flashinfer's ninja finds them up to date
[ -d "$B/cache/flashinfer-jit" ] && nodes bash -c "mkdir -p $L/.cache/flashinfer && cp -a $B/cache/flashinfer-jit/. $L/.cache/flashinfer/"
nodes bash -c "echo \"$P \$(hostname -s): ulimit -c \$(ulimit -c)\""
nvrm start
# At the end, and on the time limit (submit.sh: SIGUSR1 5 min before it): stop Ray on every node, then wait until no
# process holds a GPU (killed after 2 min), before Slurm cleans up: trig0146-0147 drained with
# health.chk.nvidia-smi.unresponsive 1 s after job 1262234 ended, 3 s after its `ray stop`. Job 1267151 (disaggregated):
# 9 trainer nodes drained while milabench's own `ray stop` (48 s) was still stopping their processes, before this ran;
# Slurm's health check runs every 300 s in any node state (README, GLM-RL-05)
release_gpus() {
    nodes bash -c '"$1" stop --force > /dev/null 2>&1
        for i in $(seq 90); do
            pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d " ")
            [ -z "$pids" ] && { echo "$2 $(hostname -s): GPUs free after $((2 * i - 2)) s"; exit 0; }
            [ "$i" = 60 ] && kill -KILL $pids
            sleep 2
        done
        echo "$2 $(hostname -s): GPUs still held by" $pids' _ "$B/venv/torch/bin/ray" "$P"
}
trap 'release_gpus; nvrm end; nodes rm -rf "$L"' EXIT
trap 'echo "$P time limit in 5 min: stopping"; kill -TERM "${child:-}" 2>/dev/null; exit 124' USR1
# prepare and run in the background, so that the USR1 trap runs while they do
bg_wait() { "$@" & child=$!; wait $child; }
# (sglang puts the inductor and CUDA caches in SGLANG_CACHE_DIR, job 1261398, and its JIT builds in SGLANG_JIT_CACHE_DIR,
# job 1261434: both default to ~/.cache/sglang; DeepGEMM's JIT cache defaults to ~/.dj)
export TMPDIR=$L/tmp TRITON_CACHE_DIR=$L/triton TILELANG_CACHE_DIR=$L/tilelang FLASHINFER_WORKSPACE_BASE=$L \
    SGLANG_CACHE_DIR=$L/sglang SGLANG_JIT_CACHE_DIR=$L/sglang/jit DG_JIT_CACHE_DIR=$L/deep_gemm
echo "$P job=$SLURM_JOB_ID nodes=$SLURM_JOB_NODELIST milabench $(git -C "$B/src" rev-parse --short HEAD) entries: $*"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | sort | uniq -c | sed "s/^/$P /"
nodes bash "$B/src/scripts/trillium/power_check.sh" || exit 4

select=$(IFS=,; echo "$*")
milabench install --config "$C" --select "$select" || { echo "$P install failed" >&2; exit 5; }
echo "$P install: $(cat "$B/venv/torch/.slime_installed")"
system=$B/slurm/system.$SLURM_JOB_ID.yaml
milabench slurm system > "$system" || exit 1
# in the batch step (the first node), not a step of its own: a multi-node conversion starts its own srun from there
bg_wait milabench prepare --config "$C" --select "$select" || { echo "$P prepare failed" >&2; exit 6; }
[ -n "${SLIME_CONVERT:-}" ] && { echo "$P SLIME_CONVERT: conversion done, no run"; exit 0; }

status_all=0
for entry in "$@"; do
    run=$SLURM_JOB_NAME.$SLURM_JOB_ID.$entry
    echo "$P === $entry (run $run)"
    start=$(date +%s)
    bg_wait milabench run --config "$C" --system "$system" --select "$entry" --run-name "$run"
    status=$?
    [ $status = 0 ] || status_all=$status
    echo "$P $entry: $([ $status = 0 ] && echo PASS || echo "FAIL ($status)") $(( $(date +%s) - start )) s"
done
exit $status_all
