#!/usr/bin/env bash
# Job script of submit.sh: slime entries (benchmarks/slime/dev.yaml) on Trillium's B200 nodes, `job.sbatch ENTRY...`.
#   1. power check on every node
#   2. `milabench install`, offline: the first job builds TransformerEngine (wheelhouse) and torch_memory_saver (git
#      source in the uv cache) against the venv's torch, later jobs return at once (venv/torch/.slime_installed)
#   3. `milabench prepare` on the first node: the HF -> torch_dist conversion (--ref-load), once
#   4. one `milabench run` per entry, in the order given; run data in $B/runs/<job name>.<job id>.<entry>
# Everything lives in B = $SCRATCH/trillium/new/glm-rl (install.sh); $HOME is read-only from the compute nodes.
set -uo pipefail
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
nodes() { srun --overlap --nodes="$SLURM_JOB_NUM_NODES" --ntasks="$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 "$@"; }
# `NVRM: knvlinkSendInbandData_IMPL` lines in each node's kernel log at the start and the end, for the hardware checks
# (the epilog's kernel messages are the first node's only)
nvrm() { nodes bash -c "echo \"$P nvrm-inband $1 \$(hostname -s) \$(dmesg | grep -c knvlinkSendInbandData)\""; }
nodes mkdir -p "$L/tmp"
nvrm start
trap 'nvrm end; nodes rm -rf "$L"' EXIT
# (sglang puts the inductor and CUDA caches in SGLANG_CACHE_DIR, job 1261398, and its JIT builds in SGLANG_JIT_CACHE_DIR,
# job 1261434: both default to ~/.cache/sglang)
export TMPDIR=$L/tmp TRITON_CACHE_DIR=$L/triton TILELANG_CACHE_DIR=$L/tilelang FLASHINFER_WORKSPACE_BASE=$L \
    SGLANG_CACHE_DIR=$L/sglang SGLANG_JIT_CACHE_DIR=$L/sglang/jit
echo "$P job=$SLURM_JOB_ID nodes=$SLURM_JOB_NODELIST milabench $(git -C "$B/src" rev-parse --short HEAD) entries: $*"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | sort | uniq -c | sed "s/^/$P /"
nodes bash "$B/src/scripts/trillium/power_check.sh" || exit 4

select=$(IFS=,; echo "$*")
milabench install --config "$C" --select "$select" || { echo "$P install failed" >&2; exit 5; }
echo "$P install: $(cat "$B/venv/torch/.slime_installed")"
system=$B/slurm/system.$SLURM_JOB_ID.yaml
milabench slurm system > "$system" || exit 1
srun --nodes=1 --ntasks=1 --nodelist="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)" \
    milabench prepare --config "$C" --select "$select" || { echo "$P prepare failed" >&2; exit 6; }

status_all=0
for entry in "$@"; do
    run=$SLURM_JOB_NAME.$SLURM_JOB_ID.$entry
    echo "$P === $entry (run $run)"
    start=$(date +%s)
    milabench run --config "$C" --system "$system" --select "$entry" --run-name "$run"
    status=$?
    [ $status = 0 ] || status_all=$status
    echo "$P $entry: $([ $status = 0 ] && echo PASS || echo "FAIL ($status)") $(( $(date +%s) - start )) s"
done
exit $status_all
