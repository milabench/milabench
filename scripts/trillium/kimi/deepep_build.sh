#!/usr/bin/env bash
# K3-XN, one node: the DeepEP venv, $X/venv/deepep, beside K3-INF's. A copy of K3-INF's vLLM venv (K3-INF's files are only
# read), NCCL 2.30.7 and nixl 1.5.2 from wheels downloaded on the login node, then DeepEP a56d615 built for sm_100: the last
# commit with the ElasticBuffer (V2, NCCL Gin) API that vLLM 0.32's deepep_v2 backend calls; V2.5 renamed it. DeepEP
# compiles its kernels at run time (JIT, needs CUDA_HOME with nvcc and cuobjdump): CUDA 13.2 from cvmfs, not its module,
# which also loads NCCL 2.29.7. Ends with checks: vLLM's deepep_v2 availability, nixl, the run-time NCCL, and DeepEP's EP
# test on the node's 8 GPUs at K3's shape (hidden 3,584, top-16 of 896 experts; NVLink only on one node).
#
#   login node, once ($X=$SCRATCH/trillium/new/k3-xn):
#     git clone https://github.com/deepseek-ai/DeepEP.git $X/deepep-src && git -C $X/deepep-src checkout a56d615
#     git -C $X/deepep-src submodule update --init third-party/fmt
#     $X/src/.venv/bin/python -m pip download --no-deps -d $X/wheels nvidia-nccl-cu13==2.30.7 nixl==1.5.2 nixl-cu13==1.5.2
#   /opt/slurm/bin/.sbatch --export=NONE --get-user-env --nodelist=<node> scripts/trillium/kimi/deepep_build.sh
#SBATCH --job-name=nw-k3-xn-deepep-build
#SBATCH --account=rrg-bengioy-ad --partition=compute_b200_full_node --reservation=b200-friendly-users --qos=bump
#SBATCH --nodes=1 --gpus-per-node=b200:8 --exclusive --mem=0 --time=0:50:00
#SBATCH --exclude=trig0125,trig0132,trig0150
#SBATCH --output=/scratch/%u/trillium/new/k3-xn/slurm/%x.%j.out
set -euo pipefail
ulimit -c 0
X=$SCRATCH/trillium/new/k3-xn K=$SCRATCH/trillium/new/k3-inf
V=$X/venv/deepep
export CUDA_HOME=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Core/cudacore/13.2.0
export PATH=$CUDA_HOME/bin:$PATH TMPDIR=/dev/shm/$USER/deepep-build EP_JIT_CACHE_DIR=/dev/shm/$USER/deepep-jit
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR" "$EP_JIT_CACHE_DIR"' EXIT
unset PYTHONPATH PIP_CONFIG_FILE
echo "[deepep] $(hostname -s) $(date -Is)"

if [ ! -d "$V" ]; then
    echo "[deepep] copying $K/venv/torch"; t=$(date +%s)
    rm -rf "$V.tmp"; cp -a "$K/venv/torch" "$V.tmp"
    # the copy's scripts and editable installs still name K3-INF's venv and checkout
    grep -rlI "$K/venv/torch" "$V.tmp/bin" "$V.tmp/pyvenv.cfg" | xargs -r sed -i "s|$K/venv/torch|$V|g"
    sed -i "s|$K/src|$X/src|g" "$V.tmp"/lib/python3.12/site-packages/*.pth
    mv "$V.tmp" "$V"; echo "[deepep] copied in $(( $(date +%s) - t )) s"
fi
UV_LINK_MODE=copy UV_CACHE_DIR=$X/uv-cache VIRTUAL_ENV=$V "$HOME/.local/bin/uv" pip install --offline --no-deps \
    "$X"/wheels/nvidia_nccl_cu13-2.30.7-*.whl "$X"/wheels/nixl_cu13-1.5.2-*.whl "$X"/wheels/nixl-1.5.2-*.whl

echo "[deepep] building $(git -C "$X/deepep-src" log -1 --oneline)"; t=$(date +%s)
cd "$X/deepep-src"
log=$X/slurm/deepep-build.$SLURM_JOB_ID.log
# CCACHE_DISABLE: the user environment compiles through ccache, whose cache is in the read-only $HOME (job 1270384)
CCACHE_DISABLE=1 CPATH=$CUDA_HOME/include/cccl TORCH_CUDA_ARCH_LIST=10.0 MAX_JOBS=32 \
    "$V/bin/python" -m pip install -v --no-build-isolation --no-deps --no-cache-dir . > "$log" 2>&1 \
    || { echo "[deepep] build failed, $log:"; grep -n -E "error|FAILED" "$log" | head -40; exit 1; }
echo "[deepep] built in $(( $(date +%s) - t )) s"

cd "$TMPDIR"
"$V/bin/python" - <<'EOF'
import torch, deep_ep
from vllm.utils.import_utils import deep_ep_v2_unavailable_reason, _get_runtime_nccl_version, has_nixl_ep
print("[deepep] torch", torch.__version__, "deep_ep", deep_ep.__file__, "ElasticBuffer", hasattr(deep_ep, "ElasticBuffer"))
print("[deepep] runtime NCCL", _get_runtime_nccl_version(), "torch NCCL", torch.cuda.nccl.version())
print("[deepep] vLLM deepep_v2 unavailable reason:", deep_ep_v2_unavailable_reason())
from nixl._api import nixl_agent
print("[deepep] nixl agent class", nixl_agent)
EOF
echo "[deepep] ibstat: $(command -v ibstat || echo missing)"
timeout 600 "$V/bin/python" "$X/deepep-src/tests/elastic/test_ep.py" --num-processes 8 \
    --num-tokens 128 --hidden 3584 --num-topk 16 --num-experts 896 --test-first-only 2>&1 | tail -40
echo "[deepep] done $(date -Is)"
