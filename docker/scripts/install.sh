export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export FORCE_CUDA=1
export GIT_ARGS="--depth 1 --recursive"

export TORCH_CUDA_ARCH_LIST="10.0;12.0;12.1"

export VENV_BIN=/milabench/env/bin/
export VENV_BIN=/home/delaunao/workspace/benchdevenv/env/bin


# -- XFORMERS

cd /tmp
git clone $GIT_ARGS https://github.com/facebookresearch/xformers.git
$VENV_BIN/pip install wheel setuptools cmake ninja

cd /tmp/xformers
git submodule update --init
$VENV_BIN/pip install --no-build-isolation --no-deps -v --force-reinstall .
$VENV_BIN/python -m xformers.info

rm -rf /tmp/xformers

# -- Pytorch GEOMETRIC
cd /tmp
git clone $GIT_ARGS https://github.com/pyg-team/pytorch_geometric.git
cd /tmp/pytorch_geometric

$VENV_BIN/pip install --no-build-isolation --no-deps -v --force-reinstall .

rm -rf /tmp/pytorch_geometric


# -- Pytorch SCATTER
cd /tmp
git clone $GIT_ARGS https://github.com/rusty1s/pytorch_scatter.git
cd /tmp/pytorch_scatter

$VENV_BIN/pip install --no-build-isolation --no-deps -v --force-reinstall .

rm -rf /tmp/pytorch_scatter

# -- Pytorch SPARSE
cd /tmp
git clone $GIT_ARGS https://github.com/rusty1s/pytorch_sparse.git
cd /tmp/pytorch_sparse

$VENV_BIN/pip install --no-build-isolation --no-deps -v --force-reinstall .

rm -rf /tmp/pytorch_sparse

# -- Pytorch CLUSTER
cd /tmp
git clone $GIT_ARGS https://github.com/rusty1s/pytorch_cluster.git
cd /tmp/pytorch_cluster

$VENV_BIN/pip install --no-build-isolation --no-deps -v --force-reinstall .

rm -rf /tmp/pytorch_cluster


# -- Torch codec
cd /tmp
git clone $GIT_ARGS https://github.com/meta-pytorch/torchcodec.git
cd /tmp/torchcodec

apt-get install cmake pkg-config ffmpeg pybind11-dev 
apt-get install libavdevice-dev libavfilter-dev libavformat-dev libavcodec-dev libavutil-dev libswresample-dev libswscale-dev

export TORCHCODEC_CMAKE_BUILD_DIR="${PWD}/build"
I_CONFIRM_THIS_IS_NOT_A_LICENSE_VIOLATION=1 ENABLE_CUDA=1 $VENV_BIN/pip install --no-build-isolation -v --force-reinstall .

# I_CONFIRM_THIS_IS_NOT_A_LICENSE_VIOLATION=1 ENABLE_CUDA=1 pip install --no-build-isolation -v --force-reinstall .


rm -rf /tmp/torchcodec

# -- Torch VLLM
cd /tmp
git clone $GIT_ARGS https://github.com/vllm-project/vllm.git
cd /tmp/vllm

apt-get install ccache cmake

uv pip install -r requirements/build.txt --torch-backend=cu130
python tools/generate_cmake_presets.py
uv pip install --torch-backend=cu130 --no-build-isolation -v --force-reinstall .

rm -rf /tmp/vllm