Quickstart
==========

`Docker Images <https://github.com/mila-iqia/milabench/pkgs/container/milabench>`_ are created for each release.
They come with all benchmarks installed and the necessary datasets. No additional downloads are necessary.

.. tip::

   We recommend using `podman <https://podman.io/>`_ over docker.
   Podman is rootless by default, compatible with docker CLI commands, and avoids
   the docker daemon. All ``docker`` commands on this page work with ``podman``
   as a drop-in replacement.


0. Requirements
---------------

.. tab-set::

   .. tab-item:: CUDA

      * NVIDIA driver
      * `docker-ce <https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository>`_
      * `nvidia-docker <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker>`_

   .. tab-item:: ROCm

      * ROCm driver
      * docker

   .. tab-item:: From Source

      * Python 3.12+
      * `uv <https://docs.astral.sh/uv/>`_ (recommended) or pip
      * CUDA or ROCm toolkit


1. Install / Pull image
-----------------------

.. tab-set::

   .. tab-item:: CUDA

      .. code-block:: bash

         export MILABENCH_BASE=$PWD/results
         export HF_TOKEN=<your_huggingface_token>
         export MILABENCH_IMAGE=ghcr.io/mila-iqia/milabench:cuda-nightly
         docker pull $MILABENCH_IMAGE

   .. tab-item:: ROCm

      .. code-block:: bash

         export MILABENCH_BASE=$PWD/results
         export HF_TOKEN=<your_huggingface_token>
         export MILABENCH_IMAGE=ghcr.io/mila-iqia/milabench:rocm-nightly
         docker pull $MILABENCH_IMAGE

   .. tab-item:: From Source

      .. code-block:: bash

         git clone git@github.com:mila-iqia/milabench.git
         cd milabench

         uv venv --python=3.12
         source ./venv/bin/activate

         export MILABENCH_BASE=$PWD/results
         export MILABENCH_CONFIG=config/standard.yaml
         export MILABENCH_GPU_ARCH=cuda   # or rocm
         export HF_TOKEN=<your_huggingface_token>

         # CUDA
         pip install -e .[cuda]

         # or ROCm
         pip install -e .[rocm]


2. Create a system file
-----------------------

Create a ``results/system.yaml`` file describing your cluster.

.. tip::

   Optional for single node runs.


.. tab-set::

   .. tab-item:: Single node

      .. code-block:: yaml

         system:
           arch: cuda    # or rocm
           docker_image: ghcr.io/mila-iqia/milabench:${system.arch}-nightly

           nodes:
             - name: node1
               ip: 127.0.0.1
               main: true
               user: <username>

   .. tab-item:: Multi-node (CUDA)

      .. code-block:: yaml

         system:
           arch: cuda
           docker:
             executable: docker
             image: ghcr.io/mila-iqia/milabench:cuda-nightly
             base: /path/to/milabench_base
             args: [
               --rm, --ipc=host, --network=host,
               --device, nvidia.com/gpu=all,
               --security-opt=label=disable,
               -e, HF_TOKEN=$HF_TOKEN,
               -v, "$SSH_KEY_FILE:/root/.ssh/id_rsa:Z",
               -v, "$MILABENCH_BASE/data:/milabench/envs/data",
               -v, "$MILABENCH_BASE/cache:/milabench/envs/cache",
               -v, "$MILABENCH_BASE/runs:/milabench/envs/runs",
             ]

           nodes:
             - name: main
               ip: 192.168.0.25
               main: true
               user: <username>

             - name: worker
               ip: 192.168.0.26
               main: false
               user: <username>

      The ``docker`` section tells milabench how to spawn containers on worker nodes.
      Make sure all nodes can SSH to each other without passwords.

   .. tab-item:: Multi-node (ROCm)

      .. code-block:: yaml

         system:
           arch: rocm
           docker:
             executable: docker
             image: ghcr.io/mila-iqia/milabench:rocm-nightly
             base: /path/to/milabench_base
             args: [
               --rm, --ipc=host, --network=host,
               --device, /dev/kfd, --device, /dev/dri,
               --security-opt=label=disable,
               --security-opt, seccomp=unconfined,
               --group-add, video,
               -e, HF_TOKEN=$HF_TOKEN,
               -v, "$SSH_KEY_FILE:/root/.ssh/id_rsa:Z",
               -v, "$MILABENCH_BASE/data:/milabench/envs/data",
               -v, "$MILABENCH_BASE/cache:/milabench/envs/cache",
               -v, "$MILABENCH_BASE/runs:/milabench/envs/runs",
             ]

           nodes:
             - name: main
               ip: 192.168.0.25
               main: true
               user: <username>

             - name: worker
               ip: 192.168.0.26
               main: false
               user: <username>

      Make sure all nodes can SSH to each other without passwords.
      Add more entries to ``nodes`` as needed.

   .. tab-item:: Multi-node (From Source)

      .. code-block:: yaml

         system:
           arch: cuda    # or rocm
           sshkey: ~/.ssh/id_ed25519

           nodes:
             - name: main
               ip: 192.168.0.25
               main: true
               user: <username>

             - name: worker
               ip: 192.168.0.26
               main: false
               user: <username>

      Milabench must be installed on every node.
      Make sure all nodes can SSH to each other without passwords.


3. Prepare
----------

Download datasets and model weights.

.. tab-set::

   .. tab-item:: CUDA

      .. code-block:: bash

         export SSH_KEY_FILE=$HOME/.ssh/id_rsa

         docker run -it --rm --gpus all --network host --ipc=host --privileged \
           -v $SSH_KEY_FILE:/milabench/id_milabench \
           -v $(pwd)/results:/milabench/envs/runs \
           $MILABENCH_IMAGE \
           milabench prepare --system /milabench/envs/runs/system.yaml

   .. tab-item:: ROCm

      .. code-block:: bash

         export SSH_KEY_FILE=$HOME/.ssh/id_rsa

         docker run -it --rm --network host --ipc=host --privileged \
           --device=/dev/kfd --device=/dev/dri \
           --security-opt seccomp=unconfined --group-add video \
           -v /opt/amdgpu/share/libdrm/amdgpu.ids:/opt/amdgpu/share/libdrm/amdgpu.ids \
           -v /opt/rocm:/opt/rocm \
           -v $SSH_KEY_FILE:/milabench/id_milabench \
           -v $(pwd)/results:/milabench/envs/runs \
           $MILABENCH_IMAGE \
           milabench prepare --system /milabench/envs/runs/system.yaml

   .. tab-item:: From Source

      .. code-block:: bash

         milabench install --config $MILABENCH_CONFIG --base $MILABENCH_BASE
         milabench prepare --config $MILABENCH_CONFIG --base $MILABENCH_BASE \
           --system results/system.yaml

For multi-node setups, run prepare on each node so datasets are available locally.


4. Run
------

.. tab-set::

   .. tab-item:: CUDA

      .. code-block:: bash

         docker run -it --rm --gpus all --network host --ipc=host --privileged \
           -v $SSH_KEY_FILE:/milabench/id_milabench \
           -v $(pwd)/results:/milabench/envs/runs \
           $MILABENCH_IMAGE \
           milabench run --system /milabench/envs/runs/system.yaml

   .. tab-item:: ROCm

      .. code-block:: bash

         docker run -it --rm --network host --ipc=host --privileged \
           --device=/dev/kfd --device=/dev/dri \
           --security-opt seccomp=unconfined --group-add video \
           -v /opt/amdgpu/share/libdrm/amdgpu.ids:/opt/amdgpu/share/libdrm/amdgpu.ids \
           -v /opt/rocm:/opt/rocm \
           -v $SSH_KEY_FILE:/milabench/id_milabench \
           -v $(pwd)/results:/milabench/envs/runs \
           $MILABENCH_IMAGE \
           milabench run --system /milabench/envs/runs/system.yaml

   .. tab-item:: From Source

      .. code-block:: bash

         milabench run --config $MILABENCH_CONFIG --base $MILABENCH_BASE \
           --system results/system.yaml

``--ipc=host`` removes shared memory restrictions.
You can use ``--shm-size 8G`` or higher instead if you prefer.

To run only specific benchmarks, add ``--select <pattern>``
(e.g. ``--select multinode`` for multi-node benchmarks only).


.. note::

   The multi-node benchmarks are sensitive to network performance.
   If the single-node variant is significantly faster, Infiniband may not be present or not in use.
   The ``--privileged`` flag is often required for the container to access Infiniband devices.


5. Report
---------

.. tab-set::

   .. tab-item:: Docker

      .. code-block:: bash

         docker run -it --rm \
           -v $(pwd)/results:/milabench/envs/runs \
           $MILABENCH_IMAGE \
           milabench report --runs /milabench/envs/runs

   .. tab-item:: From Source

      .. code-block:: bash

         milabench report --runs $MILABENCH_BASE/runs \
           --config $MILABENCH_CONFIG


Building images
---------------

Images can be built locally for prototyping and testing.

.. tab-set::

   .. tab-item:: CUDA

      .. code-block:: bash

         docker build -f docker/Dockerfile-cuda \
           -t milabench:cuda-nightly \
           --build-arg CONFIG=standard.yaml .

   .. tab-item:: ROCm

      .. code-block:: bash

         docker build -f docker/Dockerfile-rocm \
           -t milabench:rocm-nightly \
           --build-arg CONFIG=standard.yaml .
