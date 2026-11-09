# Trillium's replacement for Florida's config_hpg.sh (hpg.dgxb200x8), sourced by flux.sub after the node-count config.
# Florida also set NCCL_IB_HCA (their 8 HCAs) and NCCL_SOCKET_IFNAME=bridge-1145, which do not exist here: Trillium's job
# environment already sets NCCL_IB_HCA to its 4 x 400 Gb/s ports (mlx5_0,1,4,9), and NCCL picks the socket interface.
export MLPERF_SUBMITTER="University of Florida code on Trillium"
export MLPERF_SYSTEM_NAME="Trillium B200 (8x B200-SXM-180GB)"
export NCCL_TEST=0
export OMPI_MCA_pml=ucx
export OMPI_MCA_btl=^openib
# Trillium binds each task of a step to its CPUs (task/affinity, task/cgroup) and gives a step 1 CPU per task without
# -c: each rank (main process and its 8 data workers) gets the node's share instead, 96 / 8 = 12 cores
# CPU_SHARE=0 keeps Slurm's default instead (Florida's srun as it is; a step-gap test, gap.sbatch)
[ "${CPU_SHARE:-1}" = 0 ] || export SRUN_CPUS_PER_TASK=$((SLURM_CPUS_ON_NODE / DGXNGPU))
# $HOME is not writable from the jobs: Triton could not create ~/.triton, torch.compile fell back to eager and MBS 32
# ran out of memory (job 1271975). The node's /tmp instead (bound into the container)
export TRITON_HOME=/tmp/$USER-flux
