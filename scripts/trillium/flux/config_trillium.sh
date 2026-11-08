# Trillium's replacement for Florida's config_hpg.sh (hpg.dgxb200x8), sourced by flux.sub after the node-count config.
# Florida also set NCCL_IB_HCA (their 8 HCAs) and NCCL_SOCKET_IFNAME=bridge-1145, which do not exist here: Trillium's job
# environment already sets NCCL_IB_HCA to its 4 x 400 Gb/s ports (mlx5_0,1,4,9), and NCCL picks the socket interface.
export MLPERF_SUBMITTER="University of Florida code on Trillium"
export MLPERF_SYSTEM_NAME="Trillium B200 (8x B200-SXM-180GB)"
export NCCL_TEST=0
export OMPI_MCA_pml=ucx
export OMPI_MCA_btl=^openib
