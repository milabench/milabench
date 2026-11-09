#!/usr/bin/env bash
# K3-XN part 2, one node, changes nothing: what DeepEP's cross-node kernels need from the node. The NVIDIA driver's
# parameters (NVSHMEM's IBGDA needs EnableStreamMemOPs=1 and PeerMappingOverride=1 in RegistryDwords), the GPUDirect
# RDMA modules (nvidia_peermem, gdrdrv), the HCAs, memlock, the NCCL of vLLM's venv (DeepEP v2 runs on NCCL's GIN), then
# one 2-GPU NCCL all-reduce with NVLink and shared memory off: NCCL's log says whether it reaches the HCA with GDR.
#
#   /opt/slurm/bin/.sbatch --export=NONE --get-user-env --nodelist=<node> scripts/trillium/kimi/driver_check.sh
#SBATCH --job-name=nw-k3-xn-driver-check
#SBATCH --account=rrg-bengioy-ad --partition=compute_b200_full_node --reservation=b200-friendly-users --qos=bump
#SBATCH --nodes=1 --gpus-per-node=b200:8 --exclusive --mem=0 --time=0:05:00
#SBATCH --exclude=trig0114,trig0125,trig0132,trig0138,trig0146,trig0150
#SBATCH --output=/scratch/%u/trillium/new/k3-xn/slurm/%x.%j.out
ulimit -c 0
V=$SCRATCH/trillium/new/k3-inf/venv/torch
section() { echo; echo "== $*"; }

section "$(hostname -s) $(date -Is) kernel $(uname -r)"
nvidia-smi --query-gpu=index,name,driver_version,pci.bus_id --format=csv,noheader
section /proc/driver/nvidia/version; cat /proc/driver/nvidia/version
section /proc/driver/nvidia/params; cat /proc/driver/nvidia/params
section "modprobe.d options for nvidia"; grep -rH . /etc/modprobe.d/ 2>/dev/null | grep -i nvidia
section "modules: nvidia, peer memory, gdrcopy, IB"; lsmod | grep -E "^(nvidia|nv_peer|gdrdrv|ib_core|ib_uverbs|mlx5)"
section /dev/gdrdrv; ls -l /dev/gdrdrv 2>&1
section "memlock (ulimit -l)"; ulimit -l
section "OFED / HCAs"; ofed_info -s 2>&1; ibv_devinfo 2>&1 | grep -E "hca_id|fw_ver|state|active_mtu|link_layer"
section "GPU / NIC topology"; nvidia-smi topo -m
section "NCCL in $V"
ls "$V"/lib/python3.12/site-packages/ | grep -i -E "^nvidia_nccl|^nvidia_nvshmem|^deep_?ep|^nixl"
"$V/bin/python" -c 'import torch; v=torch.cuda.nccl.version(); print("torch", torch.__version__, "built with NCCL", v)'

section "NCCL 2-GPU all-reduce over the HCA (P2P and SHM off)"
T=$(mktemp -d)
cat > "$T/ar.py" <<'EOF'
import torch, torch.distributed as dist
dist.init_process_group("nccl")
r = dist.get_rank()
torch.cuda.set_device(r)
x = torch.ones(1 << 26, device="cuda")
dist.all_reduce(x)
torch.cuda.synchronize()
print(f"rank {r}: all_reduce of 256 MB, value {x[0].item()}")
dist.destroy_process_group()
EOF
NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET \
    timeout 120 "$V/bin/torchrun" --standalone --nproc-per-node 2 "$T/ar.py" 2>&1 \
    | grep -E "NCCL version|NET/|GDR|DMA-BUF|dmabuf|via NET|Using network|all_reduce|Error|error" | head -40
rm -rf "$T"
