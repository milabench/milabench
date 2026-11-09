#!/usr/bin/env bash
# K3-XN part 3: Kimi-K3 with prefill and decode on separate nodes, the server.command of the -pd entries. milabench runs it
# on the job's main node with the entry's vLLM flags; it starts, on the job's first 4 nodes, a prefill instance (nodes 0-1)
# and a decode instance (nodes 2-3), each EP16 (DP2 x TP8, one engine per node, as kimi-vllm-2n-dp2tp8-ep16), joined by
# vLLM's NIXL KV transfer (KV cache and KDA state), and on the main node vLLM v0.32.0's NIXL toy proxy (pd_proxy.py) on the
# port milabench's client uses. The first child to exit ends the script, which stops the others.
#
#   login node, once: curl -sSfL -o $SCRATCH/trillium/new/k3-xn/pd/toy_proxy_server.py \
#     https://raw.githubusercontent.com/vllm-project/vllm/v0.32.0/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py
# Needs the venv with nixl, venv/deepep (deepep_build.sh).
set -euo pipefail
: "${MILABENCH_BASE:?}" "${SLURM_JOB_NODELIST:?}"
V=$MILABENCH_BASE/venv/deepep
mapfile -t nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
[ "${#nodes[@]}" -ge 4 ] || { echo "[pd] needs 4 nodes, the job has ${nodes[*]}" >&2; exit 2; }
ip() { getent ahostsv4 "$1" | awk 'NR == 1 {print $1}'; }
# NIXL moves the KDA (Mamba-type) state only with the conv state in (dim, state_len) layout (vLLM's NixlConnector asserts)
export VLLM_SSM_CONV_STATE_LAYOUT=DS
kv='{"kv_connector": "NixlConnector", "kv_role": "kv_both"}'

# one EP16 instance: its API server on node $1 at port $3, its second engine headless on node $2
instance() {
    local head=$1 worker=$2 port=$3 role=$4 address
    address=$(ip "$head")
    srun --overlap --propagate=CORE -N1 -n1 -w "$head" env VLLM_NIXL_SIDE_CHANNEL_HOST="$address" \
        "$V/bin/vllm" serve "${@:5}" --port "$port" --kv-transfer-config "$kv" \
        --data-parallel-size 2 --data-parallel-size-local 1 --data-parallel-address "$address" --data-parallel-rpc-port 29550 \
        > >(sed -u "s/^/[$role 0] /") 2>&1 &
    srun --overlap --propagate=CORE -N1 -n1 -w "$worker" env VLLM_NIXL_SIDE_CHANNEL_HOST="$(ip "$worker")" \
        "$V/bin/vllm" serve "${@:5}" --kv-transfer-config "$kv" \
        --data-parallel-size 2 --data-parallel-size-local 1 --data-parallel-address "$address" --data-parallel-rpc-port 29550 \
        --data-parallel-start-rank 1 --headless \
        > >(sed -u "s/^/[$role 1] /") 2>&1 &
}

echo "[pd] prefill ${nodes[0]} ${nodes[1]}, decode ${nodes[2]} ${nodes[3]}, proxy $(hostname -s):8000"
instance "${nodes[0]}" "${nodes[1]}" 8100 prefill "$@"
instance "${nodes[2]}" "${nodes[3]}" 8200 decode "$@"
PYTHONPATH=$MILABENCH_BASE/pd "$V/bin/python" "$(dirname "$0")/pd_proxy.py" --host 127.0.0.1 --port 8000 \
    --prefiller-hosts "$(ip "${nodes[0]}")" --prefiller-ports 8100 --decoder-hosts "$(ip "${nodes[2]}")" --decoder-ports 8200 \
    > >(sed -u "s/^/[proxy] /") 2>&1 &
trap 'kill $(jobs -p) 2>/dev/null; wait' EXIT
wait -n
echo "[pd] a child exited: stopping" >&2
exit 1
