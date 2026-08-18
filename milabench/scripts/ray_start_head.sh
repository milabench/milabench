#!/bin/bash
# Start a Ray head for Slurm/multinode: detached session + --block.
#
# Ray's Slurm guide keeps the head alive with ``ray start --head ... --block``
# (often backgrounded). milabench's head step must return so workers can start;
# we detach with setsid so the process cleaner does not tear down GCS when the
# wrapper exits. The head is stopped later via ``ray stop``.
set -euo pipefail

ray_bin=$1
shift

address=""
port=6379
prev=""
for arg in "$@"; do
  case "$prev" in
    --node-ip-address) address=$arg ;;
    --port) port=$arg ;;
  esac
  case "$arg" in
    --node-ip-address=*) address=${arg#*=} ;;
    --port=*) port=${arg#*=} ;;
  esac
  prev=$arg
done

if [[ -z "$address" ]]; then
  echo "ray_start_head: missing --node-ip-address" >&2
  exit 1
fi

setsid "$ray_bin" start "$@" --block </dev/null >/dev/null 2>&1 &
head_pid=$!

if [[ -n "${MILABENCH_RAY_HEAD_PIDFILE:-}" ]]; then
  mkdir -p "$(dirname "${MILABENCH_RAY_HEAD_PIDFILE}")"
  echo "${head_pid}" > "${MILABENCH_RAY_HEAD_PIDFILE}"
fi

deadline=$((SECONDS + 120))
while (( SECONDS < deadline )); do
  if ! kill -0 "$head_pid" 2>/dev/null; then
    echo "ray_start_head: head process exited early (pid ${head_pid})" >&2
    wait "$head_pid" || true
    exit 1
  fi
  if "$ray_bin" status --address "${address}:${port}" >/dev/null 2>&1; then
    echo "Ray head ready at ${address}:${port} (pid ${head_pid})"
    exit 0
  fi
  sleep 2
done

echo "ray_start_head: timed out waiting for head at ${address}:${port}" >&2
kill "$head_pid" 2>/dev/null || true
exit 1
