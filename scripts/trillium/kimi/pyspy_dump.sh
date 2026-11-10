#!/usr/bin/env bash
# Stall investigation (README "Stall investigation"), one task per node from job.sh when PYSPY is set: the node's
# kernel.yama.ptrace_scope, then every second a non-blocking py-spy dump (the worker is not paused) of the node's oldest
# vLLM worker (the TP ranks of a stage do the same work), each headed by its time, into OUT_DIR/<host>.pyspy.txt. It
# shows the Python stack of a stage's host work in a stall, which vLLM's JIT monitor names only for Triton, CuTeDSL and
# TileLang compiles, and only at their start (CuTeDSL) or end (Triton).
#   pyspy_dump.sh PYSPY OUT_DIR
PYSPY=$1 OUT=$2/$(hostname -s).pyspy.txt
echo "$(hostname -s): kernel.yama.ptrace_scope $(cat /proc/sys/kernel/yama/ptrace_scope)" | tee "$OUT"
while :; do
    # vLLM titles its workers VLLM::Worker_PP<p>_TP<t>; Ray titles a running actor ray::RayWorkerProc...
    pid=$(pgrep -u "$USER" -o -f '^(VLLM::Worker_PP|ray::RayWorkerProc)')
    [ -n "$pid" ] && { echo "=== $(date +%T.%N | cut -c1-12) pid $pid"; "$PYSPY" dump --nonblocking --pid "$pid" 2>&1; } >> "$OUT"
    sleep 1
done
