"""Black box for machine hangs: kernel messages and GPU readings, written to disk and fsync'ed every second.

journald keeps the last minutes in memory, so a hard hang (DGX Spark / GB10 freezing under load) loses exactly
the messages that would explain it. This writes them to <out>/blackbox-<boot id>.log as they come.

    python3 scripts/homelab/blackbox.py /opt/milabench/runs &
"""
import os
import subprocess
import sys
import threading
import time

out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
boot = open("/proc/sys/kernel/random/boot_id").read().strip()[:8]
path = os.path.join(out_dir, f"blackbox-{time.strftime('%Y-%m-%d_%H-%M-%S')}-{boot}.log")
lock = threading.Lock()
f = open(path, "a", buffering=1)


def write(line):
    with lock:
        f.write(line.rstrip("\n") + "\n")
        f.flush()
        os.fsync(f.fileno())


def kernel():
    proc = subprocess.Popen(["journalctl", "-k", "-f", "-n", "0", "-o", "short-iso"],
                            stdout=subprocess.PIPE, text=True, errors="replace")
    for line in proc.stdout:
        write("KERN " + line)


def meminfo():
    info = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":", 1)
        info[k] = int(v.split()[0]) // 1024
    return info


threading.Thread(target=kernel, daemon=True).start()
write(f"START {time.strftime('%F %T')} boot {boot}")
query = "temperature.gpu,power.draw,clocks.sm,clocks_event_reasons.active,utilization.gpu"
while True:
    try:
        gpu = subprocess.run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout.strip()
    except subprocess.TimeoutExpired:
        gpu = "nvidia-smi TIMEOUT"
    m = meminfo()
    load = open("/proc/loadavg").read().split()[0]
    write(f"{time.strftime('%T')} gpu[{gpu}] used={(m['MemTotal'] - m['MemAvailable']) // 1024}G "
          f"free={m['MemFree'] // 1024}G cached={m['Cached'] // 1024}G load={load}")
    time.sleep(1)
