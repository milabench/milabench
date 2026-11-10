"""The py-spy dumps of pyspy_dump.sh over a time window: for each dump, the threads py-spy found active (running Python,
not waiting), with their innermost frames, to see what a stage's worker does during a stall.

    python3 -I stall_pyspy.py HOST.pyspy.txt FROM TO [FRAMES]     (FROM, TO: HH:MM:SS, the dumps' own clock)
"""
import re
import sys

path, t_from, t_to = sys.argv[1:4]
depth = int(sys.argv[4]) if len(sys.argv) > 4 else 8


def flush(stamp, threads):
    if not (t_from <= stamp[:8] <= t_to):
        return
    active = [(head, frames) for head, frames in threads if "(active" in head]
    print(f"=== {stamp}: {len(active)} active of {len(threads)} threads")
    for head, frames in active:
        name = re.search(r'"(.*)"', head)
        print(f"  {name.group(1) if name else head}")
        for f in frames[:depth]:
            print(f"      {f}")


stamp, threads = None, []
for line in open(path, errors="replace"):
    line = line.rstrip("\n")
    if line.startswith("=== "):
        if stamp:
            flush(stamp, threads)
        stamp, threads = line.split()[1], []
    elif line.startswith("Thread "):
        threads.append((line, []))
    elif line.startswith("    ") and threads:
        threads[-1][1].append(line.strip())
if stamp:
    flush(stamp, threads)
