#!/usr/bin/env python
"""Benchmarks this machine must not run again: the ones it hung on, the ones it cannot run.

The list is per machine ($MILABENCH_BASE/homelab-quarantine.json): a benchmark that freezes a DGX Spark may
run on another machine. scripts/homelab/run.sh excludes every listed benchmark (INCLUDE_QUARANTINED=1 runs
them anyway, e.g. after a driver update), and before each run looks for benchmarks the machine died on.

    quarantine.py list
    quarantine.py add <bench> --reason "..." [--kind hang|unsupported|manual]
    quarantine.py remove <bench>             # released: its earlier crashes are not detected again
    quarantine.py detect <run dir>...     # benchmarks running when the machine hung or rebooted
    quarantine.py exclude                 # comma-separated names, for milabench --exclude

A benchmark the machine died on: its .data file has no end event and was last written before the current
boot (the machine restarted while it ran). For a serving pass (<bench>.<pass>) the benchmark is <bench>.
Stdlib only.
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import socket
import sys


def path() -> str:
    return os.path.join(os.environ.get("MILABENCH_BASE", "/opt/milabench"), "homelab-quarantine.json")


RELEASED = "_released"  # bench -> when it was released: crashes before that are not detected again


def load() -> dict:
    try:
        with open(path()) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def benches(entries: dict) -> dict:
    return {k: v for k, v in entries.items() if k != RELEASED}


def save(entries: dict):
    tmp = path() + ".tmp"
    with open(tmp, "w") as f:
        json.dump(entries, f, indent=1, sort_keys=True)
    os.replace(tmp, path())


def add(entries: dict, bench: str, reason: str, kind: str, run: str | None = None) -> bool:
    if bench in entries:
        return False
    entries[bench] = {"reason": reason, "kind": kind, "since": datetime.datetime.now().isoformat(timespec="seconds"),
                      "host": socket.gethostname(), **({"run": run} if run else {})}
    return True


def boot_time() -> float:
    with open("/proc/stat") as f:
        for line in f:
            if line.startswith("btime"):
                return float(line.split()[1])
    raise RuntimeError("no btime in /proc/stat")


def bench_of(data_file: str, config: dict) -> str:
    """The milabench benchmark of a .data file: a serving pass <bench>.<pass> belongs to <bench>."""
    name = config.get("name") or os.path.basename(data_file).split(".")[0]
    folder = os.path.dirname(data_file)
    if "." in name:
        parent = name.rsplit(".", 1)[0]
        if glob.glob(os.path.join(folder, f"{parent}.*server*.data")):
            return parent
    return name


def died_on(run_dir: str) -> list[tuple[str, str, float]]:
    """(bench, data file, last write) of the benchmarks running when the machine went down."""
    booted = boot_time()
    found = []
    for data in glob.glob(os.path.join(run_dir, "*.data")):
        if ".server." in data:
            continue
        mtime = os.path.getmtime(data)
        if mtime >= booted:
            continue  # written since this boot: not running when the machine went down
        config, ended = {}, False
        with open(data, errors="replace") as f:
            for line in f:
                if line.startswith('{"event": "config"'):
                    try:
                        config = json.loads(line)["data"]
                    except json.JSONDecodeError:
                        pass
                elif line.startswith('{"event": "end"'):
                    ended = True
        if not ended:
            found.append((bench_of(data, config), data, mtime))
    # only the last benchmark(s) of the run: an earlier one without end was stopped by milabench itself
    if not found:
        return []
    last = max(m for _, _, m in found)
    return [f for f in found if last - f[2] < 120]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    a = sub.add_parser("add")
    a.add_argument("bench")
    a.add_argument("--reason", required=True)
    a.add_argument("--kind", default="manual", choices=["hang", "unsupported", "manual"])
    r = sub.add_parser("remove")
    r.add_argument("bench")
    d = sub.add_parser("detect")
    d.add_argument("run_dirs", nargs="+")
    sub.add_parser("exclude")
    args = parser.parse_args(argv)

    entries = load()
    if args.cmd == "list":
        for bench, e in sorted(benches(entries).items()):
            print(f"{bench:50s} {e['kind']:12s} {e['since']}  {e['reason']}")
    elif args.cmd == "add":
        if add(entries, args.bench, args.reason, args.kind):
            save(entries)
        print(f"quarantined {args.bench}")
    elif args.cmd == "remove":
        entries.pop(args.bench, None)
        entries.setdefault(RELEASED, {})[args.bench] = datetime.datetime.now().timestamp()
        save(entries)
        print(f"released {args.bench}")
    elif args.cmd == "detect":
        changed = False
        for run_dir in args.run_dirs:
            for bench, data, mtime in died_on(run_dir):
                if mtime < entries.get(RELEASED, {}).get(bench, 0):
                    continue  # released after that crash
                when = datetime.datetime.fromtimestamp(mtime).isoformat(timespec="seconds")
                reason = f"the machine hung or rebooted while it ran ({os.path.basename(data)}, last written {when})"
                if add(entries, bench, reason, "hang", run=os.path.basename(os.path.normpath(run_dir))):
                    print(f"quarantined {bench}: {reason}")
                    changed = True
        if changed:
            save(entries)
    elif args.cmd == "exclude":
        print(",".join(sorted(benches(entries))))


if __name__ == "__main__":
    sys.exit(main())
