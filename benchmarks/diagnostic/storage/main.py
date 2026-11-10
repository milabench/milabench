#!/usr/bin/env python
"""Storage throughput diagnostic for the milabench data directory.

All I/O uses ``O_DIRECT`` so the page cache (TBs of RAM on GPU nodes) does not
hide the disks. Each test runs ``--workers`` processes that start together;
throughput is total bytes over the wall time from the first start to the last
finish.

* ``write``  each worker writes its own large file sequentially.
* ``read``   each worker reads its own large file sequentially.
* ``small``  workers read a shared set of small files (dataset-like, e.g. JPEGs).

The read files are created on first use and kept in ``<dir>/files`` so later runs
skip the setup; the write test only uses temporary files.
"""

from __future__ import annotations

import argparse
import mmap
import multiprocessing as mp
import os
import shutil
import time

from benchmate.monitor import setupvoir


GB = 1e9
MB = 1024**2
KB = 1024

TESTS = ("write", "read", "small")


def aligned_buffer(size):
    # anonymous mmap is page aligned, as O_DIRECT requires
    buf = mmap.mmap(-1, size)
    buf.write(os.urandom(min(size, MB)) * (size // min(size, MB)))
    return buf


def write_file(path, size, block):
    buf = aligned_buffer(block)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_DIRECT, 0o644)
    try:
        written = 0
        while written < size:
            written += os.writev(fd, [buf])
        os.fsync(fd)
    finally:
        os.close(fd)
    return written


def read_file(path, block, buf=None):
    buf = buf or aligned_buffer(block)
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    try:
        total = 0
        while n := os.readv(fd, [buf]):
            total += n
    finally:
        os.close(fd)
    return total


# --------------------------------------------------------------------------- workers
def _worker(job, barrier, results, i):
    kind, args = job
    barrier.wait()
    t0 = time.perf_counter()
    if kind == "write":
        path, size, block = args
        nbytes, nfiles = write_file(path, size, block), 1
    elif kind == "read":
        path, block = args
        nbytes, nfiles = read_file(path, block), 1
    elif kind == "small":
        paths, block = args
        buf = aligned_buffer(block)
        nbytes = sum(read_file(p, block, buf) for p in paths)
        nfiles = len(paths)
    results[i] = (nbytes, nfiles, t0, time.perf_counter())


def run_parallel(jobs):
    """Runs one job per process, all released at once. Returns (bytes, files, seconds)."""
    ctx = mp.get_context("fork")
    barrier = ctx.Barrier(len(jobs))
    results = ctx.Manager().dict()
    procs = [ctx.Process(target=_worker, args=(job, barrier, results, i)) for i, job in enumerate(jobs)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"worker failed with exit code {p.exitcode}")

    values = list(results.values())
    nbytes = sum(v[0] for v in values)
    nfiles = sum(v[1] for v in values)
    seconds = max(v[3] for v in values) - min(v[2] for v in values)
    return nbytes, nfiles, seconds


# --------------------------------------------------------------------------- setup
def ensure_files(paths, size, block):
    """Create missing (or wrongly sized) files, in parallel."""
    missing = [p for p in paths if not os.path.exists(p) or os.path.getsize(p) != size]
    if not missing:
        return
    print(f"[storage] creating {len(missing)} files of {size / MB:.2f} MiB", flush=True)
    with mp.get_context("fork").Pool(min(len(missing), os.cpu_count(), 64)) as pool:
        pool.starmap(write_file, [(p, size, min(block, size)) for p in missing])


def big_files(args):
    root = os.path.join(args.dir, "files", "big")
    os.makedirs(root, exist_ok=True)
    return [os.path.join(root, f"{i:04d}.bin") for i in range(args.workers)]


def small_files(args):
    root = os.path.join(args.dir, "files", f"small-{args.small_kb}k")
    paths = []
    for i in range(args.small_count):
        sub = os.path.join(root, f"{i // 1000:04d}")  # 1000 files per directory, like datasets
        paths.append(os.path.join(sub, f"{i:07d}.bin"))
    for sub in {os.path.dirname(p) for p in paths}:
        os.makedirs(sub, exist_ok=True)
    return paths


# --------------------------------------------------------------------------- tests
def test_write(args):
    tmp = os.path.join(args.dir, "tmp-write")
    os.makedirs(tmp, exist_ok=True)
    size, block = int(args.file_gb * 1024 * MB), args.block_kb * KB
    try:
        jobs = [("write", (os.path.join(tmp, f"{i:04d}.bin"), size, block)) for i in range(args.workers)]
        return run_parallel(jobs)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_read(args):
    size, block = int(args.file_gb * 1024 * MB), args.block_kb * KB
    paths = big_files(args)
    ensure_files(paths, size, block)
    return run_parallel([("read", (p, block)) for p in paths])


def test_small(args):
    size = args.small_kb * KB
    paths = small_files(args)
    ensure_files(paths, size, size)
    # interleave so every worker spans every directory
    shards = [paths[i :: args.workers] for i in range(args.workers)]
    return run_parallel([("small", (shard, size)) for shard in shards])


def main():
    default_dir = os.path.join(os.getenv("MILABENCH_DIR_DATA", "."), "storage")

    parser = argparse.ArgumentParser(description="Storage throughput diagnostic")
    parser.add_argument("--tests", type=str, default=",".join(TESTS), help=f"Comma separated subset of {TESTS}")
    parser.add_argument("--dir", type=str, default=default_dir, help="Directory on the storage to test")
    parser.add_argument("--workers", type=int, default=32, help="Parallel processes")
    parser.add_argument("--file-gb", type=float, default=2, help="Size of each worker's file (GiB) for read/write")
    parser.add_argument("--block-kb", type=int, default=4096, help="I/O size (KiB) for read/write")
    parser.add_argument("--small-kb", type=int, default=128, help="Size of each small file (KiB, multiple of 4)")
    parser.add_argument("--small-count", type=int, default=200000, help="Number of small files")
    parser.add_argument("--repeats", type=int, default=10, help="Runs per test")
    args = parser.parse_args()

    tests = [t.strip() for t in args.tests.split(",") if t.strip()]
    unknown = set(tests) - set(TESTS)
    if unknown:
        parser.error(f"unknown tests {sorted(unknown)}")

    os.makedirs(args.dir, exist_ok=True)
    print(
        f"[storage] dir={args.dir} fs={shutil.disk_usage(args.dir).free / GB:.0f} GB free "
        f"workers={args.workers} file={args.file_gb} GiB block={args.block_kb} KiB "
        f"small={args.small_count} x {args.small_kb} KiB",
        flush=True,
    )

    rows = []
    for name in TESTS:
        if name not in tests:
            continue
        fn = globals()[f"test_{name}"]
        for i in range(args.repeats):
            nbytes, nfiles, seconds = fn(args)
            rate = nbytes / seconds / GB
            row = dict(test=name, rate=rate, units="GB/s", size_bytes=nbytes, files=nfiles,
                       seconds=seconds, files_per_s=nfiles / seconds, iter=i)
            rows.append(row)
            print(
                f"[storage] {name:<6} iter={i} {rate:8.2f} GB/s  {nfiles / seconds:12.0f} files/s  "
                f"({nbytes / GB:.1f} GB in {seconds:.2f}s)",
                flush=True,
            )

    log, monitor = setupvoir(monogpu=False, interval=0.5)
    for row in rows:
        log({"task": "train", "time": time.time(), **row})
    monitor.stop()


if __name__ == "__main__":
    main()
