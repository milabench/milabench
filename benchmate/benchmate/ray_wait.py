#!/usr/bin/env python3
"""raywait: wait until a Ray cluster has the expected number of alive nodes."""

import argparse
import sys
import time

import ray


def wait_for_cluster(address: str, expected: int, timeout: int) -> int:
    deadline = time.time() + timeout
    alive = 0
    while time.time() < deadline:
        try:
            if not ray.is_initialized():
                ray.init(address=address, ignore_reinit_error=True)
        except ConnectionError:
            print(f"Waiting for Ray head at {address}...")
            time.sleep(5)
            continue
        alive = sum(1 for n in ray.nodes() if n.get("Alive"))
        if alive >= expected:
            print(f"Ray cluster ready: {alive}/{expected}")
            return 0
        print(f"Waiting for Ray workers: {alive}/{expected}")
        time.sleep(5)
    print(f"Timed out waiting for Ray cluster: {alive}/{expected}")
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--address", required=True, help="Ray head address host:port")
    parser.add_argument(
        "--expected", type=int, required=True, help="Expected alive node count"
    )
    parser.add_argument(
        "--timeout", type=int, default=600, help="Timeout in seconds (default: 600)"
    )
    args = parser.parse_args(argv)
    return wait_for_cluster(args.address, args.expected, args.timeout)


if __name__ == "__main__":
    sys.exit(main())
