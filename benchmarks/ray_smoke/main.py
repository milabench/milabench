#!/usr/bin/env python
"""Prove the Ray cluster spreads work across every node, then measure real
inter-node bandwidth over Ray's own object transfer (not torch/NCCL/gloo --
see benchmarks/torchsrun for that path) so milabench's full metric pipeline
(benchmate.monitor.setupvoir -> [data] rate lines -> Score/Breakdown) gets
exercised with real numbers instead of a bare pass/fail.

Placement uses a STRICT_SPREAD placement group to pin one actor per node,
then checks each actor actually landed on a distinct hostname -- rather than
trusting `ray.nodes()` alone, which only shows who *joined*, not who *ran
anything*.
"""

import argparse
import socket
import sys
import threading
import time

import ray
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from benchmate.monitor import setupvoir

GIB = 1024**3
WARMUP_CHECKS = 5
WARMUP_INTERVAL_S = 2
PG_READY_TIMEOUT_S = 60
PG_ATTEMPTS = 3
WATCH_INTERVAL_S = 1


@ray.remote(num_cpus=1)
class NodeActor:
    def hostname(self):
        return socket.gethostname(), ray.util.get_node_ip_address()

    def receive(self, payload):
        return len(payload)

    def ping(self, other, nbytes, repeats, warmup):
        """Send `nbytes` to `other` `repeats` times; return per-call seconds.

        Payload is generated here (on this actor's node) so the timed RPC
        captures the real cross-node object transfer, not driver overhead.
        """
        import os

        payload = os.urandom(nbytes)
        for _ in range(warmup):
            ray.get(other.receive.remote(payload))

        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            ray.get(other.receive.remote(payload))
            times.append(time.perf_counter() - t0)
        return times


def alive_nodes():
    return [n for n in ray.nodes() if n.get("Alive")]


def print_node_table(nodes):
    for n in nodes:
        print(f"  {n.get('NodeManagerAddress')}  (node_id={n.get('NodeID', '')[:8]})")


def print_effective_health_check_config():
    """Confirm what this driver process actually sees for Ray's node-liveness
    RayConfig (RAY_health_check_* env vars) -- prints observed values rather
    than assumed ones, since a prior run's node death recurred even after
    both reducing prestarted workers and loosening these timeouts, which
    means at least one of those fixes isn't the (whole) story."""
    try:
        from ray._raylet import Config

        print("Effective Ray node health-check config (this process' view):")
        for name in (
            "health_check_initial_delay_ms",
            "health_check_period_ms",
            "health_check_timeout_ms",
            "health_check_failure_threshold",
        ):
            print(f"  {name} = {getattr(Config, name)()}")
    except Exception as e:
        print(f"  (could not read Config: {e})")


class NodeHealthWatcher:
    """Background poll of ray.nodes() so a node death gets a precise
    timestamp in the log instead of only being noticed later when some
    ray.get() times out. Elapsed time since watcher start pins down whether
    a death is an early race (~seconds after joining) or a later drop."""

    def __init__(self, interval=WATCH_INTERVAL_S):
        self.interval = interval
        self.start_t = time.monotonic()
        self._stop = threading.Event()
        self._known_alive = set()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._known_alive = {n["NodeID"] for n in alive_nodes()}
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=self.interval * 2)

    def _run(self):
        while not self._stop.is_set():
            time.sleep(self.interval)
            try:
                now_alive = {n["NodeID"] for n in alive_nodes()}
            except Exception:
                continue
            elapsed = time.monotonic() - self.start_t
            for node_id in self._known_alive - now_alive:
                print(
                    f"[watch t={elapsed:6.1f}s] node {node_id[:8]} went from "
                    f"alive to dead",
                    flush=True,
                )
            for node_id in now_alive - self._known_alive:
                print(
                    f"[watch t={elapsed:6.1f}s] node {node_id[:8]} (re)joined",
                    flush=True,
                )
            self._known_alive = now_alive


def wait_for_stable_heartbeats(expected):
    """A node can register with GCS slightly before its raylet's heartbeat
    cadence has stabilized; requesting a placement group right then has been
    observed to trip GCS's missed-heartbeat detector and kill the raylet
    within seconds of it joining. Require a stable alive-count across a few
    checks before touching the scheduler."""
    stable_checks = 0
    last_count = None
    for _ in range(WARMUP_CHECKS * 3):
        count = len(alive_nodes())
        if count == expected and count == last_count:
            stable_checks += 1
            if stable_checks >= WARMUP_CHECKS:
                return
        else:
            stable_checks = 0
        last_count = count
        time.sleep(WARMUP_INTERVAL_S)
    print(
        f"WARNING: alive node count never stabilized at {expected} "
        f"(last seen: {last_count}); proceeding anyway."
    )


def create_ready_placement_group(n):
    """A raylet can still die between pg creation and pg.ready() (see
    wait_for_stable_heartbeats). Retry with a fresh placement group sized to
    the currently-alive count rather than failing on one bad attempt."""
    for attempt in range(1, PG_ATTEMPTS + 1):
        current = len(alive_nodes())
        pg = placement_group([{"CPU": 1}] * current, strategy="STRICT_SPREAD")
        try:
            ray.get(pg.ready(), timeout=PG_READY_TIMEOUT_S)
            return pg, current
        except ray.exceptions.GetTimeoutError:
            remove_placement_group(pg)
            print(
                f"WARNING: placement group not ready after {PG_READY_TIMEOUT_S}s "
                f"(attempt {attempt}/{PG_ATTEMPTS}); alive nodes now: {len(alive_nodes())}"
            )
    raise RuntimeError(
        f"Placement group never became ready after {PG_ATTEMPTS} attempts"
    )


def measure_bandwidth(actors, host_info, nbytes, repeats, warmup):
    print(
        f"Measuring point-to-point bandwidth: {host_info[0][0]} -> {host_info[1][0]} "
        f"({nbytes / (1024 * 1024):.1f} MiB x {repeats}, {warmup} warmup)"
    )
    times = ray.get(actors[0].ping.remote(actors[1], nbytes, repeats, warmup))

    log, monitor = setupvoir(monogpu=False, interval=0.5)
    for i, t in enumerate(times):
        bandwidth = (nbytes / t / GIB) if t > 0 else 0.0
        latency_us = t * 1e6
        print(f"iter={i:02d} latency={latency_us:.1f} us bandwidth={bandwidth:.3f} GiB/s")
        log(
            {
                "task": "train",
                "rate": bandwidth,
                "units": "GiB/s",
                "latency_us": latency_us,
                "size_bytes": nbytes,
                "iter": i,
                "time": time.time(),
            }
        )
    monitor.stop()

    avg_bw = sum((nbytes / t / GIB) if t > 0 else 0.0 for t in times) / len(times)
    print(f"avg_bandwidth={avg_bw:.3f} GiB/s")


def main():
    parser = argparse.ArgumentParser(description="Ray cluster placement + bandwidth smoke test")
    parser.add_argument("--repeats", type=int, default=30, help="Timed transfer iterations")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument(
        "--size-mb", type=float, default=64.0, help="Payload size per transfer (MiB)"
    )
    parser.add_argument(
        "--address",
        default="auto",
        help="Ray head address (default 'auto': discover a local instance on this node)",
    )
    args = parser.parse_args()

    ray.init(address=args.address)
    print_effective_health_check_config()

    nodes = alive_nodes()
    n_expected = len(nodes)
    print(f"Ray cluster reports {n_expected} alive node(s):")
    print_node_table(nodes)

    watcher = NodeHealthWatcher().start()
    try:
        wait_for_stable_heartbeats(n_expected)
        pg, n_used = create_ready_placement_group(n_expected)
    finally:
        watcher.stop()
    if n_used != n_expected:
        print(f"NOTE: proceeding with {n_used} node(s) (started at {n_expected})")

    actors = [
        NodeActor.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg, placement_group_bundle_index=i
            )
        ).remote()
        for i in range(n_used)
    ]
    host_info = ray.get([a.hostname.remote() for a in actors])

    print("Actor placement (one actor pinned per node via STRICT_SPREAD):")
    hosts = set()
    for host, ip in host_info:
        print(f"  ran on {host} ({ip})")
        hosts.add(host)

    if len(hosts) != n_used:
        print(
            f"FAIL: expected {n_used} distinct nodes, only saw {len(hosts)}: "
            f"{sorted(hosts)}"
        )
        sys.exit(1)

    print(
        f"PASS: {n_used} actors landed on {len(hosts)} distinct nodes -- "
        f"the Ray cluster is genuinely distributing work."
    )

    if n_used < 2:
        print("Only 1 node available -- skipping bandwidth measurement (needs >= 2).")
        return

    nbytes = int(args.size_mb * 1024 * 1024)
    measure_bandwidth(actors, host_info, nbytes, args.repeats, args.warmup)


if __name__ == "__main__":
    main()
