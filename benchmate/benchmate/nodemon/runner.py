from __future__ import annotations

import atexit
import fcntl
import json
import logging
import os
import socket
import threading
import time
from enum import Enum
from pathlib import Path
from typing import IO, Any

from .clock import clock_state
from .collectors import default_registry
from .registry import CollectorRegistry
from .sink import TsvSink, epoch_ms

log = logging.getLogger(__name__)


class RunState(str, Enum):
    """The state of a sampler run, as returned by run() and written to the status file."""

    # A str mixin rather than StrEnum (Python 3.11+), which benchmate can't require yet.
    RUNNING = "running"
    CLOSED = "closed"
    FAILED = "failed"
    BUSY = "busy"
    REFUSED = "refused"


class Stop(threading.Event):
    """A stop event that remembers the first reason it was set for."""

    def __init__(self) -> None:
        super().__init__()
        self.reason: str | None = None

    def set(self, reason: str | None = None) -> None:
        if reason and not self.reason:
            self.reason = reason
        super().set()


def short_hostname() -> str:
    """This host's name without its domain, as in Slurm node lists."""
    return socket.gethostname().split(".")[0]


def lock_path(out_dir: Path, host: str) -> Path:
    return Path(out_dir) / "nodes" / f"{host}.lock"


def host_lock(out_dir: Path, host: str) -> IO[str] | None:
    """Take this host's sampler lock, or None when another sampler holds it.

    Keep the returned file open for as long as the sampler runs. The kernel
    releases the lock when the process exits, even on SIGKILL.
    """
    path = lock_path(out_dir, host)
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a")
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        stream.close()
        return None
    return stream


def write_status(path: Path, record: dict[str, Any]) -> None:
    """Replace the status file atomically, so a reader never sees half of it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(path)


class Sampler:
    """A sampler started by run(): stop it with stop(), and read where it is with state()."""

    def __init__(self, state: RunState) -> None:
        self._state = state
        self._stop = Stop()
        self._thread: threading.Thread | None = None

    def state(self) -> RunState:
        """RUNNING until the run has ended and written its final status, then CLOSED or FAILED.

        BUSY or REFUSED when it never started; see run().
        """
        return self._state

    def stop(self, reason: str = "stopped", timeout_s: float = 10.0) -> RunState:
        """Stop sampling, write the pending rows and the final status, and return the final state.

        Waits up to `timeout_s` for that, which covers the run's `join_timeout`
        plus the final writes; still RUNNING when it ran out. Safe to call twice,
        or after the run ended by itself. Don't call it from a Python signal
        handler, which could deadlock on the stop event.
        """
        self._stop.set(reason)
        if self._thread:
            self._thread.join(timeout_s)
            if self._thread.is_alive():
                log.warning(f"The sampler did not stop within {timeout_s}s; its final status is not written yet.")
        return self._state

    def _start(self, lock: IO[str], registry: CollectorRegistry, status: Path, *args: Any) -> None:
        self._thread = threading.Thread(
            target=self._supervise,
            args=(lock, registry, status, *args),
            name="nodemon-run",
            daemon=True,  # never keeps the process alive; the atexit stop() in run() closes it first
        )
        self._thread.start()

    def _supervise(self, lock: IO[str], registry: CollectorRegistry, status: Path, *args: Any) -> None:
        """Run to the end holding the host lock; publish the final state once the lock is released."""
        state = RunState.FAILED
        try:
            with lock:
                state = _run(registry, self._stop, status, *args)
        except Exception:
            log.exception("The sampler failed.")
        finally:
            self._state = state


def run(
    out_dir: Path,
    host: str | None = None,
    registry: CollectorRegistry | None = None,
    bench: str = "",
    join_timeout: float = 5.0,
    check_every: float = 1.0,
) -> Sampler:
    """Start this host's sampler in the background, unless the host already has one.

    `host` names the lock and status files, and defaults to short_hostname().
    `registry` defaults to default_registry() writing to a TsvSink in
    `out_dir`; a registry passed in should write there too, under `host`.

    The returned Sampler's state() is BUSY when another sampler holds this
    host's lock, and REFUSED when an earlier run's status file is in
    `out_dir`, whose rows would mix with this run's; neither samples
    anything. Otherwise it is RUNNING until stop() or the parent process
    exits, holding the lock until then.
    """
    host = host or short_hostname()
    lock = host_lock(out_dir, host)
    if not lock:
        # Expected when every task or rank starts a sampler, hence INFO.
        log.info(f"A sampler already runs on {host} (it holds {lock_path(out_dir, host)}); not starting another.")
        return Sampler(RunState.BUSY)
    status = Path(out_dir) / "nodes" / f"{host}.status.json"
    # Checked once the lock is held, so a second sampler on this host is BUSY instead.
    # Every run writes its status file before any row, so this also covers its TSV files.
    if status.exists():
        lock.close()
        log.error(f"An earlier run's output is already in {status.parent} ({status.name}); use a new directory per run.")
        return Sampler(RunState.REFUSED)
    if not registry:
        registry = default_registry(TsvSink(out_dir, host))
    sampler = Sampler(RunState.RUNNING)
    sampler._start(lock, registry, status, bench, join_timeout, check_every)
    # A caller that exits without stop() still writes the last rows and a closed status. It runs
    # at interpreter exit, not in a signal handler, so it can't deadlock; a kill skips it.
    atexit.register(sampler.stop, "exit")
    return sampler


def _run(
    registry: CollectorRegistry,
    stop: Stop,
    status: Path,
    bench: str,
    join_timeout: float,
    check_every: float,
) -> RunState:
    """Sample on a worker thread until stopped or orphaned; return the final state.

    The worker gets `join_timeout` seconds to finish its tick once stopped;
    the sink is flushed and closed either way, so a hung NVML call can't keep
    the rows collected before it from being written.
    """
    clock = clock_state()
    if clock.synced is False:  # None is unknown, which leaves nothing to act on
        log.warning(
            f"The kernel clock is not NTP-synchronized (max error {clock.maxerror_us} µs): "
            "this node's row times may not line up with other nodes'."
        )
    record: dict[str, Any] = {
        "state": RunState.RUNNING.value,
        "bench": bench,
        "pid": os.getpid(),
        "start": epoch_ms(time.time()),
        "flush_every": getattr(registry.sink, "flush_every", None),
        "clock": clock._asdict(),  # a plain tuple would write a JSON list
    }
    outcome: dict[str, str] = {}
    registry.start()
    record["collectors"] = registry.describe()
    write_status(status, record)
    worker = threading.Thread(
        target=_sample_loop,
        args=(registry, stop, check_every, outcome),
        name="nodemon-sampler",
        daemon=True,  # a sampler stuck in NVML must not keep the process alive
    )
    worker.start()
    try:
        stop.wait()
    finally:
        stop.set()
        worker.join(join_timeout)
        stuck = registry.sampling if worker.is_alive() else None
        registry.close(release_collectors=not stuck)
        # Again at the end: collectors disabled while running, and their CPU seconds.
        record["collectors"] = registry.describe()
        state = RunState.FAILED if "error" in outcome else RunState.CLOSED
        record["state"] = state.value
        record["reason"] = outcome.get("reason") or stop.reason or "stopped"
        if "error" in outcome:
            record["error"] = outcome["error"]
        if stuck:
            record["stuck"] = stuck
        record["end"] = epoch_ms(time.time())
        record["clock_end"] = clock_state()._asdict()
        write_status(status, record)
    return state


def _sample_loop(
    registry: CollectorRegistry,
    stop: threading.Event,
    check_every: float,
    outcome: dict[str, str],
) -> None:
    """Tick the registry until stopped or the parent process is gone."""
    parent = os.getppid()
    try:
        while not stop.is_set():
            now = time.monotonic()
            if os.getppid() != parent:  # the launcher died without signalling
                outcome["reason"] = "orphan"
                break
            registry.tick(now)
            stop.wait(max(0.0, min(registry.due() - time.monotonic(), check_every)))
    except Exception as exc:
        log.exception("The sampling loop failed.")
        outcome["error"] = repr(exc)
    finally:
        stop.set()
