from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
from pathlib import Path

from ..toggles import nodemon_dir
from . import (
    CollectorRegistry,
    RunState,
    Sampler,
    TsvSink,
    default_registry,
    run,
    short_hostname,
)

log = logging.getLogger("benchmate.nodemon")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """The command-line options."""
    parser = argparse.ArgumentParser(
        prog="python -m benchmate.nodemon",
        description="Sample node health (GPU, NVLink, InfiniBand, CPU, memory, PCIe) into TSV files, once per host.",
    )
    parser.add_argument(
        "--out", type=Path, default=nodemon_dir(),
        help="Output directory (default: BENCHMATE_NODEMON_DIR, now %(default)s; with neither, the sampler does not run)",
    )
    parser.add_argument(
        "--flush-every", type=float, default=10.0, metavar="SECONDS",
        help="Seconds between writes to the TSV files; 0 writes on every sample (default: 10)",
    )
    parser.add_argument("--bench", default="", help="Benchmark name, recorded in the status file")
    parser.add_argument(
        "--nice", type=int, default=10, choices=range(-20, 20), metavar="N",
        help="CPU scheduling niceness, -20 (highest priority) to 19 (lowest); "
        "going below the current one needs privileges (default: 10)",
    )
    parser.add_argument(
        "--log-level", default="INFO", type=str.upper,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return parser.parse_args(argv)


def log_init(level: str) -> None:
    """Log to stderr at `level`, keeping stdout empty."""
    logging.basicConfig(
        level=level, stream=sys.stderr, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )


def set_nice(niceness: int) -> None:
    """Run at `niceness`, keeping the current one when that is not permitted.

    On Linux this sets the calling thread's niceness, which threads started
    afterwards inherit; so call it before the sampler starts any.
    """
    try:
        os.setpriority(os.PRIO_PROCESS, 0, niceness)
    except OSError:
        log.exception(f"Can't set niceness {niceness}; keeping {os.getpriority(os.PRIO_PROCESS, 0)}.")


def main(argv: list[str] | None = None) -> int:
    """Run the sampler unless this host already has one; return the exit code."""
    args = parse_args(argv)
    log_init(args.log_level)
    if not args.out:
        log.info("Node sampler disabled: set BENCHMATE_NODEMON_DIR or pass --out.")
        return 0
    set_nice(args.nice)

    host = short_hostname()
    sink = TsvSink(args.out, host, args.flush_every)
    registry = default_registry(sink)

    signals = {
        signal.SIGTERM, # stop
        signal.SIGINT, # stop
        signal.SIGHUP, # stop
        signal.SIGUSR1 # write pending rows
    }
    signal.pthread_sigmask(signal.SIG_BLOCK, signals)
    sampler = run(args.out, host, registry, args.bench)
    state = serve(sampler, registry, signals)
    return {RunState.CLOSED: 0, RunState.BUSY: 0, RunState.REFUSED: 2}.get(state, 1)


def serve(sampler: Sampler, registry: CollectorRegistry, signals: set[signal.Signals]) -> RunState:
    """Take the stop and flush signals until the sampler ends; return its final state.

    Signals are taken synchronously with sigtimedwait() rather than by Python
    handlers, which run between bytecodes and could deadlock on a lock that the
    interrupted code holds. The timeout notices a run that ended by itself.
    """
    while sampler.state() == RunState.RUNNING:
        received = signal.sigtimedwait(signals, 1.0)
        if not received:
            continue
        if received.si_signo == signal.SIGUSR1:
            registry.flush()
        else:
            sampler.stop(f"signal {signal.Signals(received.si_signo).name}")
    return sampler.state()


if __name__ == "__main__":
    raise SystemExit(main())
