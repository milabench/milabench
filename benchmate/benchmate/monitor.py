import json
import sys
import os
import time
from contextlib import contextmanager

from voir.instruments.utils import Monitor, _Monitor, monitor as generic_monitor
from voir.smuggle import SmuggleWriter
from voir.tools import instrument_definition
from voir.instruments.cpu import cpu_monitor, process_monitor
from voir.instruments.gpu import gpu_monitor as gpu_monitor_fun, select_backend
from voir.instruments.io import io_monitor
from voir.instruments.network import network_monitor
from voir.instruments.monitor import monitor
from voir.helpers import current_overseer


from .metrics import sumggle_push, give_push, file_push
from .toggles import (
    get_poll_interval,
    log_pattern,
    get_observation_count,
    torchmem_enabled,
    jaxmem_enabled,
)
from .torchmem import torchmem_fetcher
from .jaxmem import jaxmem_fetcher

# NVML gpudata poll (load/memory/power). Other monitors use SYSTEM_POLL_INTERVAL.
DEFAULT_SYSTEM_POLL_INTERVAL = 1


def _torchmem_kwargs():
    if torchmem_enabled():
        return {"torchmem": torchmem_fetcher()}
    return {}


def _jaxmem_kwargs():
    if jaxmem_enabled():
        return {"jaxmem": jaxmem_fetcher()}
    return {}


def log_patterns():
    debug_metrics = ("__iter__", "overhead", "process_time")

    base_metrics = (
        "value", "progress", "rate", "units", "loss",  "time"
    )

    gpu_metrics = (
        "gpudata", "memory_peak", "torchmem", "jaxmem",
    )

    system_metrics = (
        "cpudata", "process", "iodata", "netdata"
    )

    lean = base_metrics + gpu_metrics + system_metrics

    match log_pattern.lower():
        case 'lean':
            return lean

        case 'debug':
            return lean + debug_metrics

        case 'all':
            return "*"

        case _:
            return lean


def auto_push():
    # use_stdout = int(os.getenv("MILABENCH_USE_STDOUT", 0))
    mb_managed = int(os.getenv("MILABENCH_MANAGED", 0))

    # Milabench managed: we need to push metrics to it
    if mb_managed == 1:
        # Using voir, DATA_FD is defined as well
        ov = current_overseer.get()
        if ov is not None:
            return ov.give
        
        # Not using Voir, using structured stdout
        if int(os.getenv("MILABENCH_USE_STDOUT", 0)) == 1:
            return sumggle_push()

        raise RuntimeError("Could not find something to push to")

    # Not using milabench; using stdout
    return file_push()


@instrument_definition
def monitor_monogpu(ov, poll_interval=0.25, arch=None):
    return monitor(
        ov,
        poll_interval=get_poll_interval(poll_interval),
        gpudata=gpu_monitor_fun(),
        worker_init=lambda: select_backend(arch, force=True),
    )


@instrument_definition
def monitor_allocmem(ov, poll_interval=DEFAULT_SYSTEM_POLL_INTERVAL, arch=None):
    mem = {**_torchmem_kwargs(), **_jaxmem_kwargs()}
    return monitor(
        ov,
        poll_interval=get_poll_interval(poll_interval),
        **mem,
    )


@instrument_definition
def monitor_process_monogpu(ov, poll_interval=DEFAULT_SYSTEM_POLL_INTERVAL, arch=None):
    return monitor(
        ov,
        poll_interval=get_poll_interval(poll_interval),
        process=process_monitor(os.getpid()),
    )


@instrument_definition
def monitor_node_gpu(ov, poll_interval=0.25, arch=None):
    return monitor(
        ov,
        poll_interval=get_poll_interval(poll_interval),
        gpudata=gpu_monitor_fun(),
        worker_init=lambda: select_backend(arch, force=True),
    )


@instrument_definition
def monitor_node_system(ov, poll_interval=DEFAULT_SYSTEM_POLL_INTERVAL, arch=None):
    return monitor(
        ov,
        poll_interval=get_poll_interval(poll_interval),
        iodata=io_monitor(),
        netdata=network_monitor(),
        cpudata=cpu_monitor(),
        **_torchmem_kwargs(),
        **_jaxmem_kwargs(),
    )


# Backward-compatible alias (gpudata-only at ``poll_interval``).
monitor_node = monitor_node_gpu


def _split_smuggle_monitors(monitors):
    worker_init = monitors.pop("worker_init", None)
    gpu = {k: v for k, v in monitors.items() if k == "gpudata"}
    system = {k: v for k, v in monitors.items() if k != "gpudata"}
    return gpu, system, worker_init


def _smuggle_get(monitors):
    def get():
        t = time.time()
        return [
            {"task": "main", "time": t, k: v()}
            for k, v in monitors.items()
        ]

    return get


def _smuggle_monitor(
    gpu_poll_interval=0.25,
    system_poll_interval=DEFAULT_SYSTEM_POLL_INTERVAL,
    worker_init=None,
    **monitors,
):
    gpu_monitors, system_monitors, init = _split_smuggle_monitors(monitors)
    if worker_init is None:
        worker_init = init

    data_file = SmuggleWriter(sys.stdout)

    def mblog(data):
        if data_file is not None:
            try:
                print(json.dumps(data), file=data_file)
            except ValueError:
                pass

    def push(entries):
        for entry in entries:
            mblog(entry)

    if worker_init is not None:
        worker_init()

    threads = []
    if gpu_monitors:
        gpu_get = _smuggle_get(gpu_monitors)
        threads.append(
            Monitor(
                get_poll_interval(gpu_poll_interval),
                lambda: push(gpu_get()),
            )
        )
    if system_monitors:
        sys_get = _smuggle_get(system_monitors)
        threads.append(
            Monitor(
                get_poll_interval(system_poll_interval),
                lambda: push(sys_get()),
            )
        )

    if not threads:
        mon = Monitor(1, lambda: None)
    elif len(threads) == 1:
        mon = threads[0]
    else:
        mon = _Monitor(*threads)

    mon.start()
    return mblog, mon


@contextmanager
def smuggle_monitor(
    poll_interval=0.25,
    system_poll_interval=DEFAULT_SYSTEM_POLL_INTERVAL,
    worker_init=None,
    enabled=True,
    **monitors,
):
    if enabled:
        # rank == 0
        mblog, mon = _smuggle_monitor(
            gpu_poll_interval=poll_interval,
            system_poll_interval=system_poll_interval,
            worker_init=worker_init,
            **monitors,
        )

        try:
            yield mblog
        finally:
            mon.stop()
        
    else:
        # rank > 0
        yield


def _monitors(monogpu=True, *, torchmem=None, jaxmem=None):
    """Build smuggle/voir monitor callables.

    ``torchmem`` / ``jaxmem``:
      - ``None``: follow ``BENCHMATE_TORCHMEM`` / ``BENCHMATE_JAXMEM`` toggles
      - ``True`` / ``False``: force on/off (e.g. parent DDP process disables
        torchmem; rank-0 worker enables a torchmem-only poller)
    """
    if monogpu:
        monitors = [
            ("gpudata", gpu_monitor_fun()),
            # This is too slow and slows down everything
            # ("process", process_monitor(os.getpid())),
            ("worker_init", lambda: select_backend(None, True)),
        ]
    else:
        monitors = [
            ("gpudata", gpu_monitor_fun()),
            ("iodata", io_monitor()),
            ("netdata", network_monitor()),
            ("cpudata", cpu_monitor()),
            ("worker_init", lambda: select_backend(None, True)),
        ]

    if torchmem is None:
        monitors.extend(_torchmem_kwargs().items())
    elif torchmem:
        monitors.append(("torchmem", torchmem_fetcher()))

    if jaxmem is None:
        monitors.extend(_jaxmem_kwargs().items())
    elif jaxmem:
        monitors.append(("jaxmem", jaxmem_fetcher()))

    return dict(monitors)


@contextmanager
def multigpu_monitor(*args, torchmem=None, jaxmem=None, **kwargs):
    with smuggle_monitor(
        *args, **kwargs, **_monitors(False, torchmem=torchmem, jaxmem=jaxmem)
    ) as log:
        yield log


@contextmanager
def monogpu_monitor(*args, torchmem=None, jaxmem=None, **kwargs):
    with smuggle_monitor(
        *args, **kwargs, **_monitors(True, torchmem=torchmem, jaxmem=jaxmem)
    ) as log:
        yield log


@contextmanager
def torchmem_monitor(poll_interval=3, device=None, **kwargs):
    """Smuggle only PyTorch allocator stats (for DDP workers that own tensors)."""
    with smuggle_monitor(
        system_poll_interval=poll_interval,
        torchmem=torchmem_fetcher(device=device),
        **kwargs,
    ) as log:
        yield log


@contextmanager
def rank0_torchmem_monitor(*args, **kwargs):
    """Like ``torchmem_monitor``, but only active when ``RANK=0``."""
    if get_rank() == 0:
        with torchmem_monitor(*args, **kwargs) as mon:
            yield mon
    else:
        yield


@contextmanager
def bench_monitor(*args, **kwargs):
    if get_rank() == -1:
        with monogpu_monitor(*args, **kwargs) as mon:
            yield mon

    elif get_rank() == 0:
        with multigpu_monitor(*args, **kwargs) as mon:
            yield mon
    else:
        yield

#
# Legacy compatibility
#
def setupvoir(monogpu=True, enabled=True, interval=0.25, system_interval=DEFAULT_SYSTEM_POLL_INTERVAL):
    return _smuggle_monitor(
        gpu_poll_interval=get_poll_interval(interval),
        system_poll_interval=get_poll_interval(system_interval),
        **_monitors(monogpu),
    )


def milabench_sys_monitor(monogpu=False):
    return setupvoir(monogpu)



def get_rank():
    try:
        return int(os.getenv("RANK", "-1"))
    except (TypeError, ValueError):
        return -1


def voirfile_monitor(ov, options):
    from voir.instruments import early_stop, log, dash

    if options.dash:
        ov.require(dash)
    
    instruments = [
        log(
            *log_patterns(), context="task"
        )
    ] 

    rank = get_rank()

    # -1 & 0 early stop
    if rank <= 0:
        instruments.append(
            early_stop(n=get_observation_count(options.stop), key="rate", task="train", signal="stop")
        )
    
    gpu_poll = get_poll_interval(options.gpu_poll)
    system_poll = get_poll_interval(DEFAULT_SYSTEM_POLL_INTERVAL)

    # mono gpu if rank is not set
    if rank == -1:
        instruments.append(monitor_monogpu(poll_interval=gpu_poll))
        if _torchmem_kwargs() or _jaxmem_kwargs():
            instruments.append(monitor_allocmem(poll_interval=system_poll))
        instruments.append(monitor_process_monogpu(poll_interval=system_poll))

    # rank is set only monitor main rank
    if rank == 0:
        instruments.append(monitor_node_gpu(poll_interval=gpu_poll))
        instruments.append(monitor_node_system(poll_interval=system_poll))

    ov.require(*instruments)
