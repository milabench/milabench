import sys

import torchcompat.core as accelerator
from benchmate.benchserve import (
    InferenceServer,
    run_benchmark_watched,
    set_metric_sink,
    split_args,
)
from server_backends import (
    build_server_command,
    resolved_server_backend,
    resolved_server_command,
)
from atom_aiter_compat import ensure_atom_aiter_compat


def prepare_voir():
    from benchmate.observer import BenchObserver
    from benchmate.monitor import bench_monitor
    from benchmate.toggles import get_observation_count

    observer = BenchObserver(
        accelerator.Event,
        earlystop=get_observation_count(120),
        batch_size_fn=lambda x: len(x[0]),
        raise_stop_program=False,
        stdout=True,
    )

    set_metric_sink(observer.record_metric)
    return observer, bench_monitor


def main(argv):
    ensure_atom_aiter_compat()

    server_argv, bench_argv = split_args(argv, skip_program=True)

    backend = resolved_server_backend()
    command = build_server_command(
        server_argv, backend=backend, command=resolved_server_command()
    )

    observer, bench_monitor = prepare_voir()

    with bench_monitor():
        with InferenceServer(command, name=f"{backend} server") as server:
            run_benchmark_watched(server, bench_argv)


if __name__ == "__main__":
    main(sys.argv)
