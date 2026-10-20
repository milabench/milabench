#!/usr/bin/env python
"""slime driver: runs ``train.main`` against a Ray cluster.

milabench owns the process and the arguments: the slime command line comes
from the bench config (``model.argv`` + ``argv``), see benchfile.py. For
multi-node runs ``RayCluster`` has already started Ray on every node
(``RAY_ADDRESS=auto``); otherwise ``ray.init()`` starts a local cluster.

Training steps happen in Ray actors, so ``slime_hook.py`` (installed through
``--custom-megatron-init-path``) pushes rate/loss events to a collector actor
(``slime_metrics.py``); this process drains it and sends them to milabench through ``setupvoir``'s
log function (stdout smuggling).
"""

from __future__ import annotations

import os
import sys
import threading
import time

from benchmate.monitor import setupvoir

import common


def forward_metrics(collector, log, stop: threading.Event):
    """Move events from the workers' collector actor to milabench."""
    import ray

    while True:
        done = stop.is_set()
        for event in ray.get(collector.drain.remote()):
            log(event)
        if done:
            return
        time.sleep(0.5)


# environment forwarded to the Ray workers (they do not inherit the driver's)
_FORWARD = ("PYTHONPATH", "CUDA_DEVICE_MAX_CONNECTIONS", "PYTORCH_CUDA_ALLOC_CONF",
            "PYTHONUNBUFFERED", "NCCL_NVLS_ENABLE", "NCCL_CUMEM_ENABLE", "TOKENIZERS_PARALLELISM",
            "HF_HOME", "HF_TOKEN")


def main():
    for p in reversed(os.environ["PYTHONPATH"].split(os.pathsep)):
        sys.path.insert(0, p)

    # No voirfile: it would need `voir --config ...` and the config is not on the
    # other nodes. Metrics/monitoring are started from the script itself, on the
    # main node (GPU monitor covers the main node's GPUs only).
    log, monitor = setupvoir(monogpu=False, interval=0.5)

    stop = threading.Event()
    forwarder = None

    try:
        import ray

        import slime_metrics

        env_vars = {k: os.environ[k] for k in _FORWARD if k in os.environ}
        # RAY_ADDRESS is honored when set
        ray.init(namespace=slime_metrics.NAMESPACE, runtime_env={"env_vars": env_vars})

        collector = slime_metrics.create()
        forwarder = threading.Thread(target=forward_metrics, args=(collector, log, stop), daemon=True)
        forwarder.start()

        sys.argv = ["train.py", *common.expand_multi_value(sys.argv[1:])]

        import train  # slime/train.py
        from slime.observability.logging_utils import configure_logger
        from slime.utils.arguments import parse_args

        args, restore_plan = parse_args(return_restore_plan=True)
        configure_logger()
        train.main(args, restore_plan)
    finally:
        stop.set()
        if forwarder is not None:
            forwarder.join(timeout=10)
        monitor.stop()


if __name__ == "__main__":
    main()
