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

import faulthandler
import json
import os
import sys
import threading
import time
import traceback

faulthandler.enable()


def debug(*msg):
    """Progress marker on stderr, also written under the runs dir (rsynced with the results).

    stdout is reserved for milabench's smuggled events."""
    line = f"[slime-main pid={os.getpid()} t={time.strftime('%H:%M:%S')}] " + " ".join(map(str, msg))
    print(line, file=sys.stderr, flush=True)
    runs = os.environ.get("MILABENCH_DIR_RUNS")
    if runs:
        try:
            with open(os.path.join(runs, "slime-main.debug.log"), "a") as f:
                f.write(line + "\n")
        except OSError:
            pass


debug("starting", "python", sys.executable, "argv", sys.argv[1:6], "...")
debug("RAY_ADDRESS", os.environ.get("RAY_ADDRESS"), "CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES"),
      "LD_LIBRARY_PATH", os.environ.get("LD_LIBRARY_PATH", "")[:300])

from benchmate.monitor import setupvoir

import common


def forward_metrics(collector, log, stop: threading.Event, started: float):
    """Move events from the workers' collector actor to milabench."""
    import ray

    count = 0
    first_step = False
    while True:
        done = stop.is_set()
        try:
            for event in ray.get(collector.drain.remote()):
                if count < 3:
                    debug("forwarding event", event)
                count += 1
                log(event)
                if not first_step and "rate" in event:
                    # launch -> first training step: Ray, model load, rollout engines, first rollout, first step
                    first_step = True
                    log({"time_to_first_step": time.time() - started, "units": "s", "task": "main", "time": time.time()})
        except Exception:
            debug("forward_metrics failed:\n" + traceback.format_exc())
            return
        if done:
            debug("forwarded", count, "events")
            return
        time.sleep(0.5)


# environment forwarded to the Ray workers (they do not inherit the driver's)
_FORWARD = ("PYTHONPATH", "CUDA_DEVICE_MAX_CONNECTIONS", "PYTORCH_CUDA_ALLOC_CONF",
            "PYTHONUNBUFFERED", "NCCL_NVLS_ENABLE", "NCCL_CUMEM_ENABLE", "TOKENIZERS_PARALLELISM",
            "HF_HOME", "HF_TOKEN", "LD_LIBRARY_PATH", "NVTE_CUDA_INCLUDE_DIR", "SLIME_SCHEDULE",
            "MEGATRON_USE_SGLANG_ROPE", "NVTE_FP8_BLOCK_SCALING_FP32_SCALES")


def main():
    debug("main: begin")
    for p in reversed(os.environ["PYTHONPATH"].split(os.pathsep)):
        sys.path.insert(0, p)

    # No voirfile: it would need `voir --config ...` and the config is not on the
    # other nodes. Metrics/monitoring are started from the script itself, on the
    # main node (GPU monitor covers the main node's GPUs only).
    log, monitor = setupvoir(monogpu=False, interval=0.5)
    debug("main: setupvoir done")

    stop = threading.Event()
    forwarder = None
    started = time.time()

    # in-run schedules (response-length cap, router forcing): read by slime_rollout.py / slime_hook.py in the workers
    if schedule := common.config().get("schedule"):
        os.environ["SLIME_SCHEDULE"] = json.dumps(schedule)
        debug("schedule", os.environ["SLIME_SCHEDULE"])

    try:
        import ray

        import slime_metrics

        env_vars = {k: os.environ[k] for k in _FORWARD if k in os.environ}
        # RAY_ADDRESS is honored when set
        ray.init(namespace=slime_metrics.NAMESPACE, runtime_env={"env_vars": env_vars})
        debug("main: ray initialised", ray.cluster_resources())

        collector = slime_metrics.create()
        forwarder = threading.Thread(target=forward_metrics, args=(collector, log, stop, started), daemon=True)
        forwarder.start()

        sys.argv = ["train.py", *common.expand_multi_value(sys.argv[1:])]

        import train  # slime/train.py
        from slime.observability.logging_utils import configure_logger
        from slime.utils.arguments import parse_args

        debug("main: parsing slime arguments")
        args, restore_plan = parse_args(return_restore_plan=True)
        configure_logger()
        debug("main: train.main")
        train.main(args, restore_plan)
        debug("main: train.main returned")
    except BaseException:
        debug("main: FAILED\n" + traceback.format_exc())
        raise
    finally:
        stop.set()
        if forwarder is not None:
            forwarder.join(timeout=10)
        monitor.stop()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        debug("fatal:\n" + traceback.format_exc())
        raise
    finally:
        debug("exit")
