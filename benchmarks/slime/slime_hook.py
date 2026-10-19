"""Runs inside slime's Megatron Ray actors (``--custom-megatron-init-path slime_hook.install``).

The driver never sees training steps (they run in Ray actors), so we tap
``slime.observability.logging_utils.log`` in the worker and push the perf
metrics to the collector actor created by ``main.py``, which gives them to voir.
"""

import time

_TOKENS = "perf/actor_train_tok_per_s"
_TFLOPS = "perf/actor_train_tflops"
_STEP_TIME = "perf/step_time"


def install(args):
    from slime.observability import logging_utils

    import slime_metrics

    if getattr(logging_utils.log, "_milabench", False):
        return

    orig = logging_utils.log
    collector = None

    def push(event):
        nonlocal collector
        if collector is None:
            collector = slime_metrics.get()
        collector.push.remote(event)

    def log(args, metrics, step_key):
        try:
            now = time.time()
            if _TOKENS in metrics:
                push({"rate": metrics[_TOKENS], "units": "tokens/s", "task": "train", "time": now})
            if _TFLOPS in metrics:
                push({"tflops": metrics[_TFLOPS], "units": "TFLOP/s", "task": "train", "time": now})
            if _STEP_TIME in metrics:
                push({"step_time": metrics[_STEP_TIME], "units": "s", "task": "train", "time": now})
            if "train/loss" in metrics:
                push({"loss": metrics["train/loss"], "task": "train", "time": now})
        except Exception:
            pass
        return orig(args, metrics, step_key)

    log._milabench = True
    logging_utils.log = log
