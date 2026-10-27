"""Runs inside slime's Megatron Ray actors (``--custom-megatron-init-path slime_hook.install``).

The driver never sees training steps (they run in Ray actors), so everything milabench should know is pushed
from here to the collector actor created by ``main.py``, which gives it to voir:

* ``rate`` / ``tflops`` / ``loss`` / ``step_time`` and slime's other ``perf/*`` metrics (logging rank only),
* the wall time of every rank's training step (``step_time_rank``: straggler / per-node spread),
* the MoE expert load (``expert_imbalance`` = busiest expert / mean expert, averaged over the layers) and
  whether the router is currently forced to balance,
* a routing schedule: ``schedule.router.force_balance_until: N`` forces balanced routing for the first N
  rollouts, then switches to the real router inside the same run (read from ``SLIME_SCHEDULE``).
"""

import json
import os
import socket
import time
import weakref

_TOKENS = "perf/actor_train_tok_per_s"
_TFLOPS = "perf/actor_train_tflops"
_STEP_TIME = "perf/step_time"

_state = {"rollout_id": -1}
_routers = weakref.WeakSet()
_schedule_cache = {}


def schedule():
    raw = os.environ.get("SLIME_SCHEDULE") or ""
    if raw not in _schedule_cache:
        _schedule_cache[raw] = json.loads(raw) if raw else {}
    return _schedule_cache[raw]


def force_balance(rollout_id, router_schedule):
    """None: no schedule (leave the router alone); else whether routing is forced to balance at rollout_id."""
    until = (router_schedule or {}).get("force_balance_until")
    return None if until is None else rollout_id < int(until)


def rank():
    try:
        import torch.distributed as dist

        return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    except Exception:
        return 0


def _forwarded(key):
    return key.startswith("perf/") or "response_len" in key or "raw_reward" in key


def install(args):
    from slime.observability import logging_utils

    import slime_metrics

    collector = None

    def push(event):
        nonlocal collector
        try:
            if collector is None:
                collector = slime_metrics.get()
            collector.push.remote(event)
        except Exception:
            pass

    _wrap_log(logging_utils, push)
    _wrap_train(push)
    _patch_router()


def _wrap_log(logging_utils, push):
    if getattr(logging_utils.log, "_milabench", False):
        return

    orig = logging_utils.log

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
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and _forwarded(key) and key not in (_TOKENS, _TFLOPS, _STEP_TIME):
                    push({key.replace("/", "_"): float(value), "task": "train", "time": now})
        except Exception:
            pass
        return orig(args, metrics, step_key)

    log._milabench = True
    logging_utils.log = log


def _wrap_train(push):
    """Time every rank's training step and keep the current rollout id for the router schedule."""
    import sys

    actor = sys.modules.get("slime.backends.megatron_utils.actor")
    if actor is None:
        import slime.backends.megatron_utils.actor as actor

    if getattr(actor.train, "_milabench", False):
        return

    orig = actor.train
    host = socket.gethostname()

    def train(rollout_id, *args, **kwargs):
        _state["rollout_id"] = rollout_id
        start = time.time()
        try:
            return orig(rollout_id, *args, **kwargs)
        finally:
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                now = time.time()
                r = rank()
                push({"step_time_rank": now - start, "units": "s", "rank": r, "node": host,
                      "rollout_id": rollout_id, "task": "train", "time": now})
                imbalance, forced = flush_expert_load()
                if imbalance is not None and r == 0:
                    push({"expert_imbalance": imbalance, "router_forced": bool(forced),
                          "rollout_id": rollout_id, "task": "train", "time": now})
            except Exception:
                pass

    train._milabench = True
    actor.train = train


def flush_expert_load():
    """(mean over layers of busiest-expert / mean-expert token count, any router forced) since the last flush."""
    ratios, forced = [], False
    for router in list(_routers):
        counts = getattr(router, "_mb_counts", None)
        if counts is None:
            continue
        counts = counts.float()
        mean = counts.mean()
        if mean > 0:
            ratios.append((counts.max() / mean).item())
        forced = forced or bool(router.config.moe_router_force_load_balancing)
        router._mb_counts = None
    return (sum(ratios) / len(ratios) if ratios else None), forced


def _patch_router():
    try:
        from megatron.core.transformer.moe.router import TopKRouter
    except Exception:
        return

    if getattr(TopKRouter.forward, "_milabench", False):
        return

    orig = TopKRouter.forward

    def forward(self, *args, **kwargs):
        forced = force_balance(_state["rollout_id"], schedule().get("router"))
        if forced is not None:
            self.config.moe_router_force_load_balancing = forced
        probs, routing_map = orig(self, *args, **kwargs)
        try:
            counts = routing_map.sum(dim=0)
            previous = getattr(self, "_mb_counts", None)
            self._mb_counts = counts if previous is None else previous + counts
            _routers.add(self)
        except Exception:
            pass
        return probs, routing_map

    forward._milabench = True
    TopKRouter.forward = forward
