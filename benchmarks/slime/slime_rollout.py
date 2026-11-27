"""Rollout function with a response-length schedule (``--rollout-function-path slime_rollout.generate_rollout``).

slime reads ``--rollout-max-response-len`` once (``GenerateState`` is a singleton that caches the sampling
params), so a cap that changes inside one run has to be pushed into both ``args`` and that cached state.
The schedule comes from the bench config (``schedule.response_len``: {rollout_id: cap}), through the
``SLIME_SCHEDULE`` environment variable that main.py forwards to the Ray workers.
"""

import json
import os


def schedule():
    raw = os.environ.get("SLIME_SCHEDULE")
    return json.loads(raw) if raw else {}


def cap_for(rollout_id, response_len):
    """Cap in force at ``rollout_id``: the entry with the largest id <= rollout_id (None before the first)."""
    steps = sorted((int(k), int(v)) for k, v in (response_len or {}).items())
    cap = None
    for start, value in steps:
        if start <= rollout_id:
            cap = value
    return cap


_last_cap = None


def generate_rollout(args, rollout_id, data_source, evaluation=False):
    global _last_cap
    from slime.rollout import sglang_rollout

    cap = cap_for(rollout_id, schedule().get("response_len"))
    if cap is not None and not evaluation:
        args.rollout_max_response_len = cap
        sglang_rollout.GenerateState(args).sampling_params["max_new_tokens"] = cap
        if cap != _last_cap:
            _last_cap = cap
            try:
                import slime_metrics

                slime_metrics.get().push.remote(
                    {"response_cap": cap, "rollout_id": rollout_id, "task": "rollout", "time": __import__("time").time()}
                )
            except Exception:
                pass
    return sglang_rollout.generate_rollout(args, rollout_id, data_source, evaluation)
