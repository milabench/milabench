import json
import os


def _get_flag(name, type, default):
    return type(os.getenv(name, default))


def get_poll_interval(value):
    return _get_flag("BENCHMATE_POLL_INTERVAL", float, value)


def get_observation_count(value):
    return _get_flag("BENCHMATE_OBSERVATION_COUNT", int, value)


def torchmem_enabled():
    """Whether to poll PyTorch CUDA/ROCm allocator stats.

    Disable with ``BENCHMATE_TORCHMEM=0`` (e.g. for Jax benches).
    """
    return _get_flag("BENCHMATE_TORCHMEM", int, 1) != 0


def jaxmem_enabled():
    """Whether to poll JAX device allocator stats.

    Opt-in with ``BENCHMATE_JAXMEM=1`` (e.g. for Jax benches).
    """
    return _get_flag("BENCHMATE_JAXMEM", int, 0) != 0


def vllm_request_metrics_enabled(default: bool = True) -> bool:
    """Whether vLLM emits per-request latency/token metrics on stdout (.data).

    When disabled, per-request rows are still written to the local timeline
    SQLite database; only stdout events (``ttfts``/``e2els``/``request_rate``/
    etc.) are omitted so published archives stay small. Bucket-level timeline
    metrics are always pushed.

    Disable with ``BENCHMATE_VLLM_REQUEST_METRICS=0`` or
    ``push_request_metrics: false`` in the benchmark YAML config.
    """
    if "BENCHMATE_VLLM_REQUEST_METRICS" in os.environ:
        return os.environ["BENCHMATE_VLLM_REQUEST_METRICS"] not in (
            "0",
            "false",
            "False",
        )

    raw = os.getenv("MILABENCH_CONFIG")
    if raw:
        try:
            cfg = json.loads(raw)
        except json.JSONDecodeError:
            cfg = None
        if isinstance(cfg, dict) and "push_request_metrics" in cfg:
            return bool(cfg["push_request_metrics"])

    return default


poll_interval_default = get_poll_interval(0.25)


log_pattern = _get_flag("BENCHMATE_LOG_MODE", str, 'lean')
