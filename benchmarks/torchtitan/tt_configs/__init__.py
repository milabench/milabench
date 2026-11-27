"""Config functions for ``--module tt_configs --config <name>`` (see main.py).

HF transformers backend benches come from config_registry. ``qwen38`` (native
TorchTitan model) is imported on first access only: it pulls attn-gym's CUDA
kernels, which the HF benches must not need.
"""

from .config_registry import *  # noqa: F401,F403


def __getattr__(name):
    if name == "qwen38":
        from tt_native.configs import qwen38

        return qwen38
    raise AttributeError(name)
