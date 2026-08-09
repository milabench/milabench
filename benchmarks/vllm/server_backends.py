"""Inference server launch helpers (no heavy benchmark imports)."""

from __future__ import annotations

import json
import os
import sys

from milabench.merge import merge


class InferenceServerError(BaseException):
    pass


def _atom_server_argv(argv: list) -> list:
    # Run via milabench wrapper so aiter shims apply in the server subprocess.
    here = os.path.dirname(os.path.abspath(__file__))
    entry = os.path.join(here, "atom_server_entry.py")
    return [sys.executable, entry, *argv]


SERVER_BACKENDS = {
    "vllm": lambda argv: ["vllm", "serve", *argv],
    "atom": _atom_server_argv,
}


def _milabench_config() -> dict:
    raw = os.environ.get("MILABENCH_CONFIG")
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def _merged_server_section(cfg: dict) -> dict:
    arch = cfg.get("system", {}).get("arch", "cuda")
    shared = dict(cfg.get("server", {}) or {})
    variant = dict((cfg.get("variants") or {}).get(arch, {}).get("server", {}) or {})
    return merge(shared, variant)


def resolved_server_backend(cfg: dict | None = None) -> str:
    cfg = _milabench_config() if cfg is None else cfg
    return _merged_server_section(cfg).get("backend", "vllm")


def resolved_server_command(cfg: dict | None = None) -> list[str] | None:
    cfg = _milabench_config() if cfg is None else cfg
    command = _merged_server_section(cfg).get("command")
    return list(command) if command else None


def build_server_command(
    argv,
    *,
    backend: str | None = None,
    command: list[str] | None = None,
    cfg: dict | None = None,
) -> list[str]:
    backend = backend or resolved_server_backend(cfg)
    command = command if command is not None else resolved_server_command(cfg)
    if command:
        return [*command, *argv]
    try:
        return SERVER_BACKENDS[backend](argv)
    except KeyError as exc:
        raise InferenceServerError(f"Unknown inference server backend: {backend}") from exc
