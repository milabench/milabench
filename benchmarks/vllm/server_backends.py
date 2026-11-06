"""Inference server launch helpers (no heavy benchmark imports)."""

from __future__ import annotations

import json
import os
import sys


class InferenceServerError(BaseException):
    pass


def _merge_dicts(base: dict, override: dict) -> dict:
    """Shallow merge for bench config sections (override wins on key clashes)."""
    merged = dict(base)
    merged.update(override)
    return merged


def _atom_server_argv(argv: list) -> list:
    # Run via milabench wrapper so aiter shims apply in the server subprocess.
    here = os.path.dirname(os.path.abspath(__file__))
    entry = os.path.join(here, "atom_server_entry.py")
    return [sys.executable, entry, *argv]


SERVER_BACKENDS = {
    "vllm": lambda argv: ["vllm", "serve", *argv],
    "atom": _atom_server_argv,
}


def multinode_dp_flags(
    backend: str,
    *,
    nnodes: int,
    dp_local: int,
    node_index: int,
    master_ip: str,
    master_port: int = 29550,
    # ATOM's engine sockets take master_port + 100 + 3 * rank: keep the rendezvous port below that range
    base_port: int = 29450,
) -> list[str]:
    """Server flags for multi-node data parallelism, for the node at ``node_index`` (0 = coordinator).

    One server process per node, each owning ``dp_local`` DP ranks; tensor parallel stays inside the node.
    Node 0 serves the HTTP endpoint, the others only run engines (ATOM: engine-only when the DP rank
    is not 0, vLLM: ``--headless``)."""
    size = nnodes * dp_local
    rank = node_index * dp_local
    if backend == "atom":
        return [
            "--data-parallel-size", str(size),
            "--data-parallel-size-local", str(dp_local),
            "--data-parallel-rank", str(rank),
            "--data-parallel-master-ip", master_ip,
            "--data-parallel-master-port", str(master_port),
            "--data-parallel-base-port", str(base_port),
        ]
    if backend == "vllm":
        flags = [
            "--data-parallel-size", str(size),
            "--data-parallel-size-local", str(dp_local),
            "--data-parallel-address", master_ip,
            "--data-parallel-rpc-port", str(master_port),
        ]
        if node_index > 0:
            flags += ["--data-parallel-start-rank", str(rank), "--headless"]
        return flags
    raise InferenceServerError(f"multi-node data parallel is not supported for backend {backend}")


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
    return _merge_dicts(shared, variant)


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
