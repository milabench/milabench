"""Pinned third-party evaluation tools, each in its own venv (their dependencies clash with vLLM's).

``$MILABENCH_DIR_DATA/benchserve-tools/<tool>``; installed by a pack's prepare.py (``ensure``), used by the
client passes (``executable``), which run offline.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

TOOLS = {
    # quality suite of a served model (launch80): pinned items, greedy decoding, paired comparisons
    "qualitybench": {"spec": "qualitybench @ git+https://github.com/launch80/QualityBench@v0.3.0",
                     "bin": "qualitybench"},
    # coding agent (mini-swe-agent) and the SWE-bench evaluation harness
    "mini-swe-agent": {"spec": "mini-swe-agent==2.4.6", "bin": "mini-extra"},
    "swebench": {"spec": "swebench==5.0.2", "bin": "swebench"},
}

PYTHON = "3.12"


def root() -> str:
    base = os.environ.get("MILABENCH_DIR_DATA") or os.path.join(os.path.expanduser("~"), ".cache", "milabench")
    return os.path.join(base, "benchserve-tools")


def venv(name: str) -> str:
    return os.path.join(root(), name)


def executable(name: str, program: str | None = None) -> str:
    path = os.path.join(venv(name), "bin", program or TOOLS[name]["bin"])
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} is not installed ({path}): run the benchmark's prepare first")
    return path


def python(name: str) -> str:
    return os.path.join(venv(name), "bin", "python")


def _uv() -> list[str]:
    if uv := shutil.which("uv"):
        return [uv]
    return [sys.executable, "-m", "uv"]


def ensure(name: str) -> str:
    """Install the tool's pinned version once; returns its executable."""
    tool = TOOLS[name]
    path = os.path.join(venv(name), "bin", tool["bin"])
    marker = os.path.join(venv(name), ".benchserve-spec")
    if os.path.exists(path) and os.path.exists(marker) and open(marker).read() == tool["spec"]:
        return path
    os.makedirs(root(), exist_ok=True)
    subprocess.run([*_uv(), "venv", "--allow-existing", "-p", PYTHON, venv(name)], check=True)
    subprocess.run([*_uv(), "pip", "install", "--python", python(name), tool["spec"]], check=True)
    with open(marker, "w") as f:
        f.write(tool["spec"])
    return path


# --- container runtime: docker or podman ------------------------------------------------------------
#
# The code-running tasks (SWE-bench sessions and grading, QualityBench's code tasks) need one on the
# client's machine. MILABENCH_CONTAINER=docker|podman picks it; by default the first one that answers,
# podman first. Each tool is pointed at it:
#   mini-swe-agent    MSWEA_DOCKER_EXECUTABLE (the CLI it runs)
#   swebench harness  DOCKER_HOST (docker's Python SDK; podman serves the same API on its socket)
#   QualityBench      QUALITYBENCH_CONTAINER


def _answers(cli: str) -> bool:
    return bool(shutil.which(cli)) and subprocess.run([cli, "info"], capture_output=True).returncode == 0


def container_runtime() -> str | None:
    """``docker`` or ``podman``, or None when neither runs here."""
    if (choice := os.environ.get("MILABENCH_CONTAINER")) in ("docker", "podman"):
        return choice if _answers(choice) else None
    return next((cli for cli in ("podman", "docker") if _answers(cli)), None)


def docker_available() -> bool:
    return container_runtime() is not None


def podman_socket() -> str:
    out = subprocess.run(["podman", "info", "--format", "{{.Host.RemoteSocket.Path}}"],
                         capture_output=True, text=True, check=True).stdout.strip()
    return out.removeprefix("unix://")


def _socket_answers(path: str) -> bool:
    import socket

    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(5)
            s.connect(path)
            s.sendall(b"GET /_ping HTTP/1.0\r\n\r\n")
            return b"OK" in s.recv(1024)
    except OSError:
        return False


def container_env(runtime: str | None = None) -> dict:
    """Environment pointing every tool at the runtime."""
    runtime = runtime or container_runtime()
    if runtime == "podman":
        return {"MSWEA_DOCKER_EXECUTABLE": "podman", "QUALITYBENCH_CONTAINER": "podman",
                "DOCKER_HOST": f"unix://{podman_socket()}"}
    if runtime == "docker":
        return {"MSWEA_DOCKER_EXECUTABLE": "docker", "QUALITYBENCH_CONTAINER": "docker"}
    return {}


class container_api:
    """Context: podman's docker-compatible API socket served while in it (``podman system service``), when
    nothing serves it already (``systemctl --user enable --now podman.socket`` makes this a no-op).

    ``.env``: the DOCKER_HOST to give the tools talking to the API (docker's Python SDK)."""

    def __init__(self, runtime: str | None = None):
        self.runtime = runtime or container_runtime()
        self.proc = None
        self.env = {}

    def _serve(self, path: str) -> bool:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        except OSError:
            return False
        print(f"[container] serving podman's API on {path} for this pass", flush=True)
        self.proc = subprocess.Popen(["podman", "system", "service", "--time=0", f"unix://{path}"])
        import time

        for _ in range(50):
            if _socket_answers(path):
                return True
            if self.proc.poll() is not None:
                break
            time.sleep(0.2)
        self._stop()
        return False

    def __enter__(self):
        if self.runtime != "podman":
            return self
        path = podman_socket()
        if _socket_answers(path):
            self.env = {"DOCKER_HOST": f"unix://{path}"}
            return self
        import tempfile

        # podman's own socket path, else one of ours (its runtime folder may not exist outside a login session)
        for candidate in (path, os.path.join(tempfile.gettempdir(), f"milabench-podman-{os.getuid()}.sock")):
            if self._serve(candidate):
                self.env = {"DOCKER_HOST": f"unix://{candidate}"}
                return self
        raise RuntimeError("could not serve podman's API (podman system service)")

    def _stop(self):
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.proc = None

    def __exit__(self, *exc):
        self._stop()
        return False
