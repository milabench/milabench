"""Inference servers (OpenAI-compatible API) run by milabench next to a client workload.

Each backend is its own command; ``InferenceServer`` picks one from the configured backend, and
``deploy()`` lays it out on the run's nodes::

    server = InferenceServer(server_pack, *server_argv, backend="atom")
    service, stop = server.deploy(multinode="dp")
    Service(service, SequenceCommand(eval_client, perf_client), ready=server.ready, stop=stop)

``Service`` waits for ``ready()`` before starting the clients and stops the server once they are done.
Everything backend specific (launcher, flags, environment, multi-node layout) stays in the server command.
"""

from __future__ import annotations

import asyncio
import copy
import os
import urllib.request
from typing import List

from ..utils import select_nodes
from . import ListCommand, SingleCmdCommand, CmdCommand, clone_with, count_command_errors, max_node_count, node_address
from .executors import get_or_create_warden


class FirstExit(ListCommand):
    """Run the commands together; done as soon as one of them exits.

    A multi-node server is down once any node's server is: when a worker node's engine dies, the coordinator
    stays up but can no longer answer, and a plain ``ListCommand`` would wait for it forever (the clients with
    it). ``Service`` stops the workload when this returns and kills what is left of the service."""

    async def execute(
        self, phase="run", timeout=False, timeout_delay=600,
        warden=None, resource_cleaner=True, with_gpu_warden=True, **kwargs
    ):
        run_kwargs = {**self._kwargs, **kwargs, "phase": phase, "timeout": timeout, "timeout_delay": timeout_delay}
        with get_or_create_warden(warden, with_gpu_warden=with_gpu_warden, resource_cleaner=resource_cleaner) as warden:
            tasks = [asyncio.create_task(executor.execute(**run_kwargs, warden=warden)) for executor in self.executors]
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        # still running: their processes are killed by the caller (Service); keep the tasks referenced until then
        self._pending = pending
        return sum(count_command_errors(task.result() if not task.exception() else task.exception()) for task in done)


class ServerCommand(SingleCmdCommand):
    """Base inference server: ``launcher() + filter_argv(server_argv)``, run from the pack's venv.

    Arguments:
        pack: pack the server runs on (tag it ``nolog``: it is stopped by a signal)
        *server_argv: backend arguments (model, parallelism, ...)
        host/port: where the clients reach the server, for ``ready()``
    """

    name = None

    def __init__(self, pack, *server_argv, host="127.0.0.1", port=8000, **kwargs) -> None:
        # setsid: the server spawns engine/worker processes, they are stopped as a group
        super().__init__(pack, **{"setsid": True, **kwargs})
        self.server_argv = list(server_argv)
        self.host = host
        self.port = port
        self.nnodes = 1
        self.ray_head = None

    @property
    def arch(self) -> str:
        return self.pack.config.get("system", {}).get("arch", "cuda")

    async def execute(self, *args, **kwargs):
        evict_page_cache()
        return await super().execute(*args, **kwargs)

    @property
    def python(self) -> str:
        return f"{self.pack.dirs.venv}/bin/python"

    def launcher(self) -> List:
        raise NotImplementedError()

    def filter_argv(self, argv) -> List:
        """Drop the arguments this backend does not accept"""
        return argv

    def env(self) -> dict:
        """Backend specific environment; the benchmark's ``env`` config takes precedence"""
        env = {}
        if self.arch == "cuda":
            cache = self.pack.dirs.cache
            env["FLASHINFER_CACHE_DIR"] = str(cache / "flashinfer")
            env["FLASHINFER_CUBIN_DIR"] = str(cache / "flashinfer" / "cubins")
        return env

    def dp_flags(self, nnodes, dp_local, master_ip, rank=0, master_port=29550) -> List:
        """Flags of the server owning DP ranks ``rank .. rank + dp_local`` of a multi-node data parallel group.

        ``rank == 0`` is the coordinator, which serves the HTTP endpoint; the others only run engines.
        """
        raise NotImplementedError(f"multi-node data parallel is not supported by {self.name}")

    def data_parallel(self, nnodes, dp_local, master_ip, rank=0) -> ServerCommand:
        """Copy of this server joining a multi-node data parallel group"""
        server = copy.copy(self)
        server._kwargs = dict(self._kwargs)
        server.server_argv = [*self.server_argv, *self.dp_flags(nnodes, dp_local, master_ip, rank=rank)]
        server.nnodes = nnodes
        return server

    def nodes(self) -> List[str]:
        """Addresses of the run's nodes, the main node first"""
        from .srun import _main_node

        config = self.pack.config
        main = _main_node(config)
        nodes = select_nodes(config["system"]["nodes"], max_node_count(config))
        others = [n for n in nodes if n is not main and node_address(n) != node_address(main)]
        return [node_address(main)] + [node_address(n) for n in others]

    def deploy(self, multinode=None, dp_local=1):
        """``(service, stop)`` running this server on the run's nodes (``stop`` may be None).

        ``multinode="dp"``: one server per node, joined by data parallelism (see ``data_parallel_group``).
        Backends can support other layouts (vLLM: one engine over the nodes with Ray).
        """
        nodes = self.nodes()
        if len(nodes) <= 1:
            return self, None
        if multinode == "dp":
            return self.data_parallel_group(nodes, dp_local)
        raise NotImplementedError(f"{self.name}: no multi-node layout {multinode!r}")

    def data_parallel_group(self, nodes, dp_local=1):
        """Coordinator on the main node, ``benchmate.server_worker`` running an engine-only server on the others.

        The worker servers never exit on their own, and the ssh fallback of ``milabench slurm srun`` does not
        forward signals: the returned stop command kills them by pidfile."""
        from .srun import SrunExceptMain

        coordinator = self.data_parallel(len(nodes), dp_local, nodes[0])
        # server_worker fills in the node's rank
        engine = self.data_parallel(len(nodes), dp_local, nodes[0], rank="@DP_RANK@")

        config = self.pack.config
        pack = self.pack.copy(clone_with(config, {"tag": [*config["tag"], "dpworker"]}))
        pidfile = f"{self.pack.dirs.runs}/server-worker.{config['name']}.pid"
        options = engine.options
        kwargs = {"env": options["env"], "cwd": options.get("cwd")}
        worker = CmdCommand(
            pack, self.python, "-m", "benchmate.server_worker",
            "--nodes", ",".join(nodes),
            "--dp-local", str(dp_local),
            "--pidfile", pidfile,
            "--", *engine.argv(),
            **kwargs,
        )
        stop = CmdCommand(pack, self.python, "-m", "benchmate.server_worker", "--stop", "--pidfile", pidfile, **kwargs)
        return FirstExit(coordinator, SrunExceptMain(worker)), SrunExceptMain(stop)

    def _argv(self, **kwargs) -> List:
        return [*self.launcher(), *cap_memory_fraction(self.filter_argv(self.server_argv))]

    @property
    def options(self):
        options = super().options
        return {
            **options,
            "env": {**self.env(), **options.get("env", {}), **self.pack.config.get("env", {})},
        }

    def ready_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1/models"

    async def ready(self) -> bool:
        return await asyncio.to_thread(_http_ok, self.ready_url())


def evict_page_cache(paths: str | None = None) -> int:
    """Drop the page cache of the files under ``MILABENCH_EVICT_PAGE_CACHE`` (``:``-separated folders),
    before a server starts; returns the bytes advised away.

    On unified memory the page cache of the previous models' weights (tens of GB) is memory CUDA reports as
    used: vLLM refuses to start when its ``--gpu-memory-utilization`` share is above the free memory, and
    every model a run loads adds its weights to the cache. Dropping the whole cache needs root
    (``/proc/sys/vm/drop_caches``); advising the kernel the files' pages are not needed
    (``posix_fadvise(DONTNEED)``) does not, and only the model caches matter."""
    paths = paths if paths is not None else os.environ.get("MILABENCH_EVICT_PAGE_CACHE", "")
    advised = 0
    for root in filter(None, paths.split(":")):
        for folder, _, files in os.walk(root):
            for name in files:
                path = os.path.join(folder, name)
                try:
                    if os.path.islink(path):
                        continue
                    fd = os.open(path, os.O_RDONLY)
                except OSError:
                    continue
                try:
                    advised += os.fstat(fd).st_size
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                except OSError:
                    pass
                finally:
                    os.close(fd)
    return advised


# share of the device memory a server may reserve up front (vLLM, SGLang)
MEMORY_FRACTION_FLAGS = ("--gpu-memory-utilization", "--mem-fraction-static")


def cap_memory_fraction(argv: List) -> List:
    """Clamp the servers' memory fraction to ``MILABENCH_GPU_MEMORY_FRACTION_MAX``, a property of the machine.

    On unified memory (DGX Spark, APUs) the "GPU memory" is the system RAM: a server reserving 90% of it
    leaves too little for the OS, the client and its containers, and the machine swaps until it freezes
    instead of failing with an out-of-memory error. The benchmark configs stay hardware neutral; the machine
    sets the ceiling (e.g. 0.75 on a Spark)."""
    cap = os.environ.get("MILABENCH_GPU_MEMORY_FRACTION_MAX")
    if not cap:
        return list(argv)
    cap = float(cap)
    out = []
    for i, arg in enumerate(argv):
        arg = str(arg)
        prev = str(argv[i - 1]) if i else ""
        if prev in MEMORY_FRACTION_FLAGS:
            out.append(str(min(float(arg), cap)))
            continue
        name, sep, value = arg.partition("=")
        if sep and name in MEMORY_FRACTION_FLAGS:
            out.append(f"{name}={min(float(value), cap)}")
            continue
        out.append(argv[i])
    return out


def _flag(argv, name, default=None):
    """Last value of ``name`` in argv (``name value`` or ``name=value``)"""
    for i, arg in enumerate(argv):
        arg = str(arg)
        if arg == name and i + 1 < len(argv):
            default = argv[i + 1]
        elif arg.startswith(name + "="):
            default = arg.split("=", 1)[1]
    return default


def _http_ok(url) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            return response.status == 200
    except OSError:
        return False


class VLLMServer(ServerCommand):
    name = "vllm"

    def launcher(self):
        return [f"{self.pack.dirs.venv}/bin/vllm", "serve"]

    def env(self):
        env = super().env()
        if self.arch == "rocm":
            env["VLLM_TARGET_DEVICE"] = "rocm"
            env["VLLM_DISABLE_FLASHINFER"] = "1"
        # atom registers vLLM platform/model plugins that swap vLLM's own models for ATOM's whenever atom is installed
        env["ATOM_DISABLE_VLLM_PLUGIN"] = "1"
        if "milabench_sharded" in [str(a) for a in self.server_argv]:
            # benchmate's vLLM loader: per-rank weight shards, written on the first run (node-local)
            env["VLLM_MILABENCH_SHARD_DIR"] = os.path.join(
                str(self.pack.dirs.data), "vllm-shards", self.pack.config["name"]
            )
        if self.ray_head:
            # vLLM only forwards VLLM_*, NCCL_*, HF_*, ... to its Ray actors, and `ray start` on the workers
            # (over ssh) does not see this env either: without it, worker nodes load ATOM's model while the
            # head loads vLLM's
            known = {**self.pack.make_env(), **env, **self.pack.config.get("env", {})}
            env["VLLM_RAY_EXTRA_ENV_VARS_TO_COPY"] = ",".join(
                name
                for name in ("ATOM_DISABLE_VLLM_PLUGIN", "XDG_CACHE_HOME", "TORCH_HOME", "GLOO_SOCKET_IFNAME")
                if name in known
            )
            # the engine runs on the main node: same address as the Ray head (rayrun sets it for the actors)
            env["VLLM_HOST_IP"] = self.ray_head
        return env

    def uses_ray(self) -> bool:
        return _flag(self.server_argv, "--distributed-executor-backend") == "ray"

    def deploy(self, multinode=None, dp_local=1):
        """Without ``multinode="dp"``: one engine over every node with Ray (TP inside the nodes, PP across)"""
        nodes = self.nodes()
        if multinode == "dp" or (len(nodes) <= 1 and not self.uses_ray()):
            return super().deploy(multinode, dp_local)

        from .ray import RayCluster

        server = copy.copy(self)
        server._kwargs = dict(self._kwargs)
        if not self.uses_ray():
            server.server_argv = [*self.server_argv, "--distributed-executor-backend", "ray"]
        server.ray_head = nodes[0]
        cluster = RayCluster(server)
        # `ray stop` is the last step of RayCluster's sequence, which a stopped server cuts short
        return cluster, cluster.stop_executor()

    def dp_flags(self, nnodes, dp_local, master_ip, rank=0, master_port=29550):
        flags = [
            "--data-parallel-size", str(nnodes * dp_local),
            "--data-parallel-size-local", str(dp_local),
            "--data-parallel-address", master_ip,
            "--data-parallel-rpc-port", str(master_port),
        ]
        if rank != 0:
            flags += ["--data-parallel-start-rank", str(rank), "--headless"]
        return flags


class AtomServer(ServerCommand):
    """ROCm's ATOM. ``entry`` replaces ATOM's own entry point (e.g. a script applying compatibility shims)."""

    name = "atom"

    # vLLM flags ATOM rejects, so one server config can be shared by both backends
    VLLM_ONLY_FLAGS = frozenset(
        {
            "--dtype",
            "--tensor-parallel-size",
            "--distributed-executor-backend",
            "--tokenizer-mode",
            "--config_format",
            "--load_format",
            "--tool-call-parser",
            "--enable-auto-tool-choice",
            "--kv-cache-dtype",
        }
    )

    def __init__(self, pack, *server_argv, entry=None, **kwargs) -> None:
        super().__init__(pack, *server_argv, **kwargs)
        self.entry = entry

    def launcher(self):
        if self.entry:
            return [self.python, str(self.entry)]
        return [self.python, "-m", "atom.entrypoints.openai.api_server"]

    def filter_argv(self, argv):
        filtered = []
        i = 0
        while i < len(argv):
            arg = str(argv[i])
            if "=" in arg:
                if arg.split("=", 1)[0] in self.VLLM_ONLY_FLAGS:
                    i += 1
                    continue
            elif arg in self.VLLM_ONLY_FLAGS:
                i += 2 if i + 1 < len(argv) and not str(argv[i + 1]).startswith("-") else 1
                continue
            filtered.append(argv[i])
            i += 1
        return filtered

    def env(self):
        env = super().env()
        # flydsl (aiter's kernel DSL) serializes through MLIR's ROCDL target, which needs a ROCm tree with
        # amdgcn/bitcode and llvm/bin/ld.lld; the torch rocm-sdk wheels lay theirs out differently
        # (an empty ROCM_PATH, as exported by env.sh, counts as unset)
        env["ROCM_PATH"] = os.environ.get("ROCM_PATH") or "/opt/rocm"
        if self.nnodes > 1:
            # aiter's custom all-gather (IPC) is intra-node: it is disabled on a DP group that spans nodes,
            # and ATOM's DP-sharded lm_head then asserts on it. ATOM's opt-out falls back to RCCL.
            env["ATOM_USE_CUSTOM_ALL_GATHER"] = "0"
        return env

    def dp_flags(self, nnodes, dp_local, master_ip, rank=0, master_port=29550,
                 # ATOM's engine sockets take master_port + 100 + 3 * rank: keep the rendezvous port below that range
                 base_port=29450):
        # engine-only when the DP rank is not 0
        return [
            "--data-parallel-size", str(nnodes * dp_local),
            "--data-parallel-size-local", str(dp_local),
            "--data-parallel-rank", str(rank),
            "--data-parallel-master-ip", master_ip,
            "--data-parallel-master-port", str(master_port),
            "--data-parallel-base-port", str(base_port),
        ]


class OmniServer(VLLMServer):
    """vLLM-Omni: ``vllm serve <model> --omni``, diffusion (image / video / audio) and TTS models behind
    OpenAI-style endpoints (``/v1/images/generations``, ``/v1/videos``, ``/v1/audio/speech``)"""

    name = "vllm-omni"

    def _argv(self, **kwargs) -> List:
        argv = cap_memory_fraction(self.filter_argv(self.server_argv))
        return [*self.launcher(), *argv, *([] if "--omni" in argv else ["--omni"])]


class SGLangServer(ServerCommand):
    name = "sglang"

    def launcher(self):
        return [self.python, "-m", "sglang.launch_server"]


class TRTLLMServer(ServerCommand):
    """NVIDIA's TensorRT-LLM (``trtllm-serve``), the CUDA counterpart of ATOM"""

    name = "trtllm"

    def launcher(self):
        return [f"{self.pack.dirs.venv}/bin/trtllm-serve"]


class RawServer(ServerCommand):
    """Server launched by an explicit command line (``server.command`` in the benchmark config)"""

    name = "command"

    def __init__(self, pack, command, *server_argv, **kwargs) -> None:
        super().__init__(pack, *server_argv, **kwargs)
        self.command = list(command)

    def launcher(self):
        return self.command


SERVER_COMMANDS = {cls.name: cls for cls in (VLLMServer, OmniServer, AtomServer, SGLangServer, TRTLLMServer)}


def InferenceServer(pack, *server_argv, backend="vllm", command=None, **kwargs) -> ServerCommand:
    """Server command of *backend*, or of an explicit *command* line"""
    if command:
        kwargs.pop("entry", None)
        return RawServer(pack, command, *server_argv, **kwargs)
    try:
        cls = SERVER_COMMANDS[backend]
    except KeyError as exc:
        raise ValueError(f"Unknown inference server backend: {backend} (known: {sorted(SERVER_COMMANDS)})") from exc
    if cls is not AtomServer:
        kwargs.pop("entry", None)
    return cls(pack, *server_argv, **kwargs)
