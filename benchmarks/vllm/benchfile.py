import asyncio
import os
import sys
from copy import deepcopy
from pathlib import Path

from milabench.alt_async import destroy
from milabench.pack import Package
import milabench.commands as cmd
from milabench.commands.executors import get_or_create_warden
from milabench.commands.ray import RayCluster
from milabench.commands.srun import SrunExceptMain, _main_node
from milabench.merge import merge
from milabench.utils import assemble_options, select_nodes

sys.path.insert(0, str(Path(__file__).resolve().parent))
from server_backends import multinode_dp_flags  # noqa: E402


# vLLM-only server flags; ATOM accepts --served-model-name and --trust-remote-code.
_VLLM_ONLY_SERVER_FLAGS = frozenset(
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


def filter_vllm_server_argv_for_atom(argv: list) -> list:
    """Drop vLLM-only server flags before launching ATOM."""
    filtered: list = []
    i = 0
    while i < len(argv):
        arg = str(argv[i])
        if "=" in arg:
            flag = arg.split("=", 1)[0]
            if flag in _VLLM_ONLY_SERVER_FLAGS:
                i += 1
                continue
        elif arg in _VLLM_ONLY_SERVER_FLAGS:
            i += 2 if i + 1 < len(argv) and not str(argv[i + 1]).startswith("-") else 1
            continue
        filtered.append(argv[i])
        i += 1
    return filtered


class VLLMParallel(cmd.Command):
    """This is like a torchrun but it handles the tensor parallel as well"""
    def __init__(self, base_cmd, dataparallel_gpu, tensorparallel_gpu):
        # assert dataparallel_gpu * tensorparallel_gpu <= ngpu

        self.local_world = ngpu / tensorparallel_gpu
        self.world_size = self.local_world * self.num_machine

    def rank():
        os.environ["VLLM_DP_RANK"] = str(global_dp_rank)
        os.environ["VLLM_DP_RANK_LOCAL"] = str(local_dp_rank)
        os.environ["VLLM_DP_SIZE"] = str(self.world_size)

        os.environ["VLLM_DP_MASTER_IP"] = dp_master_ip
        os.environ["VLLM_DP_MASTER_PORT"] = str(dp_master_port)


class DataParallelServers(cmd.SequenceCommand):
    """Multi-node data parallel serving: engine-only servers on the worker nodes, *workload* on the main node.

    The workload (main.py) starts the coordinator server (DP rank 0, HTTP endpoint) and the client; every
    other node runs ``server_worker.py``, which joins the coordinator's DP group. The worker servers never
    exit on their own, and the ssh fallback of ``milabench slurm srun`` does not forward signals, so once
    the workload is done ``server_worker.py --stop`` kills them by pidfile (same idea as RayCluster)."""

    def __init__(self, executor, worker, stop, **kwargs):
        super().__init__(None, **kwargs)
        self.executor = executor
        self.worker = worker
        self.stop = stop

    @property
    def pack(self):
        return self.executor.pack

    @property
    def executors(self):
        return [cmd.ListCommand(self.worker, cmd.SequenceCommand(self.executor, self.stop))]

    async def execute(self, phase="run", timeout=False, timeout_delay=600, warden=None, resource_cleaner=True,
                      with_gpu_warden=True, **kwargs):
        run_kwargs = {**self._kwargs, **kwargs, "phase": phase, "timeout": timeout, "timeout_delay": timeout_delay}
        with get_or_create_warden(warden, with_gpu_warden=with_gpu_warden, resource_cleaner=resource_cleaner) as warden:
            worker_task = asyncio.create_task(self.worker.execute(**run_kwargs, warden=warden))
            workload_task = asyncio.create_task(self.executor.execute(**run_kwargs, warden=warden))
            try:
                done, _ = await asyncio.wait({worker_task, workload_task}, return_when=asyncio.FIRST_COMPLETED)
                if workload_task not in done:
                    # a worker server exited first: the coordinator would wait for its engines until the timeout
                    await self.executor.pack.message("A worker node's server exited: stopping the workload")
                    for proc in self.executor.pack.processes:
                        if proc.poll() is None:
                            destroy(proc)
                error_count = await workload_task
                if workload_task not in done:
                    error_count = max(1, error_count)
            finally:
                await self.stop.execute(**run_kwargs, warden=warden)
                for proc in self.worker.pack.processes:
                    if proc.poll() is None:
                        destroy(proc)
                try:
                    await asyncio.wait_for(worker_task, timeout=60)
                except Exception:
                    worker_task.cancel()
        return error_count

    def set_run_options(self, **kwargs):
        self.executor.set_run_options(**kwargs)
        return self

    def copy(self, pack):
        # the worker / stop steps keep their own (nolog) pack
        copy = deepcopy(self)
        copy.executor._set_pack(pack)
        return copy


class VLLM(Package):
    # Requirements file installed by install(). It can be empty or absent.
    base_requirements = "requirements.in"

    # The preparation script called by prepare(). It must be executable,
    # but it can be any type of script. It can be empty or absent.
    prepare_script = "prepare.py"

    # The main script called by run(). It must be a Python file. It has to
    # be present.
    main_script = f"main.py"

    # You can remove the functions below if you don't need to modify them.

    def make_env(self):
        env = super().make_env()
        env["XDG_CACHE_HOME"] = str(self.dirs.cache)
        env["MILABENCH_TIMELINE_DB"] = str(self.logdir / "benchmark_results.db")

        if self._arch() == "cuda":
            env["FLASHINFER_CACHE_DIR"] = str(self.dirs.cache / "flashinfer")
            # flashinfer >= 0.7 writes <base>/.cache/flashinfer/<version>, base defaults to $HOME
            # (read only on compute nodes)
            env["FLASHINFER_WORKSPACE_BASE"] = str(self.dirs.cache)
            # No FLASHINFER_CUBIN_DIR: it overrides the flashinfer-cubin package with a folder that
            # is empty unless something downloads the cubins into it (not possible offline)

        if self._arch() == "rocm":
            env["VLLM_TARGET_DEVICE"] = "rocm"
            env["VLLM_DISABLE_FLASHINFER"] = "1"
            # Not forcing FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE: vLLM only wants it
            # on RDNA, and on CDNA it switches the installed (CK) flash-attn to its
            # Triton backend, whose autotune keys triton >= 3.8 rejects at import.
            # Export it on RDNA; the server inherits the environment.
            if self.server_backend() == "atom":
                # flydsl (aiter's kernel DSL) serializes through MLIR's ROCDL
                # target, which needs a ROCm tree with amdgcn/bitcode and
                # llvm/bin/ld.lld; the torch rocm-sdk wheels lay theirs out differently
                # (an empty ROCM_PATH, as exported by env.sh, counts as unset)
                env["ROCM_PATH"] = env.get("ROCM_PATH") or os.environ.get("ROCM_PATH") or "/opt/rocm"
                if self.multinode_dp():
                    # aiter's custom all-gather (IPC) is intra-node: it is disabled on a DP group that spans nodes,
                    # and ATOM's DP-sharded lm_head then asserts on it. ATOM's opt-out falls back to RCCL.
                    env["ATOM_USE_CUSTOM_ALL_GATHER"] = "0"
            else:
                # atom registers vLLM platform/model plugins that swap vLLM's
                # own models for ATOM's whenever atom is installed
                env["ATOM_DISABLE_VLLM_PLUGIN"] = "1"

        if "milabench_sharded" in [str(a) for a in self.server_argv()]:
            # benchmate's vLLM loader: per-rank weight shards, written on the first run (node-local)
            env["VLLM_MILABENCH_SHARD_DIR"] = os.path.join(str(self.dirs.data), "vllm-shards", self.config["name"])

        if self.uses_ray():
            # vLLM only forwards VLLM_*, NCCL_*, HF_*, ... to its Ray actors, and
            # `ray start` on the workers (over ssh) does not see this env either:
            # without it, worker nodes load ATOM's model while the head loads vLLM's
            env["VLLM_RAY_EXTRA_ENV_VARS_TO_COPY"] = ",".join(
                name
                for name in ("ATOM_DISABLE_VLLM_PLUGIN", "XDG_CACHE_HOME", "TORCH_HOME", "GLOO_SOCKET_IFNAME")
                if name in env
            )
            # the engine runs on the main node: same address as the Ray head (rayrun sets it for the actors)
            env.setdefault("VLLM_HOST_IP", cmd.node_address(_main_node(self.config)))

        return env

    async def install(self):
        await super().install()

    async def prepare(self):
        await super().prepare()  # super() call executes prepare_script

    @property
    def num_machines(self):
        return max(1, int(self.config.get("num_machines", 1)))

    def _arch(self) -> str:
        return self.config.get("system", {}).get("arch", "cuda")

    def _variant(self) -> dict:
        return (self.config.get("variants") or {}).get(self._arch(), {})

    def _merged_section(self, name: str, *, prepare: bool = False) -> dict:
        shared = dict(self.config.get(name, {}) or {})
        variant = dict(self._variant().get(name, {}) or {})
        argv = merge(shared.get("argv", {}), variant.get("argv", {}))
        if prepare and name == "client" and isinstance(argv, dict):
            argv = merge(argv, {"--num-prompts": 1})
        merged = merge(shared, variant)
        merged["argv"] = argv
        return merged

    def server_backend(self) -> str:
        return self._merged_section("server").get("backend", "vllm")

    def server_command(self) -> list[str] | None:
        command = self._merged_section("server").get("command")
        return list(command) if command else None

    def client_argv(self, prepare=False):
        return assemble_options(self._merged_section("client", prepare=prepare)["argv"])

    def multinode_dp(self) -> bool:
        """``server.multinode: dp``: one server per node joined by data parallelism (ATOM, vLLM), instead of
        one vLLM engine spread over the nodes with Ray."""
        return self.num_machines > 1 and self._merged_section("server").get("multinode") == "dp"

    def dp_local(self) -> int:
        return int(self._merged_section("server").get("dp_local", 1))

    def dp_nodes(self) -> list[str]:
        """Addresses of the nodes of the run, the main node (coordinator) first."""
        main = _main_node(self.config)
        nodes = select_nodes(self.config["system"]["nodes"], self.num_machines)
        others = [n for n in nodes if n is not main and cmd.node_address(n) != cmd.node_address(main)]
        return [cmd.node_address(main)] + [cmd.node_address(n) for n in others]

    def base_server_argv(self, prepare=False):
        args = assemble_options(self._merged_section("server", prepare=prepare)["argv"])
        if self.server_backend() == "atom":
            args = filter_vllm_server_argv_for_atom(args)
        return args

    def server_argv(self, prepare=False):
        args = self.base_server_argv(prepare=prepare)
        if self.multinode_dp():
            nodes = self.dp_nodes()
            return args + multinode_dp_flags(
                self.server_backend(), nnodes=len(nodes), dp_local=self.dp_local(), node_index=0, master_ip=nodes[0]
            )
        if self.server_backend() == "vllm" and self.num_machines > 1 and not any(
            a == "--distributed-executor-backend"
            or str(a).startswith("--distributed-executor-backend=")
            for a in args
        ):
            args.extend(["--distributed-executor-backend", "ray"])
        return args

    @property
    def argv(self):
        return self.server_argv() + ['--'] + self.client_argv()

    def uses_ray(self) -> bool:
        """True for multi-node or when the server explicitly asks for Ray."""
        if self.server_backend() != "vllm" or self.multinode_dp():
            return False
        if self.num_machines > 1:
            return True
        args = self.server_argv()
        for i, a in enumerate(args):
            s = str(a)
            if s.startswith("--distributed-executor-backend="):
                return s.split("=", 1)[1] == "ray"
            if s == "--distributed-executor-backend":
                return i + 1 < len(args) and str(args[i + 1]) == "ray"
        return False

    def build_prepare_plan(self):
        # Prefer prepare-friendly client argv (1 prompt) while still downloading
        # the server model from the positional / server argv.
        prep = self.dirs.code / self.prepare_script
        if self.prepare_script is None or not prep.exists():
            return cmd.VoidCommand(self)
        argv = self.server_argv(prepare=True) + ["--"] + self.client_argv(prepare=True)
        return cmd.PackCommand(
            self, prep, *argv, env=self.make_env(), cwd=prep.parent
        )

    def client_passes(self) -> list:
        return self._merged_section("client").get("passes") or []

    def build_run_plan(self):
        main = self.dirs.code / self.main_script
        target = self
        if self.client_passes():
            # main.py writes every client pass to its own <bench>.<pass>.<job>.data (the metrics to report);
            # what milabench records from the process is the server's (start, load, warmup, idle): nolog
            target = self.copy(cmd.clone_with(self.config, {"tag": [*self.config["tag"], "server", "nolog"]}))
        pack = cmd.PackCommand(target, *self.argv, lazy=True)
        workload = cmd.VoirCommand(pack, cwd=main.parent).use_stdout()
        if self.multinode_dp():
            return self.dp_servers(workload)
        if self.uses_ray():
            return RayCluster(workload)
        return workload

    def dp_servers(self, workload):
        worker_pack = self.copy(cmd.clone_with(self.config, {"tag": [*self.config["tag"], "dpworker", "nolog"]}))
        python = f"{self.dirs.venv}/bin/python"
        script = str(self.dirs.code / "server_worker.py")
        pidfile = f"{self.dirs.runs}/server-worker.{self.config['name']}.pid"
        env = self.make_env()
        worker = cmd.CmdCommand(
            worker_pack, python, script,
            "--backend", self.server_backend(),
            "--nodes", ",".join(self.dp_nodes()),
            "--dp-local", str(self.dp_local()),
            "--pidfile", pidfile,
            "--", *self.base_server_argv(),
            env=env, cwd=str(self.dirs.code),
        )
        stop = cmd.CmdCommand(
            worker_pack, python, script, "--stop", "--pidfile", pidfile, env=env, cwd=str(self.dirs.code),
        )
        return DataParallelServers(workload, SrunExceptMain(worker), SrunExceptMain(stop))


__pack__ = VLLM
