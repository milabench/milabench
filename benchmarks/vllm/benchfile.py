import os

from milabench.pack import Package
import milabench.commands as cmd
from milabench.commands.ray import RayCluster
from milabench.merge import merge
from milabench.utils import assemble_options


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

        env["FLASHINFER_CACHE_DIR "] = str(self.dirs.cache / "flashinfer")
        env["FLASHINFER_CUBIN_DIR "] = str(self.dirs.cache / "flashinfer" / "cubins")

        # flashinfer defaults its workspace to ~/.cache/flashinfer/<version>
        # which may not be writable on compute nodes; redirect to our cache dir
        env["FLASHINFER_WORKSPACE_DIR"] = os.path.join(
            str(self.dirs.cache), "flashinfer", "workspace"
        )
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

    def server_argv(self, prepare=False):
        args = assemble_options(self._merged_section("server", prepare=prepare)["argv"])
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
        if self.server_backend() != "vllm":
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

    def build_run_plan(self):
        main = self.dirs.code / self.main_script
        pack = cmd.PackCommand(self, *self.argv, lazy=True)
        workload = cmd.VoirCommand(pack, cwd=main.parent).use_stdout()
        if self.uses_ray():
            return RayCluster(workload)
        return workload


__pack__ = VLLM
