from milabench.pack import Package
import milabench.commands as cmd
from milabench.commands.serving import InferenceServer
from milabench.merge import merge
from milabench.utils import assemble_options


def _flag(argv, name, default=None):
    """Last value of ``name`` in argv (``name value`` or ``name=value``)"""
    for i, arg in enumerate(argv):
        arg = str(arg)
        if arg == name and i + 1 < len(argv):
            default = argv[i + 1]
        elif arg.startswith(name + "="):
            default = arg.split("=", 1)[1]
    return default


class OmniPlan(cmd.Service):
    """The vLLM-Omni server and its client. NJobs / PerGPU copy the plan onto each job's pack: rebuild it."""

    def __init__(self, bench, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bench = bench

    @property
    def pack(self):
        return self.bench

    def copy(self, pack):
        return pack.build_run_plan()


class Omni(Package):
    """Diffusion serving with vLLM-Omni (image, video): milabench runs ``vllm serve <model> --omni``
    (milabench.commands.serving, backend vllm-omni), main.py is the client.

    Config: ``server.argv`` (model, parallelism: --tensor-parallel-size / --usp, --cache-backend, ...) and
    ``client.argv`` (--task, resolution, steps, guidance, concurrency), like benchmarks/serving."""

    base_requirements = "requirements.in"
    prepare_script = "prepare.py"
    main_script = "main.py"

    def make_env(self):
        env = super().make_env()
        env["XDG_CACHE_HOME"] = str(self.dirs.cache)
        return env

    def _section(self, name):
        arch = self.config.get("system", {}).get("arch", "cuda")
        shared = dict(self.config.get(name, {}) or {})
        variant = dict(((self.config.get("variants") or {}).get(arch, {}) or {}).get(name, {}) or {})
        merged = merge(shared, variant)
        merged["argv"] = merge(shared.get("argv", {}), variant.get("argv", {}))
        return merged

    def server_argv(self):
        return assemble_options(self._section("server")["argv"])

    def client_argv(self):
        return assemble_options(self._section("client")["argv"])

    @property
    def argv(self):
        return self.client_argv()

    def build_prepare_plan(self):
        prep = self.dirs.code / self.prepare_script
        return cmd.PackCommand(self, prep, *self.server_argv(), env=self.make_env(), cwd=prep.parent)

    def build_run_plan(self):
        client_argv = self.client_argv()
        server = InferenceServer(
            # stopped by a signal once the client is done: its exit code is not a result
            self.copy(cmd.clone_with(self.config, {"tag": [*self.config["tag"], "server", "nolog"]})),
            *self.server_argv(),
            backend=self._section("server").get("backend", "vllm-omni"),
            command=self._section("server").get("command"),
            host=_flag(client_argv, "--host", "127.0.0.1"),
            port=_flag(client_argv, "--port", 8000),
            cwd=str(self.dirs.code),
        )
        main = self.dirs.code / self.main_script
        client = cmd.VoirCommand(cmd.PackCommand(self, *client_argv), cwd=str(main.parent)).use_stdout()
        service, stop = server.deploy()
        ready_timeout = int(_flag(client_argv, "--ready-check-timeout-sec", 3600))
        return OmniPlan(self, service, client, ready=server.ready, stop=stop, ready_timeout=ready_timeout)


__pack__ = Omni
