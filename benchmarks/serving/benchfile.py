import asyncio

from milabench.pack import Package
import milabench.commands as cmd
from milabench.commands.executors import get_or_create_warden
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


class PassSequence(cmd.SequenceCommand):
    """The client passes, one after the other, against the same server.

    Unlike ``SequenceCommand`` a failed pass (e.g. out of memory at the longest context) does not skip the
    following ones: each pass is its own result. If the server dies, ``Service`` stops the workload."""

    async def execute(self, warden=None, resource_cleaner=True, with_gpu_warden=True, **kwargs):
        error_count = 0

        def on_message(msg):
            nonlocal error_count
            if msg.event == "error":
                error_count += 1
            if msg.event == "end":
                error_count += int(msg.data.get("return_code", 0) != 0)

        loop = asyncio.get_running_loop()
        loop._callbacks.append(on_message)
        try:
            with get_or_create_warden(warden, with_gpu_warden=with_gpu_warden, resource_cleaner=resource_cleaner) as warden:
                for executor in self.executors:
                    await executor.execute(**{**self._kwargs, **kwargs}, warden=warden)
        finally:
            loop._callbacks.remove(on_message)
        return error_count


class InferencePlan(cmd.Service):
    """The inference server and its clients.

    NJobs / PerGPU copy the plan onto each job's pack: rebuild the whole plan
    (server, clients, worker servers) from that pack."""

    def __init__(self, bench, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bench = bench

    @property
    def pack(self):
        # the benchmark's own pack, not one of its steps' (tagged server, eval, perf)
        return self.bench

    def copy(self, pack):
        return pack.build_run_plan()


class Serving(Package):
    """Serving benchmark: milabench runs the inference server (milabench.commands.serving),
    main.py is only the client (vLLM's serving benchmark, lm-eval passes).

    Everything backend specific (launcher, flags, environment, multi-node layout) lives in the server
    commands; this file only reads the config. Meant to replace benchmarks/vllm."""

    base_requirements = "requirements.in"
    prepare_script = "prepare.py"
    main_script = "main.py"

    def make_env(self):
        env = super().make_env()
        env["XDG_CACHE_HOME"] = str(self.dirs.cache)
        env["MILABENCH_TIMELINE_DB"] = str(self.logdir / "benchmark_results.db")
        return env

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

    def server_argv(self, prepare=False):
        return assemble_options(self._merged_section("server", prepare=prepare)["argv"])

    def client_argv(self, prepare=False):
        return assemble_options(self._merged_section("client", prepare=prepare)["argv"])

    def client_passes(self) -> list[dict]:
        # same lookup as main.py: the arch variant's passes replace the shared ones
        variant = self._variant().get("client") or {}
        return variant.get("passes") or (self.config.get("client") or {}).get("passes") or []

    @property
    def argv(self):
        return self.client_argv()

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

    def _tagged(self, *tags):
        return self.copy(cmd.clone_with(self.config, {"tag": [*self.config["tag"], *tags]}))

    def inference_server(self):
        server = self._merged_section("server")
        client_argv = self.client_argv()
        return InferenceServer(
            # stopped by a signal once the clients are done: its exit code is not a result
            self._tagged("server", "nolog"),
            *self.server_argv(),
            backend=server.get("backend", "vllm"),
            command=server.get("command"),
            entry=self.dirs.code / "atom_server_entry.py",
            host=_flag(client_argv, "--host", "127.0.0.1"),
            port=_flag(client_argv, "--port", 8000),
            cwd=str(self.dirs.code),
        )

    def client(self, *args, pack=None):
        pack = pack or self
        main = self.dirs.code / self.main_script
        return cmd.VoirCommand(
            cmd.PackCommand(pack, *args, *self.client_argv()), cwd=str(main.parent)
        ).use_stdout()

    def pass_pack(self, spec):
        """A client pass as its own benchmark: ``<name>.<pass>``, tagged with the pass name.

        Its own voir monitor and .data file (``<name>.<pass>.<job>.data``), so ``milabench report`` lists every
        load point and its power on its own row. lm-eval reports scores, not a rate: nolog, or the report
        counts it as a failed run."""
        name = spec["name"]
        extra = ["nolog"] if "eval" in spec else []
        tag = [f"{self.config['tag'][0]}.{name}", *self.config["tag"][1:], *extra]
        return self.copy(cmd.clone_with(self.config, {"name": f"{self.config['name']}.{name}", "tag": tag}))

    def clients(self):
        """One client per ``client.passes`` entry (in order), each its own benchmark; one perf client without passes"""
        passes = self.client_passes()
        if not passes:
            return self.client("--mode", "perf")
        return PassSequence(*(self.client("--pass", spec["name"], pack=self.pass_pack(spec)) for spec in passes))

    def build_run_plan(self):
        section = self._merged_section("server")
        server = self.inference_server()
        # ``server.multinode: dp``: one server per node joined by data parallelism; otherwise the backend's
        # own multi-node layout (vLLM: Ray)
        service, stop = server.deploy(multinode=section.get("multinode"), dp_local=int(section.get("dp_local", 1)))
        ready_timeout = int(_flag(self.client_argv(), "--ready-check-timeout-sec", 3600))
        return InferencePlan(self, service, self.clients(), ready=server.ready, stop=stop, ready_timeout=ready_timeout)


__pack__ = Serving
