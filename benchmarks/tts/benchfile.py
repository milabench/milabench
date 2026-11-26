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
    """The client passes one after the other against the same server; a failed pass does not skip the next
    ones (benchmarks/serving's PassSequence)."""

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


class TTSPlan(cmd.Service):
    """The vLLM-Omni server and its client passes. NJobs / PerGPU copy the plan onto each job's pack: rebuild it."""

    def __init__(self, bench, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bench = bench

    @property
    def pack(self):
        return self.bench

    def copy(self, pack):
        return pack.build_run_plan()


class TTS(Package):
    """Text-to-speech and voice-assistant serving with vLLM-Omni: milabench runs ``vllm serve <model> --omni``
    (milabench.commands.serving, backend vllm-omni), main.py is the client (/v1/audio/speech, or
    /v1/chat/completions with audio out for omni models).

    Config: ``server.argv`` (model first, then --deploy-config / --stage-overrides / ...), ``client.argv``
    (shared) and ``client.passes`` (one result per pass, ``<bench>.<pass>``, like benchmarks/serving)."""

    base_requirements = "requirements.in"
    prepare_script = "prepare.py"
    main_script = "main.py"

    async def install(self):
        await super().install()
        # VoxCPM 2 (vllm-omni serves it through voxcpm.core): without its declared dependencies, which pin dill,
        # fsspec, omegaconf and tomlkit below this venv's versions and pull its demo app (gradio, funasr, modelscope,
        # datasets); what inference needs is in requirements.toml
        await self.pip_install("--no-deps", "voxcpm==2.0.3")

    def make_env(self):
        env = super().make_env()
        env["XDG_CACHE_HOME"] = str(self.dirs.cache)
        return env

    def _arch(self):
        return self.config.get("system", {}).get("arch", "cuda")

    def _variant(self):
        return (self.config.get("variants") or {}).get(self._arch(), {}) or {}

    def _section(self, name):
        shared = dict(self.config.get(name, {}) or {})
        variant = dict(self._variant().get(name, {}) or {})
        merged = merge(shared, variant)
        merged["argv"] = merge(shared.get("argv", {}), variant.get("argv", {}))
        return merged

    def server_argv(self):
        return assemble_options(self._section("server")["argv"])

    def client_argv(self):
        return assemble_options(self._section("client")["argv"])

    def client_passes(self):
        variant = self._variant().get("client") or {}
        return variant.get("passes") or (self.config.get("client") or {}).get("passes") or []

    @property
    def argv(self):
        return self.client_argv()

    def build_prepare_plan(self):
        prep = self.dirs.code / self.prepare_script
        argv = [*self.server_argv(), "--", *self.client_argv()]
        return cmd.PackCommand(self, prep, *argv, env=self.make_env(), cwd=prep.parent)

    def _client(self, *args, pack=None):
        main = self.dirs.code / self.main_script
        return cmd.VoirCommand(cmd.PackCommand(pack or self, *args, *self.client_argv()), cwd=str(main.parent)).use_stdout()

    def _pass_pack(self, spec):
        """A client pass as its own benchmark ``<name>.<pass>`` (own voir monitor and .data file)."""
        name = spec["name"]
        tag = [f"{self.config['tag'][0]}.{name}", *self.config["tag"][1:]]
        return self.copy(cmd.clone_with(self.config, {"name": f"{self.config['name']}.{name}", "tag": tag}))

    def clients(self):
        passes = self.client_passes()
        if not passes:
            return self._client()
        return PassSequence(*(self._client("--pass", spec["name"], pack=self._pass_pack(spec)) for spec in passes))

    def build_run_plan(self):
        client_argv = self.client_argv()
        server = InferenceServer(
            # stopped by a signal once the clients are done: its exit code is not a result
            self.copy(cmd.clone_with(self.config, {"tag": [*self.config["tag"], "server", "nolog"]})),
            *self.server_argv(),
            backend=self._section("server").get("backend", "vllm-omni"),
            command=self._section("server").get("command"),
            host=_flag(client_argv, "--host", "127.0.0.1"),
            port=_flag(client_argv, "--port", 8000),
            cwd=str(self.dirs.code),
        )
        service, stop = server.deploy()
        ready_timeout = int(_flag(client_argv, "--ready-check-timeout-sec", 3600))
        return TTSPlan(self, service, self.clients(), ready=server.ready, stop=stop, ready_timeout=ready_timeout)


__pack__ = TTS
