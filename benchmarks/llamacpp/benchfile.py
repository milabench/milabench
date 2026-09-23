import os

from milabench.merge import merge
from milabench.pack import Package
import milabench.commands as cmd
from milabench.utils import assemble_options


class LlamaCpp(Package):
    base_requirements = "requirements.in"
    prepare_script = "prepare.py"
    main_script = "main.py"

    def make_env(self):
        env = super().make_env()
        env["XDG_CACHE_HOME"] = str(self.dirs.cache)
        env["MILABENCH_TIMELINE_DB"] = str(self.logdir / "benchmark_results.db")
        return env

    def _section_argv(self, name, prepare=False):
        section = self.config.get(name, {}) or {}
        argv = dict(section.get("argv", {}) or {})
        if prepare and name == "client" and isinstance(argv, dict):
            argv = merge(argv, {"--num-prompts": 1})
        return assemble_options(argv)

    @property
    def argv(self):
        return [*self._section_argv("server"), "--", *self._section_argv("client")]

    def build_run_plan(self):
        main = self.dirs.code / self.main_script
        pack = cmd.PackCommand(self, *self.argv, lazy=True)
        return cmd.VoirCommand(pack, cwd=main.parent).use_stdout()


__pack__ = LlamaCpp
