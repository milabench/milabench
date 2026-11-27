"""Classic document pipelines (tesseract, PP-OCRv6 through RapidOCR, docling): no server. Each ``client.passes``
entry runs as its own benchmark (``<name>.<pass>``, its own .data file and GPU / power samples), like the
serving benches' passes, so the report reads both the same way."""

import importlib.util
import os

import milabench.commands as cmd
from milabench.pack import Package

_spec = importlib.util.spec_from_file_location(
    "serving_benchfile", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "serving", "benchfile.py"))
_serving = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_serving)


class PipelinePlan(_serving.PassSequence):
    """The passes one after the other; NJobs / PerGPU copy the plan onto each job's pack: rebuild it."""

    def __init__(self, bench, *executors):
        super().__init__(*executors)
        self.bench = bench

    @property
    def pack(self):
        return self.bench

    def copy(self, pack):
        return pack.build_run_plan()


class DocParsePipeline(Package):
    base_requirements = "requirements.in"
    prepare_script = "prepare.py"
    main_script = "main.py"

    def client_passes(self) -> list[dict]:
        return (self.config.get("client") or {}).get("passes") or []

    def pass_pack(self, spec):
        name = spec["name"]
        quality = (spec.get("docparse") or {}).get("mode", "quality") == "quality"
        tag = [f"{self.config['tag'][0]}.{name}", *self.config["tag"][1:], *(["nolog"] if quality else [])]
        return self.copy(cmd.clone_with(self.config, {"name": f"{self.config['name']}.{name}", "tag": tag}))

    def build_run_plan(self):
        main = self.dirs.code / self.main_script
        clients = [cmd.VoirCommand(cmd.PackCommand(self.pass_pack(spec), "--pass", spec["name"], *self.argv),
                                   cwd=str(main.parent)).use_stdout()
                   for spec in self.client_passes()]
        return PipelinePlan(self, *clients)


__pack__ = DocParsePipeline
