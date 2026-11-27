"""Document parsing with a served VLM: benchmarks/serving's package (milabench runs the server, one client process
per pass), with this folder's client (main.py: ``docparse`` speed and quality passes) and prepare."""

import importlib.util
import os

_spec = importlib.util.spec_from_file_location(
    "serving_benchfile", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "serving", "benchfile.py"))
_serving = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_serving)


class DocParseServing(_serving.Serving):
    # the serving venv's own requirements (install_group homelab-serving): the client needs nothing more; the
    # scorers and vendor SDKs live in their own venvs (toolenv.py), installed by prepare.py
    base_requirements = "../serving/requirements.in"

    def pass_pack(self, spec):
        """A ``docparse`` quality pass reports scores, not a rate: nolog, like the other scoring passes."""
        pack = super().pass_pack(spec)
        quality = "docparse" in spec and (spec["docparse"] or {}).get("mode", "quality") == "quality"
        if quality and "nolog" not in pack.config["tag"]:
            pack = pack.copy(_serving.cmd.clone_with(pack.config, {"tag": [*pack.config["tag"], "nolog"]}))
        return pack


__pack__ = DocParseServing
