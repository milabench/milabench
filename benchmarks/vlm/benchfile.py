"""Vision-language serving: benchmarks/serving's package (milabench runs the server, one client process per
pass), with this folder's client (main.py: image throughput passes + ``vqa`` accuracy passes) and prepare."""

import importlib.util
import os

_spec = importlib.util.spec_from_file_location(
    "serving_benchfile", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "serving", "benchfile.py"))
_serving = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_serving)


class VisionServing(_serving.Serving):
    # the serving venv's own requirements (install_group homelab-serving): nothing more to install
    base_requirements = "../serving/requirements.in"

    def pass_pack(self, spec):
        """A ``vqa`` pass reports accuracy, not a rate: nolog, like the other scoring passes."""
        pack = super().pass_pack(spec)
        if "vqa" in spec and "nolog" not in pack.config["tag"]:
            pack = pack.copy(_serving.cmd.clone_with(pack.config, {"tag": [*pack.config["tag"], "nolog"]}))
        return pack


__pack__ = VisionServing
