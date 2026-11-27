"""Decision models (typed choice / yes-no / score questions, a probability per option, no text generation): the
serving pack (milabench runs the System One server: engine_server.py with backend "systemone", or a model's own
server with server.command), with this folder's client (main.py: Decision Index accuracy and load passes).

The serving pack is reused as is (runpy: benchfiles are loaded by path, not imported). ``decision`` passes report
scores, not a rate (nolog, like the QualityBench and retrieval passes); prepare.py downloads the model and installs
the Decision Index kit.
"""

import os
import runpy

import milabench.commands as cmd

_serving = runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "serving", "benchfile.py"))
Serving = _serving["Serving"]

# client pass kinds that report scores, not a rate
SCORE_KINDS = ("eval", "quality", "decision", "safety")


class Decision(Serving):
    def pass_pack(self, spec):
        name = spec["name"]
        extra = ["nolog"] if any(kind in spec for kind in SCORE_KINDS) else []
        tag = [f"{self.config['tag'][0]}.{name}", *self.config["tag"][1:], *extra]
        return self.copy(cmd.clone_with(self.config, {"name": f"{self.config['name']}.{name}", "tag": tag}))


__pack__ = Decision
