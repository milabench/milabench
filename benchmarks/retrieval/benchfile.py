"""Embedding / reranking serving: benchmarks/serving's pack (milabench runs the vLLM server, one client process per
client pass), with this folder's client (main.py: pooling metrics, batched real-text requests, retrieval passes).

The serving pack is reused as is (runpy: benchfiles are loaded by path, not imported); only two things differ:
``retrieval`` passes report scores, not a rate (nolog, like QualityBench passes), and prepare.py downloads
their datasets.
"""

import os
import runpy

import milabench.commands as cmd

_serving = runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "serving", "benchfile.py"))
Serving = _serving["Serving"]

# client pass kinds that report scores, not a rate
SCORE_KINDS = ("eval", "quality", "swebench", "terminalbench", "retrieval")


class Retrieval(Serving):
    def pass_pack(self, spec):
        name = spec["name"]
        extra = ["nolog"] if any(kind in spec for kind in SCORE_KINDS) else []
        tag = [f"{self.config['tag'][0]}.{name}", *self.config["tag"][1:], *extra]
        return self.copy(cmd.clone_with(self.config, {"name": f"{self.config['name']}.{name}", "tag": tag}))


__pack__ = Retrieval
