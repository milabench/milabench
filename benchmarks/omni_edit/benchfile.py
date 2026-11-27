"""vLLM-Omni image editing: the omni pack (server = ``vllm serve <model> --omni``, prepare = model download),
with this folder's client (main.py: POST /v1/images/edits). Config as for benchmarks/omni."""
import os
import runpy

_omni = runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "omni", "benchfile.py"))


class OmniEdit(_omni["Omni"]):
    pass


__pack__ = OmniEdit
