#!/usr/bin/env python3
"""BFCL (bfcl-eval) against the benchmark's OpenAI-compatible server, in FC mode: the request carries ``tools``
and the server's own tool-call parser turns the reply into ``tool_calls``, as an agent sees them.

Runs in QualityBench's pinned BFCL environment ($QUALITYBENCH_HOME/venvs/bfcl, bfcl-eval 2026.3.23), standalone
(imports nothing from milabench). Same registration as QualityBench's helpers/bfcl_run.py, plus:

  BFCL_THINKING     on / off: chat_template_kwargs.enable_thinking on every request (unset: the template's default)
  BFCL_MAX_TOKENS   max_tokens on every request (unset: the server's default, the rest of the context)

  OPENAI_BASE_URL=http://host:port/v1 OPENAI_API_KEY=local BFCL_PROJECT_ROOT=<dir> \
      python bfcl_shim.py <served-model-name> generate|evaluate [bfcl args]
"""
import dataclasses
import os
import sys

import openai

_OpenAI = openai.OpenAI
# one hour per request, three attempts at most (QualityBench's settings)
openai.OpenAI = lambda *a, **k: _OpenAI(*a, **{**k, "timeout": 3600, "max_retries": 2})

from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING  # noqa: E402
from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler  # noqa: E402

_create = OpenAICompletionsHandler.generate_with_backoff


def _generate(self, **kwargs):
    thinking = os.environ.get("BFCL_THINKING")
    if thinking in ("on", "off"):
        extra = dict(kwargs.get("extra_body") or {})
        extra.setdefault("chat_template_kwargs", {})["enable_thinking"] = thinking == "on"
        kwargs["extra_body"] = extra
    if max_tokens := os.environ.get("BFCL_MAX_TOKENS"):
        kwargs["max_tokens"] = int(max_tokens)
    # "store" is an OpenAI-platform field: vLLM ignores it, llama-server may not
    kwargs.pop("store", None)
    return _create(self, **kwargs)


OpenAICompletionsHandler.generate_with_backoff = _generate

model, cmd, rest = sys.argv[1], sys.argv[2], sys.argv[3:]
template = next(c for c in MODEL_CONFIG_MAPPING.values() if c.model_handler is OpenAICompletionsHandler and c.is_fc_model)
key = f"{model}-FC"
MODEL_CONFIG_MAPPING[key] = dataclasses.replace(
    template, model_name=model, display_name=f"{model} (FC, OpenAI-compatible)", url="", org="local",
    license="", input_price=None, output_price=None, is_fc_model=True)

from bfcl_eval.__main__ import cli  # noqa: E402

sys.argv = ["bfcl", cmd, "--model", key, *rest]
cli()
