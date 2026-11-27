#!/usr/bin/env python
"""System One over HTTP for any decision model: the model's own code behind ``POST /v1/systemone``.

    python engine_server.py <model: Hub id or folder> --adapter ADAPTER [--engine module:Class] [--option key=value ...]
        [--path SUBDIR] [--device cuda] [--served-model-name NAME] [--host 127.0.0.1] [--port 8000]

Adapters (how the model is called; each is the path the model's card evaluates with):

    kit         the model's own Decision Index engine class (``--engine module:Class``, a file of the model's repo,
                e.g. d3_engine:D3Engine, decision_index_engine:VelaEncoderEngine): exactly the published path,
                including its option-order averaging
    transformers  the Decision Index kit's generic engine: any causal LM scored by the log-probability of each
                option key, one forward pass per question (the general-LLM baseline)
    system_one  ``AutoModel.from_pretrained(..., trust_remote_code=True).system_one(state=, questions=, images=)``
    decide      a ``Decider``-style package of the model's repo (``--engine module:Class``, loaded with
                ``Class.from_pretrained(folder, device=)`` or ``Class.load(folder)``), called as
                ``.decide({"state", "questions", "images", **request options})``: THX-02, pplx-decider
    zeroshot    a zero-shot NLI cross-encoder (``transformers`` zero-shot-classification pipeline): choice = the
                options as hypotheses ("This example is {option}: {description}."), noul = entailment of the question

``--option key=value`` (repeatable, JSON values accepted) goes to the engine / loader; ``{model_dir}`` in a value is
the model's local folder. ``request.key=value`` options are added to every request instead (decide adapter: THX-02's
``tta``). The model runs one request at a time (a lock); requests are read concurrently.

Errors follow the Decision Index kit's http engine: an input over the model's limit is 422 with "maximum model length"
(counted as unsupported, nothing is truncated), any other failure 500. ``GET /health`` and ``GET /v1/models`` answer
once the model is loaded and warmed up (milabench's readiness check).
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TOO_LONG = "longer than the maximum model length"


def _value(text: str, model_dir: str):
    text = text.replace("{model_dir}", model_dir)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def parse_options(pairs, model_dir):
    """key=value pairs; milabench joins a YAML list into one comma-separated argument, so a comma that starts a new
    key= also separates pairs (a JSON value may hold commas of its own)."""
    import re

    engine, request = {}, {}
    for pair in [p for arg in pairs or [] for p in re.split(r",(?=[A-Za-z_][\w.]*=)", arg)]:
        key, _, value = pair.partition("=")
        target = request if key.startswith("request.") else engine
        target[key.removeprefix("request.")] = _value(value, model_dir)
    return engine, request


def model_folder(model: str) -> str:
    if os.path.isdir(model):
        return os.path.abspath(model)
    from huggingface_hub import snapshot_download

    # prepare.py downloaded it; offline runs find it in the cache
    return snapshot_download(model)


def _as_dict(obj):
    if isinstance(obj, dict):
        return obj
    for method in ("to_dict", "model_dump", "dict"):
        if hasattr(obj, method):
            return getattr(obj, method)()
    import dataclasses

    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    raise TypeError(f"cannot read a {type(obj).__name__} response")


def _unsupported_types():
    types = []
    try:
        from decision_index.engines import Unsupported

        types.append(Unsupported)
    except ImportError:
        pass
    return tuple(types)


class Model:
    """One decision model behind a lock: ``answer(state, questions, images) -> System One response``."""

    def __init__(self, args):
        self.dir = model_folder(args.model)
        self.name = args.served_model_name or args.model
        self.adapter = args.adapter
        self.lock = threading.Lock()
        options, self.request_options = parse_options(args.option, self.dir)
        sys.path.insert(0, self.dir)  # the model's own modules (engine classes, runtimes)
        for sub in args.path:  # packages in a sub-folder of the repo (pplx-decider: source/src)
            sys.path.insert(0, os.path.join(self.dir, sub))
        t = time.perf_counter()
        if args.adapter in ("kit", "transformers"):
            if args.adapter == "transformers":
                from decision_index.engines.transformers_engine import TransformersEngine as cls

                options.setdefault("model", self.dir)
            else:
                module, _, name = args.engine.partition(":")
                cls = getattr(importlib.import_module(module), name)
            options.setdefault("device", args.device)
            self.engine = cls(**options)
            self.call = lambda state, questions, images: self.engine(state, questions, **({"images": images} if images else {}))[0]
        elif args.adapter == "system_one":
            import torch
            from transformers import AutoModel

            self.engine = AutoModel.from_pretrained(self.dir, trust_remote_code=True, dtype=torch.bfloat16, **options).to(args.device).eval()
            self.call = lambda state, questions, images: self.engine.system_one(state=state, questions=questions, images=images or None)
        elif args.adapter == "decide":
            module, _, name = args.engine.partition(":")
            cls = getattr(importlib.import_module(module), name)
            loader = getattr(cls, "from_pretrained", None) or getattr(cls, "load")
            self.engine = loader(self.dir, device=args.device, **options)
            def call(state, questions, images):
                request = {"state": state, "questions": questions, **self.request_options}
                if images:
                    request["images"] = images
                return _as_dict(self.engine.decide(request))
            self.call = call
        elif args.adapter == "zeroshot":
            from transformers import pipeline

            self.engine = pipeline("zero-shot-classification", model=self.dir, device=args.device, **options)
            self.call = self._zeroshot
        else:
            raise SystemExit(f"unknown adapter {args.adapter}")
        self.loaded_s = time.perf_counter() - t
        self.unsupported = _unsupported_types()
        if not args.no_warmup:
            warm = {"warmup": {"type": "choice", "instructions": "Which color is named?", "criteria": {"red": "red", "blue": "blue"}}}
            for _ in range(2):
                self.answer("The color is red.", warm, None)

    def _zeroshot(self, state, questions, images):
        text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        answers = {}
        for key, q in questions.items():
            if q["type"] == "choice":
                labels = list(q["criteria"])
                hyps = {k: (f"{k}: {d}" if d else k) for k, d in q["criteria"].items()}
                out = self.engine(f"{q.get('instructions', '')}\n\n{text}".strip(), [hyps[k] for k in labels],
                                  hypothesis_template="This example is {}.", multi_label=False)
                back = {hyps[k]: k for k in labels}
                probs = {back[l]: float(s) for l, s in zip(out["labels"], out["scores"])}
                answers[key] = {"type": "choice", "choice": max(probs, key=probs.get), "probabilities": probs}
            elif q["type"] == "noul":
                out = self.engine(text, [q.get("instructions", "")], hypothesis_template="{}", multi_label=True)
                answers[key] = {"type": "noul", "noul": float(out["scores"][0])}
            else:
                raise ValueError(f"question type {q['type']} is not supported by the zeroshot adapter")
        return {"model": self.name, "answers": answers}

    def answer(self, state, questions, images):
        with self.lock:
            response = self.call(state, questions, images)
        response = _as_dict(response)
        response.setdefault("model", self.name)
        return response


class Handler(BaseHTTPRequestHandler):
    server_version = "milabench-systemone/1"

    def log_message(self, fmt, *args):  # one line per request would flood the run log
        pass

    def _send(self, status, body):
        data = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        model = self.server.model
        if self.path == "/health":
            self._send(200, {"status": "ready", "model": model.name, "adapter": model.adapter, "loaded_s": model.loaded_s})
        elif self.path == "/v1/models":
            self._send(200, {"object": "list", "data": [{"id": model.name, "object": "model", "adapter": model.adapter}]})
        else:
            self._send(404, {"detail": "not found"})

    def do_POST(self):
        if self.path != "/v1/systemone":
            return self._send(404, {"detail": "not found"})
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(body, dict) or not isinstance(body.get("questions"), dict) or not body["questions"]:
                return self._send(422, {"detail": "a System One request needs a non-empty questions object"})
        except (ValueError, json.JSONDecodeError) as exc:
            return self._send(422, {"detail": f"unreadable request: {exc}"})
        model = self.server.model
        try:
            self._send(200, model.answer(body.get("state"), body["questions"], body.get("images")))
        except model.unsupported + (OverflowError,) as exc:
            self._send(422, {"detail": f"{TOO_LONG}: {exc}"})
        except Exception as exc:  # noqa: BLE001 one bad request must not stop the server
            message = str(exc)
            too_long = any(m in message.lower() for m in ("maximum model length", "too long", "context length", "input limit"))
            self._send(422 if too_long else 500, {"detail": f"{TOO_LONG}: {message}" if too_long else message,
                                                  **({} if too_long else {"traceback": traceback.format_exc(limit=5)})})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model")
    ap.add_argument("--adapter", required=True, choices=["kit", "transformers", "system_one", "decide", "zeroshot"])
    ap.add_argument("--engine", help="module:Class in the model's repo (kit, decide)")
    ap.add_argument("--option", action="append", default=[], help="key=value for the engine / loader; request.key=value per request")
    ap.add_argument("--path", action="append", default=[], help="extra import folder, relative to the model's folder")
    ap.add_argument("--device", default=None)
    ap.add_argument("--served-model-name", default=None)
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    # serving-pack flags that do not apply here (a decision server has no KV cache or memory fraction)
    args, ignored = ap.parse_known_args(argv)
    if ignored:
        print(f"[systemone] ignoring {ignored}", flush=True)
    if args.device is None:
        try:
            import torch

            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            args.device = "cpu"
    if args.adapter in ("kit", "decide") and not args.engine:
        raise SystemExit(f"--adapter {args.adapter} needs --engine module:Class")
    model = Model(args)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.model = model
    print(f"[systemone] {model.name} ({model.adapter}) loaded in {model.loaded_s:.1f}s on {args.device}: "
          f"http://{args.host}:{args.port}/v1/systemone", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
