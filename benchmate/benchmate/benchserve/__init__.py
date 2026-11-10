"""Backend-agnostic LLM serving benchmark client.

Drives `vllm.benchmarks.serve.main()` against any OpenAI-compatible endpoint
(vLLM, ATOM, sglang, llama.cpp llama-server, ...) and streams per-window /
per-request metrics to a benchmate observer installed via `set_metric_sink()`.

Shared by the serving packs:

* ``benchmarks/vllm``: starts the server itself (`InferenceServer`) and runs
  every client pass in-process against it (`run_benchmark_watched`), each pass
  recorded as its own bench (`PassRecorder`).
* ``benchmarks/serving``: milabench runs the server, the client runs one pass
  or a group of passes per process (`main`, `client_passes`).
* ``benchmarks/llamacpp``: starts llama-server, single default pass.

Client passes (``client.passes`` in the config, the arch variant's replacing
the shared ones) are either load points (``argv`` overlay on the client argv)
or lm-eval accuracy passes (``eval``).

Can also run standalone as the `benchserve` console script — an
instrumented drop-in for `vllm bench serve` against any endpoint that is
already up:

    benchserve --base-url http://127.0.0.1:8000 --model <hf-id> \
        --dataset-name random --num-prompts 40 --random-input-len 1024 ...

The vllm (and numpy) imports are deferred inside functions so that benchmate
stays importable in environments where vllm is not installed.
"""

from argparse import ArgumentParser
import bisect
from collections import defaultdict
import glob
import json
import os
import subprocess
import sys
import threading
import time

from benchmate.timeline import timeline, TimelineConfig, _default_db_path
from benchmate.toggles import vllm_request_metrics_enabled


# name of the client pass being run (client.passes), None for the single default pass
current_pass = None

# ITL is binned by the position of the token in the sequence (prompt + generated so far)
CONTEXT_BINS = [0, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 1 << 30]


# ---------------------------------------------------------------------------
# Metric sink
# ---------------------------------------------------------------------------

_metric_sink = None


def set_metric_sink(fn):
    """Install the observer callback used to record benchmark metrics."""
    global _metric_sink
    _metric_sink = fn


def push_metric(**kwargs):
    """Record a metric: into the running pass's own .data file if one is recorded, else to the observer."""
    kwargs.setdefault("time", time.time())
    if not recorder.write(kwargs, task="train"):
        _metric_sink(**kwargs)


def make_observer():
    """The observer every serving pack uses, installed as the metric sink; returns (observer, bench_monitor).

    The GPU monitor's samples (power, memory, load) go through `recorder` too: while a pass is recorded
    they land in the pass's file, so its energy is that of its own requests.
    """
    import torchcompat.core as accelerator
    from benchmate.observer import BenchObserver
    from benchmate.toggles import get_observation_count
    import benchmate.monitor as monitor

    observer = BenchObserver(
        accelerator.Event,
        earlystop=get_observation_count(120),
        batch_size_fn=lambda x: len(x[0]),
        raise_stop_program=False,
        stdout=True,
    )
    set_metric_sink(observer.record_metric)

    if not getattr(monitor.metric_log, "_benchserve_routed", False):
        metric_log = monitor.metric_log

        def recorded_metric_log():
            log = metric_log()

            def route(data):
                if not recorder.write(data, task=data.get("task", "main")):
                    log(data)

            return route

        recorded_metric_log._benchserve_routed = True
        monitor.metric_log = recorded_metric_log

    return observer, monitor.bench_monitor


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _milabench_config() -> dict:
    raw = os.environ.get("MILABENCH_CONFIG")
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def _run_description() -> str:
    cfg = _milabench_config()
    if not cfg:
        return "vllm"
    name = cfg.get("name") or ".".join(cfg.get("tag", []))
    if current_pass:
        name = f"{name}/{current_pass}"
    model = (cfg.get("client") or {}).get("argv", {}).get("--model")
    if model:
        return f"{name} ({model})"
    return name or "vllm"


def _percentiles(name, values, unit_scale=1.0):
    import numpy as np

    if not values:
        return {}
    arr = np.asarray(values, dtype=float) * unit_scale
    out = {f"{name}_mean": float(arr.mean())}
    for p in (50, 90, 99):
        out[f"{name}_p{p}"] = float(np.percentile(arr, p))
    return out


def _itl_by_context(input_requests, outputs):
    """Mean inter-token latency per context-length bin: how decoding slows down as the KV cache fills."""
    sums = defaultdict(float)
    counts = defaultdict(int)
    for inp, out in zip(input_requests, outputs):
        if not out.success:
            continue
        for j, itl in enumerate(out.itl):
            position = inp.prompt_len + j + 1
            b = bisect.bisect_right(CONTEXT_BINS, position) - 1
            sums[b] += itl
            counts[b] += 1
    return {
        f"{CONTEXT_BINS[b]}-{CONTEXT_BINS[b + 1]}": {"itl_mean_ms": 1000 * sums[b] / counts[b], "tokens": counts[b]}
        for b in sorted(counts)
    }


def push_pass_summary(input_requests, outputs, dur_s, output_lens):
    """One event per client pass: the latency/throughput point and the ITL-vs-context curve."""
    import numpy as np

    ok = [(inp, out, n) for inp, out, n in zip(input_requests, outputs, output_lens) if out.success]
    tpots = [
        (out.latency - out.ttft) / (n - 1) for _, out, n in ok if n > 1
    ]
    end = time.time()
    summary = {
        "name": current_pass or "default",
        # the window in which the requests ran (no server start / warmup): energy per pass uses the GPU power in it
        "start_time": end - dur_s,
        "end_time": end,
        "completed": len(ok),
        "failed": len(outputs) - len(ok),
        "duration_s": dur_s,
        "request_throughput": len(ok) / dur_s if dur_s else 0.0,
        "output_throughput": sum(n for _, _, n in ok) / dur_s if dur_s else 0.0,
        "total_throughput": sum(inp.prompt_len + n for inp, _, n in ok) / dur_s if dur_s else 0.0,
        "mean_input_len": float(np.mean([inp.prompt_len for inp, _, _ in ok])) if ok else 0.0,
        "mean_output_len": float(np.mean([n for _, _, n in ok])) if ok else 0.0,
        **_percentiles("ttft_ms", [out.ttft for _, out, _ in ok], 1000),
        **_percentiles("tpot_ms", tpots, 1000),
        **_percentiles("itl_ms", [x for _, out, _ in ok for x in out.itl], 1000),
        **_percentiles("e2el_ms", [out.latency for _, out, _ in ok], 1000),
        "itl_by_context": _itl_by_context(input_requests, outputs),
    }
    print("[pass]", json.dumps(summary), flush=True)
    push_metric(pass_summary=summary)
    recorder.last_summary = summary


def log_request(input_requests, outputs):
    for inp, out in zip(input_requests, outputs):
        push_metric(**{
            "request_id": inp.request_id,
            "start_time": out.start_time,
            "prompt_len": out.prompt_len,
            "output_len": out.output_tokens,
            "success": out.success,
            "latency": out.latency,
            "ttft": out.ttft,
            "itl": out.itl,
            "tpot": out.tpot,
        })


def calculate_metrics(
    input_requests,
    outputs,
    dur_s: float,
    tokenizer,
    selected_percentiles,
    goodput_config_dict,
):
    """Push timeline windows, per-request metrics (``vllm_request_metrics_enabled``) and the pass summary.

    Called before vLLM's own ``calculate_metrics``, with the same arguments.
    """

    # log_request(input_requests, outputs)

    config = TimelineConfig()
    description = _run_description()
    db_path = _default_db_path()
    push_request_metrics = vllm_request_metrics_enabled()
    print(f"[timeline] database path: {db_path}", flush=True)
    print(
        f"[timeline] push_request_metrics={push_request_metrics}",
        flush=True,
    )

    for sampled_obs in timeline(
        outputs,
        config=config,
        description=description,
        persist=True,
    ):
        push_metric(**sampled_obs)

    actual_output_lens: list[int] = []

    for i in range(len(outputs)):
        if outputs[i].success:
            output_len = outputs[i].output_tokens

            if not output_len:
                # We use the tokenizer to count the number of output tokens
                # for some serving backends instead of looking at
                # len(outputs[i].itl) since multiple output tokens may be
                # bundled together
                # Note : this may inflate the output token count slightly
                output_len = len(
                    tokenizer(
                        outputs[i].generated_text, add_special_tokens=False
                    ).input_ids
                )
            actual_output_lens.append(output_len)

            if push_request_metrics:
                push_metric(ttfts=outputs[i].ttft, units="s")
                push_metric(e2els=outputs[i].latency, units="s")

                if len(outputs[i].itl) > 0:
                    push_metric(
                        itl=sum(outputs[i].itl) / len(outputs[i].itl), units="s"
                    )

                push_metric(
                    input_tok=input_requests[i].prompt_len, units="count"
                )
                push_metric(output_tok=output_len, units="count")

                tok_s = (input_requests[i].prompt_len + output_len) / outputs[
                    i
                ].latency
                push_metric(request_rate=tok_s, units="tok/s")
        else:
            actual_output_lens.append(0)

    push_pass_summary(input_requests, outputs, dur_s, actual_output_lens)


# ---------------------------------------------------------------------------
# vllm bench serve
# ---------------------------------------------------------------------------


def _gpqa_diamond_cls():
    import vllm.benchmarks.datasets as datasets
    from vllm.benchmarks.serve import SampleRequest

    class GPQADiamond(datasets.HuggingFaceDataset):
        IS_MULTIMODAL = False
        SUPPORTED_DATASET_PATHS = {'hendrydong/gpqa_diamond'}

        def sample(
            self,
            tokenizer,
            num_requests: int,
            output_len: int | None = None,
            enable_multimodal_chat: bool = False,
            request_id_prefix: str = "",
            no_oversample: bool = False,
            **kwargs,
        ) -> list:
            sampled_requests = []
            ind = 0
            dynamic_output = output_len is None

            for item in self.data["test"]:
                if len(sampled_requests) >= num_requests:
                    break

                prompt, completion = item["problem"], item["solution"]

                prompt_ids = tokenizer(prompt).input_ids
                completion_ids = tokenizer(completion).input_ids

                prompt_len = len(prompt_ids)
                completion_len = len(completion_ids)
                output_len = completion_len if dynamic_output else output_len

                assert isinstance(output_len, int) and output_len > 0

                sampled_requests.append(
                    SampleRequest(
                        prompt=prompt,
                        prompt_len=prompt_len,
                        expected_output_len=output_len,
                        multi_modal_data=None,
                        request_id=request_id_prefix + str(ind),
                    )
                )
                ind += 1
            self.maybe_oversample_requests(
                sampled_requests, num_requests, request_id_prefix, no_oversample
            )
            return sampled_requests

    return GPQADiamond


# pack-specific extensions of the current benchmark() call, see benchmark()
_samples_hook = None


def add_real_text_options(parser):
    # real text instead of random token ids, same lengths (--random-input-len / --random-output-len)
    parser.add_argument("--real-text", action="store_true")


def real_text_samples(args, tokenizer, datasets):
    """``--real-text``: prompts cut from the OpenHermes corpus (text_corpus) to exactly
    ``--random-input-len`` tokens. ``--dataset-name sharegpt`` without a path reads the same corpus.

    Random token ids change the measured performance (speculative decoding rejects almost every
    draft, MoE routing and attention see unnatural inputs): perf passes should use real text."""
    from benchmate.benchserve import text_corpus

    if getattr(args, "real_text", False):
        path = args.dataset_path if args.dataset_name not in ("random", "prefix_repetition") and args.dataset_path else None
        dataset = text_corpus.make_real_text_dataset(datasets)(
            dataset_path=path or text_corpus.build_corpus(), random_seed=args.seed,
        )
        if args.dataset_name == "prefix_repetition":
            return dataset.sample_shared_prefix(
                tokenizer=tokenizer,
                num_requests=args.num_prompts,
                prefix_len=args.prefix_repetition_prefix_len,
                suffix_len=args.prefix_repetition_suffix_len,
                num_prefixes=args.prefix_repetition_num_prefixes,
                output_len=args.prefix_repetition_output_len,
                request_id_prefix=args.request_id_prefix,
            )
        return dataset.sample(
            tokenizer=tokenizer,
            num_requests=args.num_prompts,
            input_len=args.random_input_len,
            output_len=args.random_output_len,
            request_id_prefix=args.request_id_prefix,
        )
    if args.dataset_name == "sharegpt" and args.dataset_path is None:
        args.dataset_path = text_corpus.build_corpus()
    return None


def prepare_passes(passes=None):
    """For a pack's prepare.py: install the tools and data of the QualityBench / SWE-bench passes
    (the runs are offline). passes: default ``client_passes()`` (MILABENCH_CONFIG)."""
    passes = client_passes() if passes is None else passes
    for spec in passes or []:
        if "quality" in spec:
            from benchmate.benchserve import quality
            quality.prepare(spec["quality"] or {})
        if "swebench" in spec:
            from benchmate.benchserve import swebench
            swebench.prepare(spec["swebench"] or {})


def needs_text_corpus(argv, passes=()) -> bool:
    """The client argv, or one of its passes' overlays, reads the corpus: ``--real-text``, or
    ``--dataset-name sharegpt`` without its own ``--dataset-path``."""
    for args in [list(argv), *(_options(spec.get("argv")) for spec in passes or [])]:
        if "--real-text" in args:
            return True
        if _flag(args, "--dataset-name") == "sharegpt" and not _flag(args, "--dataset-path"):
            return True
    return False


def prepare_text_corpus(argv, passes=None) -> str | None:
    """For a pack's prepare.py: build the corpus when the client reads it (runs are offline).

    passes: the client passes, default ``client_passes()`` (MILABENCH_CONFIG)."""
    from benchmate.benchserve import text_corpus

    passes = client_passes() if passes is None else passes
    if needs_text_corpus(argv, passes):
        return text_corpus.build_corpus()
    return None


def _patch_vllm_bench(bench):
    """Wrap vLLM's functions once: client passes call benchmark() several times."""
    if getattr(bench, "_milabench_patched", False):
        return
    bench._milabench_patched = True

    import vllm.benchmarks.datasets as datasets

    def open_dataset(dataset_cls, args, tokenizer):
        hf_kwargs = {}
        return dataset_cls(
            dataset_path=args.dataset_path,
            dataset_subset=args.hf_subset,
            dataset_split=args.hf_split,
            random_seed=args.seed,
            no_stream=args.no_stream,
            hf_name=args.hf_name,
            # missing arg ?
            # disable_shuffle=args.disable_shuffle,
        ).sample(
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            output_len=args.hf_output_len,
            request_id_prefix=args.request_id_prefix,
            no_oversample=args.no_oversample,
            # missing arg ?
            # skip_chat_template=args.skip_chat_template,
            **hf_kwargs,
        )

    original_get_samples = bench.get_samples

    def get_samples(args, tokenizer):
        if _samples_hook is not None:
            samples = _samples_hook(args, tokenizer, datasets)
            if samples is not None:
                return samples

        samples = real_text_samples(args, tokenizer, datasets)
        if samples is not None:
            return samples

        match args.hf_name:
            case "openslr/librispeech_asr":
                return open_dataset(datasets.ASRDataset, args, tokenizer)

            case "hendrydong/gpqa_diamond":
                return open_dataset(_gpqa_diamond_cls(), args, tokenizer)

            case _:
                return original_get_samples(args, tokenizer)

    bench.get_samples = get_samples

    original = bench.calculate_metrics

    def new_calculate_metrics(*args, **kwargs):
        calculate_metrics(*args, **kwargs)
        return original(*args, **kwargs)

    bench.calculate_metrics = new_calculate_metrics


def benchmark(argv, *, samples_hook=None, parser_hook=None):
    """Run ``vllm bench serve <argv>`` in-process, instrumented.

    samples_hook(args, tokenizer, datasets) -> list | None
        Called first by ``get_samples``: returns the requests, or None for the default lookup
        (``--real-text``, then vLLM's datasets; it may still adjust ``args``).
        ``datasets`` is ``vllm.benchmarks.datasets``.
    parser_hook(parser)
        Adds the pack's own options to vLLM's argument parser.
    """
    # vllm bench serve                                      \
    #     --backend openai                                  \
    #     --label milabench                                 \
    #     --model <your_model>                              \
    #     --dataset-name <dataset_name. Default 'random'>   \
    #     --request-rate inf                                \
    #     --num-prompts 1000
    #
    # Other datasets: datasets.InstructCoderDataset / BlazeditDataset (coding),
    # datasets.MTBenchDataset (open ended writing), datasets.AIMODataset (reasoning)
    global _samples_hook
    import vllm.benchmarks.serve as bench

    _patch_vllm_bench(bench)
    _samples_hook = samples_hook

    parser = ArgumentParser()
    bench.add_cli_args(parser)
    add_real_text_options(parser)
    if parser_hook is not None:
        parser_hook(parser)

    print("BENCH:", " ".join(['vllm', 'bench', 'serve'] + argv))
    args = parser.parse_args(argv)

    bench.main(args)
    print("FINISHED")


# ---------------------------------------------------------------------------
# lm-eval passes
# ---------------------------------------------------------------------------


def _options(overlay: dict) -> list[str]:
    argv = []
    for key, value in (overlay or {}).items():
        if value is True:
            argv.append(key)
        elif value is not False and value is not None:
            argv += [key, str(value)]
    return argv


def _flag(argv, name, default=None):
    for i, arg in enumerate(argv):
        if arg == name and i + 1 < len(argv):
            value = argv[i + 1]
        elif arg.startswith(name + "="):
            value = arg.split("=", 1)[1]
        else:
            continue
        default = value
    return default


def wait_ready(url, timeout):
    import urllib.request

    deadline = time.time() + timeout
    while True:
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                if response.status == 200:
                    return
        except OSError:
            pass
        if time.time() > deadline:
            raise TimeoutError(f"server not ready after {timeout}s: {url}")
        time.sleep(5)


def run_eval(bench_argv, spec: dict, *, eval_tasks: str | None = None, output_name: str | None = None):
    """lm-eval against the running server (OpenAI chat API); one ``eval_summary`` event per task.

    eval_tasks: folder of the pack's lm-eval task definitions (``--include_path``).
    output_name: lm-eval output folder, next to the run's .data files (default ``<bench>.lm_eval``).
    """
    host = _flag(bench_argv, "--host", "127.0.0.1")
    port = _flag(bench_argv, "--port", "8000")
    model = _flag(bench_argv, "--served-model-name") or _flag(bench_argv, "--model")
    tokenizer = _flag(bench_argv, "--tokenizer") or _flag(bench_argv, "--model")
    # MILABENCH_DIR_RUNS is the root of every run: the answers go next to this run's .data files
    cfg = _milabench_config()
    runs = os.path.join(os.environ.get("MILABENCH_DIR_RUNS", "."), cfg.get("run_name", ""))
    output = os.path.join(runs, output_name or f"{cfg.get('name', 'vllm')}.lm_eval")

    model_args = ",".join([
        f"base_url=http://{host}:{port}/v1/chat/completions",
        f"model={model}",
        f"tokenizer={tokenizer}",
        "tokenizer_backend=huggingface",
        "trust_remote_code=True",
        f"num_concurrent={spec.get('num_concurrent', 64)}",
        "max_retries=3",
        f"timeout={spec.get('timeout', 3600)}",
    ])
    gen_kwargs = ",".join(f"{k}={v}" for k, v in (spec.get("gen_kwargs") or {"max_gen_toks": 8192}).items())
    wait_ready(f"http://{host}:{port}/v1/models", float(_flag(bench_argv, "--ready-check-timeout-sec", 3600)))

    for task in spec["tasks"]:
        cmd = [
            sys.executable, "-m", "lm_eval",
            "--model", "local-chat-completions",
            "--model_args", model_args,
            "--tasks", task,
            *(["--include_path", eval_tasks] if eval_tasks else []),
            "--apply_chat_template",
            "--gen_kwargs", gen_kwargs,
            "--output_path", output,
            "--seed", "0",
        ]
        limit = spec.get("limit")
        if isinstance(limit, dict):
            limit = limit.get(task)
        if limit:
            cmd += ["--limit", str(limit)]
        # every question's prompt, response and score (samples_<task>_*.jsonl), to compare configs per question
        if spec.get("log_samples"):
            cmd += ["--log_samples"]
        print("EVAL:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)

        # lm-eval writes <output>/<model>/results_<timestamp>.json
        result_files = sorted(glob.glob(os.path.join(output, "**", "results_*.json"), recursive=True), key=os.path.getmtime)
        with open(result_files[-1]) as f:
            results = json.load(f)
        for task_name, metrics in results["results"].items():
            # metric keys are "<metric>,<filter>"; lm-eval also reports e.g. sample_len
            scores = {
                key: value for key, value in metrics.items()
                if isinstance(value, (int, float)) and "," in key and "stderr" not in key
            }
            stderr = {
                key: value for key, value in metrics.items()
                if isinstance(value, (int, float)) and "stderr" in key
            }
            summary = {
                "name": current_pass or "eval",
                "task": task_name,
                "samples": results.get("n-samples", {}).get(task_name, {}).get("effective"),
                "scores": scores,
                "stderr": stderr,
            }
            print("[eval]", json.dumps(summary), flush=True)
            push_metric(eval_summary=summary)


# ---------------------------------------------------------------------------
# Client passes
# ---------------------------------------------------------------------------


class PassRecorder:
    """Each client pass as its own bench of the run: ``runs/<run>/<bench>.<pass>.<job>.data``.

    Used when one process runs every pass against a server it started (``benchmarks/vllm``). While a pass
    runs, its events (metrics pushed by this module and the GPU monitor's samples, see `make_observer`) go
    to that file only, between milabench-style config / meta / start ... end events; its bench name is
    ``<bench>.<pass>``, so ``milabench report`` lists every load point as its own row. What milabench
    records from stdout is the server's (``<bench>.server.nolog.<job>.data``): server start, weight load,
    warmup, the samples between passes, and a ``pass_done`` pointer per pass.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.fp = None
        self.folder = None
        self.path = None
        self.last_summary = None

    def _event(self, event, data):
        self.fp.write(json.dumps({"event": event, "data": data, "pipe": None if event != "data" else "data"}) + "\n")

    def _parent_meta(self, cfg):
        # milabench writes the meta event at the top of the bench's own .data, before this script starts
        path = os.path.join(os.environ.get("MILABENCH_DIR_RUNS", "."), cfg["run_name"], ".".join(cfg["tag"]) + ".data")
        try:
            with open(path) as f:
                for line in f:
                    if line.startswith('{"event": "meta"'):
                        return json.loads(line)["data"]
        except (OSError, json.JSONDecodeError):
            pass
        return None

    @property
    def active(self) -> bool:
        return self.fp is not None

    def open(self, spec: dict):
        cfg = _milabench_config()
        if not cfg.get("run_name"):
            return
        bench = f"{cfg['name']}.{spec['name']}"
        # the server's own tag is [<bench>, server, nolog, <job>]: a pass is a regular bench, [<bench>.<pass>, <job>].
        # An eval pass reports accuracy (eval_summary), no rate: nolog, or milabench's report counts it as failed
        tag = [bench, *(["nolog"] if is_score_pass(spec) else []), str(cfg.get("job-number", 0))]
        self.folder = os.path.join(os.environ.get("MILABENCH_DIR_RUNS", "."), cfg["run_name"])
        os.makedirs(self.folder, exist_ok=True)
        self.path = os.path.join(self.folder, ".".join(tag) + ".data")
        self.last_summary = None
        with self.lock:
            self.fp = open(self.path, "w")
            self._event("config", {**cfg, "name": bench, "tag": tag, "server_bench": cfg["name"], "pass": spec})
            if meta := self._parent_meta(cfg):
                self._event("meta", meta)
            self._event("start", {"command": [*sys.argv, "--pass", spec["name"]], "time": time.time()})

    def write(self, data: dict, task: str) -> bool:
        """Write to the pass's file if a pass runs; False: the caller sends it to milabench (server file)."""
        with self.lock:
            if self.fp is None:
                return False
            self._event("data", {"task": task, **data})
            return True

    def close(self, return_code: int):
        with self.lock:
            if self.fp is not None:
                self._event("end", {"command": [*sys.argv, "--pass", current_pass], "time": time.time(),
                                    "return_code": return_code})
                self.fp.close()
            self.fp = None


recorder = PassRecorder()


# pass kinds that report scores (accuracy, solve rate), not a rate: lm-eval, QualityBench, SWE-bench agent
SCORE_PASSES = ("eval", "quality", "swebench")


def is_score_pass(spec: dict) -> bool:
    return any(kind in spec for kind in SCORE_PASSES)


def client_passes(mode: str = "all", name: str | None = None) -> list[dict] | None:
    """Passes of ``client.passes`` (shared section, then the arch variant) run by *mode*, or the pass *name*.

    ``eval`` keeps the lm-eval passes, ``perf`` the others, ``all`` both. ``None`` when the config has
    no passes: a single default benchmark run (``perf`` / ``all``)."""
    cfg = _milabench_config()
    arch = cfg.get("system", {}).get("arch", "cuda")
    variant = ((cfg.get("variants") or {}).get(arch) or {}).get("client") or {}
    passes = variant.get("passes") or (cfg.get("client") or {}).get("passes")
    if name is not None:
        # one client process per pass (benchmarks/serving benchfile.clients())
        selected = [spec for spec in passes or [] if spec["name"] == name]
        if not selected:
            raise SystemExit(f"no client pass named {name!r} in {[spec['name'] for spec in passes or []]}")
        return selected
    if not passes:
        return [] if mode == "eval" else None
    if mode == "eval":
        return [spec for spec in passes if is_score_pass(spec)]
    if mode == "perf":
        return [spec for spec in passes if not is_score_pass(spec)]
    return passes


def _pass_output(kind: str, record: bool) -> tuple[str, str]:
    """(folder, bench name) of a scoring pass' outputs, next to the run's .data files."""
    cfg = _milabench_config()
    runs = os.path.join(os.environ.get("MILABENCH_DIR_RUNS", "."), cfg.get("run_name", ""))
    # recorded passes share the server's bench name: the pass name keeps their outputs apart
    name = f"{cfg.get('name', 'vllm')}.{current_pass or kind}" if record else cfg.get("name", "vllm")
    return os.path.join(runs, f"{name}.{kind}"), name


def _run_pass(bench_argv, spec, *, record, eval_tasks, **hooks):
    if "eval" in spec:
        cfg = _milabench_config()
        output_name = f"{cfg.get('name', 'vllm')}.{current_pass or 'eval'}.lm_eval" if record else None
        run_eval(bench_argv, spec["eval"], eval_tasks=eval_tasks, output_name=output_name)
    elif "quality" in spec:
        from benchmate.benchserve import quality

        folder, name = _pass_output("qualitybench", record)
        wait_ready(f"http://{_flag(bench_argv, '--host', '127.0.0.1')}:{_flag(bench_argv, '--port', '8000')}/v1/models",
                   float(_flag(bench_argv, "--ready-check-timeout-sec", 3600)))
        quality.run(bench_argv, spec["quality"] or {}, flag=_flag, output_dir=folder, name=name, push=push_metric)
    elif "swebench" in spec:
        from benchmate.benchserve import swebench

        folder, name = _pass_output("swebench", record)
        wait_ready(f"http://{_flag(bench_argv, '--host', '127.0.0.1')}:{_flag(bench_argv, '--port', '8000')}/v1/models",
                   float(_flag(bench_argv, "--ready-check-timeout-sec", 3600)))
        swebench.run(bench_argv, spec["swebench"] or {}, flag=_flag, output_dir=folder, name=name, push=push_metric)
    else:
        benchmark([*bench_argv, *_options(spec.get("argv"))], **hooks)


def run_passes(bench_argv, passes: list[dict], *, record: bool = False, eval_tasks: str | None = None, **hooks):
    """Several client passes against one server: the server (and its weights) load once.

    record: write each pass to its own .data file (`PassRecorder`) and push a ``pass_done`` pointer.
    """
    global current_pass

    for spec in passes:
        current_pass = spec["name"]
        print(f"=== pass {current_pass}", flush=True)
        if not record:
            _run_pass(bench_argv, spec, record=False, eval_tasks=eval_tasks, **hooks)
            continue

        recorder.open(spec)
        return_code = 1
        try:
            _run_pass(bench_argv, spec, record=True, eval_tasks=eval_tasks, **hooks)
            return_code = 0
        finally:
            recorder.close(return_code)
        # in the server's .data: where the pass went
        done = {"name": current_pass, "data": recorder.path, "return_code": return_code}
        if summary := recorder.last_summary:
            done["output_throughput"] = summary["output_throughput"]
        push_metric(pass_done=done)
    current_pass = None


def run_client(bench_argv, passes: list[dict] | None = None, **kwargs):
    """The given passes, or the single default benchmark when there are none (``None``)."""
    if passes is None:
        benchmark(bench_argv, **{k: v for k, v in kwargs.items() if k in ("samples_hook", "parser_hook")})
    else:
        run_passes(bench_argv, passes, **kwargs)


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------


class InferenceServerError(BaseException):
    pass


class InferenceServer:
    """Run an inference server (given its full command) and abort the
    benchmark if it exits abnormally."""

    def __init__(self, command: list[str], *, name: str = "server"):
        print("SERVER:", " ".join(command), flush=True)
        self.name = name
        self.command = command
        self.proc = subprocess.Popen(command)
        self.returncode: int | None = None
        self._failed = threading.Event()
        self._watch = threading.Thread(target=self._monitor, daemon=True)
        self._watch.start()

    def _monitor(self):
        self.returncode = self.proc.wait()
        if self.returncode != 0:
            print(
                f"\n[ERROR] {self.name} exited with code {self.returncode}",
                file=sys.stderr,
                flush=True,
            )
            self._failed.set()

    def check(self):
        if self._failed.is_set():
            raise InferenceServerError(
                f"{self.name} exited early (code {self.returncode})"
            )

    def shutdown(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.shutdown()
        return False


def split_args(argv, *, skip_program: bool = False):
    """Split `argv` on the first `--` into (server_argv, bench_argv)."""
    sep = len(argv)
    for i, arg in enumerate(argv):
        if arg == "--":
            sep = i
            break

    start = 1 if skip_program else 0
    server_argv = [
        # voir already has a --config argument so we have to rename it
        "--config" if arg == "--wth-config" else arg
        for arg in argv[start:sep]
    ]
    return server_argv, argv[(sep + 1):]


def run_benchmark_watched(server, bench_argv, passes: list[dict] | None = None, **kwargs):
    """Run the bench client (`run_client`) in a thread, aborting if the server dies."""
    bench_error: list[BaseException] = []

    def target():
        try:
            run_client(bench_argv, passes, **kwargs)
        except BaseException as exc:
            bench_error.append(exc)

    bench_thread = threading.Thread(target=target, daemon=True)
    bench_thread.start()

    while bench_thread.is_alive():
        server.check()
        bench_thread.join(timeout=0.5)

    server.check()

    if bench_error:
        raise bench_error[0]


# ---------------------------------------------------------------------------
# Client-only entry point
# ---------------------------------------------------------------------------


def main(argv=None, **kwargs):
    """`benchserve` — benchmate-instrumented `vllm bench serve` client.

    Client only: the server is already running (or milabench runs it, see milabench.commands.serving).

        benchserve [--mode eval|perf|all] [--pass NAME] --base-url http://127.0.0.1:8000 --model <hf-id> \
            --dataset-name random --num-prompts 40 --random-input-len 1024 ...

    kwargs go to `run_client` (eval_tasks, samples_hook, parser_hook).
    """
    argv = sys.argv[1:] if argv is None else list(argv)

    parser = ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--mode", choices=["eval", "perf", "all"], default="all")
    parser.add_argument("--pass", dest="pass_name", default=None, help="run only this client pass")
    args, bench_argv = parser.parse_known_args(argv)

    observer, bench_monitor = make_observer()

    with bench_monitor():
        run_client(bench_argv, client_passes(args.mode, args.pass_name), **kwargs)
