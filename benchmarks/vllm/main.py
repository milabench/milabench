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

import numpy as np
import torchcompat.core as accelerator
from vllm.benchmarks.serve import SampleRequest, RequestFuncOutput, BenchmarkMetrics, MILLISECONDS_TO_SECONDS_CONVERSION
from transformers import PreTrainedTokenizerBase
import vllm.benchmarks.datasets as datasets
from benchmate.timeline import timeline, TimelineConfig, _default_db_path
from benchmate.toggles import vllm_request_metrics_enabled
from server_backends import (
    InferenceServerError,
    build_server_command,
    resolved_server_backend,
    resolved_server_command,
)
from atom_aiter_compat import ensure_atom_aiter_compat

push_metric = None

# name of the client pass being run (client.passes), None for the single default pass
current_pass = None

# ITL is binned by the position of the token in the sequence (prompt + generated so far)
CONTEXT_BINS = [0, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 1 << 30]


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


def log_request(input_requests: list[SampleRequest], outputs: list[RequestFuncOutput]):
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
    input_requests: list[SampleRequest],
    outputs: list[RequestFuncOutput],
    dur_s: float,
    tokenizer: PreTrainedTokenizerBase,
    selected_percentiles: list[float],
    goodput_config_dict: dict[str, float],
) -> tuple[BenchmarkMetrics, list[int]]:
    """Calculate the metrics for the benchmark.

    Args:
        input_requests: The input requests.
        outputs: The outputs of the requests.
        dur_s: The duration of the benchmark.
        tokenizer: The tokenizer to use.
        selected_percentiles: The percentiles to select.
        goodput_config_dict: The goodput configuration.

    Returns:
        A tuple of the benchmark metrics and the actual output lengths.
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
    total_input = 0
    completed = 0
    good_completed = 0
    itls: list[float] = []
    tpots: list[float] = []
    all_tpots: list[float] = []
    ttfts: list[float] = []
    e2els: list[float] = []
    
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
            total_input += input_requests[i].prompt_len
            tpot = 0
            if output_len > 1:
                latency_minus_ttft = outputs[i].latency - outputs[i].ttft
                tpot = latency_minus_ttft / (output_len - 1)
                tpots.append(tpot)
            # Note: if output_len <= 1, we regard tpot as 0 for goodput
            all_tpots.append(tpot)
            itls += outputs[i].itl
            ttfts.append(outputs[i].ttft)
            e2els.append(outputs[i].latency)

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

            completed += 1
        else:
            actual_output_lens.append(0)

    push_pass_summary(input_requests, outputs, dur_s, actual_output_lens)


class GPQADiamond(datasets.HuggingFaceDataset):
    IS_MULTIMODAL = False
    SUPPORTED_DATASET_PATHS = {'hendrydong/gpqa_diamond'}

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
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


def benchmark(argv):
    # vllm bench serve --model meta-llama/Meta-Llama-3-8B-Instruct --request-rate inf --dataset-name random --label milabench --backend openai --num-prompts 1000

    # vllm bench serve                                      \
    #     --backend openai                                  \
    #     --label milabench                                 \
    #     --model <your_model>                              \
    #     --dataset-name <dataset_name. Default 'random'>   \
    #     --request-rate inf                                \
    #     --num-prompts 1000
    import vllm.benchmarks.serve as bench
    import vllm.benchmarks.datasets as datasets

    
    # datasets.InstructCoderDataset
    # datasets.BlazeditDataset
    #       Coding Task
    # https://huggingface.co/datasets/likaixin/InstructCoder

    # datasets.MTBenchDataset
    #       Open ended writing
    #  https://huggingface.co/datasets/philschmid/mt-bench

    # datasets.AIMODataset
    #       reasoning questions


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
           

    if getattr(bench, "_milabench_patched", False):
        # client passes call benchmark() several times: wrap vLLM's functions once, not once per pass
        return _bench_main(bench, argv)
    bench._milabench_patched = True

    original_get_samples = bench.get_samples
    def get_samples(args, tokenizer):
        match args.hf_name:
            case "openslr/librispeech_asr":
                return open_dataset(datasets.ASRDataset, args, tokenizer)
            
            case "hendrydong/gpqa_diamond":
                return open_dataset(GPQADiamond, args, tokenizer)

            case _:
                return original_get_samples(args, tokenizer)

    bench.get_samples = get_samples

    original = bench.calculate_metrics

    def new_calculate_metrics(*args, **kwargs):
        calculate_metrics(*args, **kwargs)
        return original(*args, **kwargs)

    bench.calculate_metrics = new_calculate_metrics
    return _bench_main(bench, argv)


def _bench_main(bench, argv):
    parser = ArgumentParser()
    bench.add_cli_args(parser)

    print("BENCH:", " ".join(['vllm', 'bench', 'serve'] + argv))
    args = parser.parse_args(argv)

    bench.main(args)
    print("FINISHED")


def prepare_voir():
    global push_metric 

    from benchmate.observer import BenchObserver
    from benchmate.monitor import bench_monitor
    from benchmate.toggles import get_observation_count

    observer = BenchObserver(
        accelerator.Event, 
        earlystop=get_observation_count(120),
        batch_size_fn=lambda x: len(x[0]),
        raise_stop_program=False,
        stdout=True,
    )

    def push(**kwargs):
        kwargs.setdefault("time", time.time())
        if not recorder.write(kwargs, task="train"):
            observer.record_metric(**kwargs)

    push_metric = push

    # the GPU monitor writes its samples (power, memory, load) straight to stdout: during a pass they go to the
    # pass's file instead, so its energy is that of its own requests
    import benchmate.monitor as monitor

    metric_log = monitor.metric_log

    def recorded_metric_log():
        log = metric_log()

        def route(data):
            if not recorder.write(data, task=data.get("task", "main")):
                log(data)

        return route

    monitor.metric_log = recorded_metric_log
    return observer, bench_monitor



class InferenceServer:
    """Run an inference server and abort the benchmark if it exits abnormally."""

    def __init__(self, argv, *, backend: str | None = None, command: list[str] | None = None):
        backend = backend or resolved_server_backend()
        server_args = build_server_command(argv, backend=backend, command=command)
        print("SERVER:", " ".join(server_args), flush=True)
        self.backend = backend
        self.proc = subprocess.Popen(server_args)
        self.returncode: int | None = None
        self._failed = threading.Event()
        self._watch = threading.Thread(target=self._monitor, daemon=True)
        self._watch.start()

    def _monitor(self):
        self.returncode = self.proc.wait()
        if self.returncode != 0:
            print(
                f"\n[ERROR] {self.backend} server exited with code {self.returncode}",
                file=sys.stderr,
                flush=True,
            )
            self._failed.set()

    def check(self):
        if self._failed.is_set():
            raise InferenceServerError(
                f"{self.backend} server exited early (code {self.returncode})"
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


# milabench's zero-shot tasks for reasoning models (milabench_gsm8k, milabench_arc_challenge, milabench_gpqa_diamond)
EVAL_TASKS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_tasks")


def wait_ready(url, timeout):
    import time
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


def run_eval(bench_argv, spec: dict):
    """lm-eval against the running server (OpenAI chat API); one ``eval_summary`` event per task."""
    host = _flag(bench_argv, "--host", "127.0.0.1")
    port = _flag(bench_argv, "--port", "8000")
    model = _flag(bench_argv, "--served-model-name") or _flag(bench_argv, "--model")
    tokenizer = _flag(bench_argv, "--tokenizer") or _flag(bench_argv, "--model")
    # the model's answers go next to the run's .data files (MILABENCH_DIR_RUNS is the root of every run)
    cfg = _milabench_config()
    runs = os.path.join(os.environ.get("MILABENCH_DIR_RUNS", "."), cfg.get("run_name", ""))
    output = os.path.join(runs, f"{cfg.get('name', 'vllm')}.{current_pass or 'eval'}.lm_eval")

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
            "--include_path", EVAL_TASKS,
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


class PassRecorder:
    """Each client pass as its own bench of the run: ``runs/<run>/<bench>.<pass>.<job>.data``.

    While a pass runs, its events (metrics pushed by this script and the GPU monitor's samples) go to that file
    only, between milabench-style config / meta / start ... end events; its bench name is ``<bench>.<pass>``,
    so ``milabench report`` lists every load point as its own row. What milabench records from stdout is the
    server's (``<bench>.server.nolog.<job>.data``, see benchfile.py): server start, weight load, warmup, the
    samples between passes, and a ``pass_done`` pointer per pass.
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
        tag = [bench, *(["nolog"] if "eval" in spec else []), str(cfg.get("job-number", 0))]
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


def run_passes(bench_argv, passes: list[dict]):
    """Several client passes against one server: the server (and its weights) load once."""
    global current_pass

    for spec in passes:
        current_pass = spec["name"]
        print(f"=== pass {current_pass}", flush=True)
        recorder.open(spec)
        return_code = 1
        try:
            if "eval" in spec:
                run_eval(bench_argv, spec["eval"])
            else:
                benchmark([*bench_argv, *_options(spec.get("argv"))])
            return_code = 0
        finally:
            recorder.close(return_code)
        # in the server's .data: where the pass went
        done = {"name": current_pass, "data": recorder.path, "return_code": return_code}
        if summary := recorder.last_summary:
            done["output_throughput"] = summary["output_throughput"]
        push_metric(pass_done=done)
    current_pass = None


def _run_benchmark(bench_argv, error_box: list):
    try:
        cfg = _milabench_config()
        arch = cfg.get("system", {}).get("arch", "cuda")
        variant = ((cfg.get("variants") or {}).get(arch) or {}).get("client") or {}
        passes = variant.get("passes") or (cfg.get("client") or {}).get("passes")
        if passes:
            run_passes(bench_argv, passes)
        else:
            benchmark(bench_argv)
    except BaseException as exc:
        error_box.append(exc)


def inference_server(argv):
    """Deprecated: use InferenceServer context manager."""
    return InferenceServer(argv).proc


def split_args(argv):
    sep = len(argv)
    for i, arg in enumerate(argv):
        if arg == "--":
            sep = i
            break

    server_argv = argv[1:sep]
    bench_argv = argv[(sep + 1):]

    args = []
    for arg in server_argv:
        # Voir already has a --config argument so we have to rename it
        if arg == "--wth-config":
            args.append("--config")
        else:
            args.append(arg)

    return args, bench_argv


def main(argv):
    global push_metric 

    ensure_atom_aiter_compat()

    # from argparse import ArgumentParser
    # parser = ArgumentParser()
    # parser.add_argument("--prepare", action="store_true", default=True)
    # args, argv = parser.parse_known_args(argv)

    server_argv, bench_argv = split_args(argv)

    observer, bench_monitor= prepare_voir()

    with bench_monitor() as log:
        with InferenceServer(server_argv) as server:
            bench_error: list[BaseException] = []
            bench_thread = threading.Thread(
                target=_run_benchmark,
                args=(bench_argv, bench_error),
                daemon=True,
            )
            bench_thread.start()

            while bench_thread.is_alive():
                server.check()
                bench_thread.join(timeout=0.5)

            server.check()

            if bench_error:
                raise bench_error[0]


if __name__ == "__main__":
    import sys

    # import debugpy
    # debugpy.listen(("0.0.0.0", 5678))

    # python -c "import debugpy; debugpy.connect(('localhost', 5678')); debugpy.breakpoint()"

    main(sys.argv)
