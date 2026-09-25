"""Backend-agnostic LLM serving benchmark client.

Drives `vllm.benchmarks.serve.main()` against any OpenAI-compatible endpoint
(vLLM, llama.cpp llama-server, ...) and streams per-window / per-request
metrics to a benchmate observer installed via `set_metric_sink()`.

Can also run standalone as the `benchserve` console script — an
instrumented drop-in for `vllm bench serve` against any endpoint that is
already up (vLLM, llama.cpp llama-server, ...):

    benchserve --base-url http://127.0.0.1:8000 --model <hf-id> \
        --dataset-name random --num-prompts 40 --random-input-len 1024 ...

The vllm imports are deferred inside functions so that benchmate stays
importable in environments where vllm is not installed.
"""

import json
import os
import subprocess
import sys
import threading

from benchmate.timeline import timeline, TimelineConfig, _default_db_path

push_metric = None


def set_metric_sink(fn):
    """Install the observer callback used to record benchmark metrics."""
    global push_metric
    push_metric = fn


def _run_description() -> str:
    raw = os.environ.get("MILABENCH_CONFIG")
    if not raw:
        return "vllm"
    try:
        cfg = json.loads(raw)
    except json.JSONDecodeError:
        return "vllm"
    name = cfg.get("name") or ".".join(cfg.get("tag", []))
    model = (cfg.get("client") or {}).get("argv", {}).get("--model")
    if model:
        return f"{name} ({model})"
    return name or "vllm"


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
    """Calculate the metrics for the benchmark.

    Side effect: pushes timeline windows and per-request metrics to the
    observer installed with `set_metric_sink()`.
    """

    # log_request(input_requests, outputs)

    config = TimelineConfig()
    description = _run_description()
    db_path = _default_db_path()
    print(f"[timeline] database path: {db_path}", flush=True)

    for sampled_obs in timeline(
        outputs,
        config=config,
        description=description,
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

            push_metric(ttfts=outputs[i].ttft, units="s")
            push_metric(e2els=outputs[i].latency, units="s")

            if len(outputs[i].itl) > 0:
                push_metric(itl=sum(outputs[i].itl)/len(outputs[i].itl), units="s")

            # push_metric(tpot=outputs[i].tpot, unit="ms")
            push_metric(input_tok=input_requests[i].prompt_len, units="count")
            push_metric(output_tok=output_len, units="count")

            tok_s = (input_requests[i].prompt_len + output_len) / outputs[i].latency
            push_metric(request_rate=tok_s, units="tok/s")

            completed += 1
        else:
            actual_output_lens.append(0)


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


def benchmark(argv):
    # vllm bench serve --model meta-llama/Meta-Llama-3-8B-Instruct --request-rate inf --dataset-name random --label milabench --backend openai --num-prompts 1000

    # vllm bench serve                                      \
    #     --backend openai                                  \
    #     --label milabench                                 \
    #     --model <your_model>                              \
    #     --dataset-name <dataset_name. Default 'random'>   \
    #     --request-rate inf                                \
    #     --num-prompts 1000
    from argparse import ArgumentParser
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

    original_get_samples = bench.get_samples

    def get_samples(args, tokenizer):
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

    parser = ArgumentParser()
    bench.add_cli_args(parser)

    print("BENCH:", " ".join(['vllm', 'bench', 'serve'] + argv))
    args = parser.parse_args(argv)

    bench.main(args)
    print("FINISHED")


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
        "--config" if arg == "--wth-config" else arg
        for arg in argv[start:sep]
    ]
    return server_argv, argv[(sep + 1):]


def _run_benchmark(bench_argv, error_box: list):
    try:
        benchmark(bench_argv)
    except BaseException as exc:
        error_box.append(exc)


def run_benchmark_watched(server, bench_argv):
    """Run the bench client in a thread, aborting if the server dies."""
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


def main(argv=None):
    """`benchserve` — benchmate-instrumented `vllm bench serve` client.

    Run against an already-running OpenAI-compatible server, e.g.

        benchserve --base-url http://127.0.0.1:8000 --model <hf-id> \
            --dataset-name random --num-prompts 40 --random-input-len 1024 ...
    """
    argv = sys.argv[1:] if argv is None else list(argv)

    import torchcompat.core as accelerator
    from benchmate.monitor import bench_monitor
    from benchmate.observer import BenchObserver
    from benchmate.toggles import get_observation_count

    observer = BenchObserver(
        accelerator.Event,
        earlystop=get_observation_count(120),
        batch_size_fn=lambda x: len(x[0]),
        raise_stop_program=False,
        stdout=True,
    )
    set_metric_sink(observer.record_metric)

    with bench_monitor():
        benchmark(argv)
