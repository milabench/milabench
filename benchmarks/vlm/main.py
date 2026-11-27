"""Vision-language serving client: milabench runs the server (benchmarks/serving's Serving package, see benchfile.py).

Two kinds of client passes (``client.passes``):

* throughput passes: vLLM's serving benchmark through benchmate.benchserve, on real images
  (``--dataset-name custom_image --dataset-path vlm/perf-*.jsonl``, built by prepare.py; a relative path is
  read from $MILABENCH_DIR_DATA/vlm)
* ``vqa`` passes: accuracy and per-image latency on the pinned DocVQA / ChartQA / OCRBench / MMMU slices
  (vqa.py), reported as ``eval_summary`` events like the other accuracy passes
"""

import os
import sys
from argparse import ArgumentParser

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmate import benchserve  # noqa: E402

import vqa  # noqa: E402


def samples_hook(args, tokenizer, datasets):
    """vLLM's custom datasets with a path relative to the vlm data folder; the default lookup does the rest."""
    if args.dataset_name in ("custom_image", "custom") and args.dataset_path and not os.path.isabs(args.dataset_path):
        args.dataset_path = vqa.data_path(args.dataset_path)
    return None


def run_vqa(bench_argv, spec):
    benchserve.current_pass = spec["name"]
    folder, name = benchserve._pass_output("vqa", False)
    url = f"http://{benchserve._flag(bench_argv, '--host', '127.0.0.1')}:{benchserve._flag(bench_argv, '--port', '8000')}/v1/models"
    benchserve.wait_ready(url, float(benchserve._flag(bench_argv, "--ready-check-timeout-sec", 3600)))
    vqa.run(bench_argv, spec["vqa"] or {}, flag=benchserve._flag, output_dir=folder, name=name,
            push=benchserve.push_metric)


def main(argv):
    argv = list(argv[1:])
    parser = ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--mode", default="all")
    parser.add_argument("--pass", dest="pass_name", default=None)
    args, bench_argv = parser.parse_known_args(argv)

    if args.pass_name:
        spec = benchserve.client_passes("all", args.pass_name)[0]
        if "vqa" in spec:
            bench_argv = benchserve.later_pass_ready_timeout(bench_argv, args.pass_name)
            _, monitor = benchserve.make_observer()
            with monitor():
                run_vqa(bench_argv, spec)
            return

    benchserve.main(argv, samples_hook=samples_hook)


if __name__ == "__main__":
    main(sys.argv)
