"""Document parsing (page -> Markdown) client: milabench runs the model server (benchmarks/serving's Serving package,
see benchfile.py), this process sends the pages of one client pass (``--pass NAME``).

``docparse`` passes (client.py): ``speed`` (pages/min on the fixed page set at a given concurrency) and ``quality``
(olmOCR-bench, OmniDocBench, omni_parse_bench French and handwriting slices, each with its official scorer).
Any other pass kind (vLLM's serving benchmark, lm-eval, QualityBench) goes to benchmate.benchserve unchanged.
"""

import os
import sys
from argparse import ArgumentParser

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmate import benchserve  # noqa: E402

import client  # noqa: E402


def run_docparse(bench_argv, spec):
    benchserve.current_pass = spec["name"]
    folder, name = benchserve._pass_output("docparse", False)
    host = benchserve._flag(bench_argv, "--host", "127.0.0.1")
    port = benchserve._flag(bench_argv, "--port", "8000")
    benchserve.wait_ready(f"http://{host}:{port}/v1/models",
                          float(benchserve._flag(bench_argv, "--ready-check-timeout-sec", 3600)))
    model = benchserve._flag(bench_argv, "--served-model-name") or benchserve._flag(bench_argv, "--model")
    client.run(f"http://{host}:{port}", model, spec["docparse"] or {}, output_dir=folder,
               name=name, pass_name=spec["name"], push=benchserve.push_metric)


def main(argv):
    argv = list(argv[1:])
    parser = ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--mode", default="all")
    parser.add_argument("--pass", dest="pass_name", default=None)
    args, bench_argv = parser.parse_known_args(argv)

    if args.pass_name:
        spec = benchserve.client_passes("all", args.pass_name)[0]
        if "docparse" in spec:
            bench_argv = benchserve.later_pass_ready_timeout(bench_argv, args.pass_name)
            _, monitor = benchserve.make_observer()
            with monitor():
                run_docparse(bench_argv, spec)
            return

    benchserve.main(argv)


if __name__ == "__main__":
    main(sys.argv)
