"""Embedding / reranking serving client: milabench runs the vLLM server (``--runner pooling``), this is the client.

Speed passes are ``vllm bench serve`` with a pooling backend (benchmate.benchserve, instrumented):

    --backend openai-embeddings --endpoint /v1/embeddings   N documents per request (--random-batch-size)
    --backend vllm-rerank       --endpoint /v1/rerank       one query + N documents per request

``--num-prompts`` counts documents (vLLM's convention for batched pooling requests): a pass sends
num_prompts / batch requests. Documents are real text (``--retrieval-text real``, default): OpenHermes cut to exactly
``--random-input-len`` tokens (benchmate.benchserve.text_corpus), the reranker's queries to
``--rerank-query-len`` tokens; ``--retrieval-text random`` keeps vLLM's random token ids.

vLLM computes pooling metrics in ``calculate_metrics_for_embeddings``, which benchserve does not wrap (it wraps the
generation ``calculate_metrics``): this client wraps it and pushes the same ``pass_summary`` event as the generation
passes (request/s, tokens/s, latency percentiles; ttft = end-to-end latency), plus ``docs_per_s``.

``retrieval`` passes (retrieval_eval.py) score retrieval quality on the same server (eval_summary).
"""

from __future__ import annotations

import json
import os
import sys
import time
from argparse import ArgumentParser
from types import SimpleNamespace

from benchmate import benchserve

import retrieval_eval

RERANK_BACKENDS = ("vllm-rerank",)


# --- requests ---------------------------------------------------------------------------------------------


def parser_hook(parser):
    group = parser.add_argument_group("retrieval")
    group.add_argument("--retrieval-text", choices=["real", "random"], default="real")
    group.add_argument("--rerank-query-len", type=int, default=32)


def samples_hook(args, tokenizer, datasets):
    """Batched real-text requests: a list of documents per request (embeddings), [query, *documents] (rerank)."""
    if getattr(args, "retrieval_text", "real") != "real":
        return None
    from benchmate.benchserve import text_corpus

    batch = max(1, int(args.random_batch_size))
    requests = max(1, args.num_prompts // batch)
    dataset = text_corpus.make_real_text_dataset(datasets)(dataset_path=text_corpus.build_corpus(), random_seed=args.seed)
    docs = dataset.sample(tokenizer=tokenizer, num_requests=requests * batch, input_len=args.random_input_len, output_len=1)
    queries = []
    if args.backend in RERANK_BACKENDS:
        queries = dataset.sample(tokenizer=tokenizer, num_requests=requests, input_len=args.rerank_query_len, output_len=1)
    samples = []
    for i in range(requests):
        group = docs[i * batch:(i + 1) * batch]
        prompt = [d.prompt for d in group]
        prompt_len = sum(d.prompt_len for d in group)
        if queries:
            # every (query, document) pair is one sequence for the cross-encoder
            prompt = [queries[i].prompt, *prompt]
            prompt_len += queries[i].prompt_len * len(group)
        samples.append(datasets.SampleRequest(prompt=prompt if batch > 1 or queries else prompt[0],
                                              prompt_len=prompt_len, expected_output_len=0,
                                              request_id=f"{args.request_id_prefix}{i}"))
    return samples


# --- metrics ----------------------------------------------------------------------------------------------


def push_pooling_summary(outputs, dur_s):
    """The pass_summary of a pooling pass (no tokens out): the generation fields the report reads, plus docs/s."""
    from benchmate.timeline import timeline, TimelineConfig

    ok = [o for o in outputs if o.success]
    try:
        for sampled in timeline(outputs, config=TimelineConfig(), description=benchserve._run_description(), persist=True):
            benchserve.push_metric(**sampled)
    except Exception as exc:  # the timeline is a convenience: the summary below is the result
        print(f"[retrieval] timeline skipped: {exc!r}", flush=True)
    docs = sum(o.num_input_sequences for o in ok)
    tokens = sum(o.prompt_len for o in ok)
    end = time.time()
    e2el = [o.latency for o in ok]
    summary = {
        "name": benchserve.current_pass or "default",
        "kind": "pooling",
        "start_time": end - dur_s,
        "end_time": end,
        "completed": len(ok),
        "failed": len(outputs) - len(ok),
        "duration_s": dur_s,
        "request_throughput": len(ok) / dur_s if dur_s else 0.0,
        "output_throughput": 0.0,
        "total_throughput": tokens / dur_s if dur_s else 0.0,
        "docs_per_s": docs / dur_s if dur_s else 0.0,
        "docs_per_request": docs / len(ok) if ok else 0.0,
        "mean_input_len": tokens / len(ok) if ok else 0.0,
        "mean_output_len": 0.0,
        **benchserve._percentiles("ttft_ms", e2el, 1000),
        **benchserve._percentiles("e2el_ms", e2el, 1000),
    }
    print("[pass]", json.dumps(summary), flush=True)
    benchserve.push_metric(pass_summary=summary)
    benchserve.recorder.last_summary = summary


def patch_pooling_metrics():
    import vllm.benchmarks.serve as bench

    if getattr(bench, "_retrieval_patched", False):
        return
    original = bench.calculate_metrics_for_embeddings

    def calculate_metrics_for_embeddings(outputs, dur_s, selected_percentiles):
        push_pooling_summary(outputs, dur_s)
        return original(outputs, dur_s, selected_percentiles)

    bench.calculate_metrics_for_embeddings = calculate_metrics_for_embeddings
    bench._retrieval_patched = True


# --- entry ------------------------------------------------------------------------------------------------


def run_retrieval_pass(bench_argv, spec):
    flag = benchserve._flag
    base = f"http://{flag(bench_argv, '--host', '127.0.0.1')}:{flag(bench_argv, '--port', '8000')}"
    model = flag(bench_argv, "--served-model-name") or flag(bench_argv, "--model")
    cfg = benchserve._milabench_config()
    name = f"{cfg.get('name', 'retrieval')}"
    runs = os.path.join(os.environ.get("MILABENCH_DIR_RUNS", "."), cfg.get("run_name", ""))
    output = os.path.join(runs, f"{name}.retrieval.json")
    _, bench_monitor = benchserve.make_observer()
    with bench_monitor():
        benchserve.wait_ready(f"{base}/v1/models", float(flag(bench_argv, "--ready-check-timeout-sec", 3600)))
        # client.retrieval: the model's defaults (query / document format), under the pass' own keys
        options = {**((cfg.get("client") or {}).get("retrieval") or {}), **(spec["retrieval"] or {})}
        retrieval_eval.run(base, model, options, name=name, output=output, push=benchserve.push_metric)


def main(argv):
    argv = list(argv[1:])
    parser = ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--pass", dest="pass_name", default=None)
    args, rest = parser.parse_known_args(argv)
    if args.pass_name:
        spec = (benchserve.client_passes("all", args.pass_name) or [{}])[0]
        if "retrieval" in spec:
            return run_retrieval_pass(benchserve.later_pass_ready_timeout(rest, args.pass_name), spec)
    patch_pooling_metrics()
    benchserve.main(argv, samples_hook=samples_hook, parser_hook=parser_hook)


if __name__ == "__main__":
    main(sys.argv)
