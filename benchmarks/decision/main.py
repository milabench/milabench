#!/usr/bin/env python
"""Decision-model client: milabench runs the System One server (engine_server.py, or a model's own server), this is
the client. Two pass kinds, both against ``POST /v1/systemone``:

    - name: di-quick
      decision: {rows: quick}        # Decision Index public suite through the kit's http engine, then its scorer:
                                     # quick = the stratified 6,000-request sample, full = all 140,178 requests,
                                     # or a rows file path. eval_summary per area + the index, with the per-request
                                     # latency the kit measured (one request at a time)
    - name: load-c8
      load: {concurrency: 8, requests: 400, rows: quick}
                                     # real suite requests, N in flight: decisions/s, questions/s, latency
                                     # percentiles (pass_summary, as the other serving passes)

Data: ``$MILABENCH_DECISION_INDEX`` (default ``$MILABENCH_DIR_DATA/decision-index``) holds ``suite-0.3/`` (the rebuilt
public suite) and ``sample-6000.jsonl.gz``. The kit runs from its own venv (benchmate.benchserve.tools,
"decision-index"); prepare.py installs it.
"""

from __future__ import annotations

import gzip
import json
import os
import random
import subprocess
import sys
import time
import urllib.error
import urllib.request
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor

from benchmate import benchserve
from benchmate.benchserve import tools

ROWS = {"quick": "sample-6000.jsonl.gz"}


def data_dir() -> str:
    return os.environ.get("MILABENCH_DECISION_INDEX") or os.path.join(
        os.environ.get("MILABENCH_DIR_DATA", os.path.expanduser("~/.cache/milabench")), "decision-index")


def rows_path(rows: str | None) -> str | None:
    """None: the full suite (the kit's --suite-dir); else a rows file."""
    if rows in (None, "full"):
        return None
    path = os.path.join(data_dir(), ROWS[rows]) if rows in ROWS else rows
    if not os.path.exists(path):
        raise SystemExit(f"decision rows {rows!r}: {path} is missing (rebuild the suite, see prepare.py)")
    return path


SUITE_FILES = ("selected-rows.jsonl.gz", "added-rows.jsonl.gz", "gsm8k-rows.jsonl.gz")


def suite_for(path: str | None) -> str:
    """The suite folder the kit scores against. The kit's scorer iterates over every row of its suite and counts the
    ones not answered as wrong, so a sample is scored against a copy of the suite cut to the sample's run ids
    (built once next to the full suite): the index of the sampled questions, with the same benchmark and area
    weighting, as THX-02's card scores its 6,018-request sample."""
    full = os.path.join(data_dir(), "suite-0.3")
    if path is None:
        return full
    stem = os.path.basename(path).split(".")[0]
    cut = os.path.join(data_dir(), f"suite-0.3-{stem}")
    if os.path.exists(os.path.join(cut, "manifest.json")):
        return cut
    import shutil

    keep = {r["_evaluation"]["run_id"] for r in read_rows(path)}
    tmp = cut + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    for name in os.listdir(full):
        src, dst = os.path.join(full, name), os.path.join(tmp, name)
        if name in SUITE_FILES:
            with gzip.open(dst, "wt", encoding="utf-8") as out:
                for row in read_rows(src):
                    if row["_evaluation"]["run_id"] in keep:
                        out.write(json.dumps(row, ensure_ascii=False) + "\n")
        else:
            shutil.copyfile(src, dst)
    os.replace(tmp, cut)
    return cut


def read_rows(path: str):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _runs_dir(name: str) -> str:
    cfg = benchserve._milabench_config()
    return os.path.join(os.environ.get("MILABENCH_DIR_RUNS", "."), cfg.get("run_name", ""), f"{name}.decision")


def _base_and_model(bench_argv):
    flag = benchserve._flag
    base = f"http://{flag(bench_argv, '--host', '127.0.0.1')}:{flag(bench_argv, '--port', '8000')}"
    model = flag(bench_argv, "--served-model-name") or flag(bench_argv, "--model") or "default"
    return base, model


# --- decision: the Decision Index kit -------------------------------------------------------------------


def _kit(*args) -> str:
    exe = tools.executable("decision-index")
    proc = subprocess.run([exe, *args], capture_output=True, text=True)
    sys.stdout.write(proc.stdout[-4000:])
    sys.stderr.write(proc.stderr[-4000:])
    if proc.returncode != 0:
        raise RuntimeError(f"decision-index {args[0]} failed with exit {proc.returncode}")
    return proc.stdout


def _last_json(text: str):
    """The kit prints one JSON document last (run: one line; score: indented)."""
    start = text.rfind("\n{")
    start = 0 if start < 0 and text.lstrip().startswith("{") else start + 1
    return json.loads(text[start:])


def _latency(results_path: str) -> dict:
    walls = [r["total_wall_ms"] for r in read_rows(results_path) if r.get("status") == "ok" and "total_wall_ms" in r]
    return {k.replace("e2el_ms", "latency_ms"): v for k, v in benchserve._percentiles("e2el_ms", walls).items()}


def run_decision_pass(bench_argv, spec, name, push):
    options = spec.get("decision") or {}
    base, model = _base_and_model(bench_argv)
    rows = options.get("rows", "quick")
    out = os.path.join(_runs_dir(name), spec.get("name", "decision"))
    os.makedirs(out, exist_ok=True)
    path = rows_path(rows)
    suite = suite_for(path)
    run_args = ["run", "--engine", "http", "--option", f"base_url={base}", "--option", f"model={model}",
                "--out", out, "--compact", "--suite-dir", suite]
    if path:
        run_args += ["--rows", path]
    if options.get("limit"):
        run_args += ["--limit", str(int(options["limit"]))]
    t = time.time()
    status = _last_json(_kit(*run_args))
    wall = time.time() - t
    results = os.path.join(out, "results.jsonl")
    scores = _last_json(_kit("score", "--results", results, "--suite-dir", suite, "--engine", model, "--out", out))
    common = {"name": name, "suite": "decision-index", "edition": scores.get("edition"), "rows": rows,
              "samples": status.get("completed"), "counts": status.get("counts"), "complete": scores.get("complete"),
              "results": results, "wall_s": wall, **_latency(results)}
    summaries = [{**common, "task": "index", "score": scores["decision_index"]}]
    for area in scores.get("areas", []):
        summaries.append({**common, "task": f"area:{area['id']}", "score": 100 * area["skill"], "raw": area.get("raw"),
                          "coverage": area.get("coverage"), "n": area.get("n")})
    for s in summaries:
        print("[eval]", json.dumps(s), flush=True)
        push(eval_summary=s)
    with open(os.path.join(out, "scores-brief.json"), "w") as f:
        json.dump(scores, f, indent=1)
    return summaries


# --- load: concurrent System One requests ---------------------------------------------------------------


def load_requests(path: str, n: int, seed: int = 608) -> list[dict]:
    rows = [{"state": r["state"], "questions": r["questions"]} for r in read_rows(path)]
    random.Random(seed).shuffle(rows)
    return [rows[i % len(rows)] for i in range(n)]


def _post(url: str, body: dict, model: str, timeout: float):
    data = json.dumps({"model": model, **body}).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    t = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp.read()
            ok = resp.status == 200
    except urllib.error.HTTPError as exc:
        ok = False if exc.code != 422 else None  # 422: over the model's input limit, not a failure of the server
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        ok = False
    return ok, time.perf_counter() - t, len(body["questions"])


def run_load_pass(bench_argv, spec):
    options = spec.get("load") or {}
    base, model = _base_and_model(bench_argv)
    concurrency = int(options.get("concurrency", 1))
    n = int(options.get("requests", 50 * concurrency))
    warm = int(options.get("warmup", concurrency))
    requests = load_requests(rows_path(options.get("rows", "quick")), n + warm)
    url = f"{base}/v1/systemone"
    timeout = float(options.get("timeout", 600))
    with ThreadPoolExecutor(concurrency) as pool:
        list(pool.map(lambda b: _post(url, b, model, timeout), requests[:warm]))
        t = time.perf_counter()
        results = list(pool.map(lambda b: _post(url, b, model, timeout), requests[warm:]))
        dur = time.perf_counter() - t
    ok = [r for r in results if r[0]]
    end = time.time()
    lat = [r[1] for r in ok]
    summary = {
        "name": benchserve.current_pass or spec.get("name", "load"),
        "kind": "decision",
        "start_time": end - dur, "end_time": end, "duration_s": dur,
        "completed": len(ok), "failed": sum(1 for r in results if r[0] is False),
        "unsupported": sum(1 for r in results if r[0] is None),
        "concurrency": concurrency,
        "request_throughput": len(ok) / dur if dur else 0.0,
        "questions_per_s": sum(r[2] for r in ok) / dur if dur else 0.0,
        "questions_per_request": sum(r[2] for r in ok) / len(ok) if ok else 0.0,
        "output_throughput": 0.0, "total_throughput": 0.0,
        **benchserve._percentiles("ttft_ms", lat, 1000),
        **benchserve._percentiles("e2el_ms", lat, 1000),
    }
    print("[pass]", json.dumps(summary), flush=True)
    benchserve.push_metric(pass_summary=summary)
    benchserve.recorder.last_summary = summary
    return summary


# --- entry ----------------------------------------------------------------------------------------------


def main(argv):
    argv = list(argv[1:])
    parser = ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--pass", dest="pass_name", default=None)
    args, rest = parser.parse_known_args(argv)
    specs = benchserve.client_passes("all", args.pass_name) if args.pass_name else (benchserve.client_passes("all") or [])
    if not specs:
        raise SystemExit("decision: no client passes configured (client.passes)")
    rest = benchserve.later_pass_ready_timeout(rest, args.pass_name)
    base, _ = _base_and_model(rest)
    cfg = benchserve._milabench_config()
    _, bench_monitor = benchserve.make_observer()
    with bench_monitor():
        benchserve.wait_ready(f"{base}/v1/models", float(benchserve._flag(rest, "--ready-check-timeout-sec", 3600)))
        for spec in specs:
            benchserve.current_pass = spec.get("name")
            if "decision" in spec:
                run_decision_pass(rest, spec, cfg.get("name", "decision"), benchserve.push_metric)
            elif "load" in spec:
                run_load_pass(rest, spec)
            else:
                raise SystemExit(f"decision: pass {spec.get('name')!r} has neither decision nor load")


if __name__ == "__main__":
    main(sys.argv)
