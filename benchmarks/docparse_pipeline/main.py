"""Classic OCR / document-conversion pipelines on the docparse page sets: no server, the engine runs here.

Same passes, page sets, scorers and metrics as benchmarks/docparse (its pages.py / scoring.py / client.py are
imported from there), so a pipeline and a served VLM land on the same chart:

    {mode: speed, concurrency: N}    the fixed 56-page set, N worker processes (each its own engine)
    {mode: quality, tier, tasks}     the scored slices, 8 workers

The engine and its options are the bench's ``docparse`` section: engine (tesseract | rapidocr | docling), device
(cpu | cuda), render (pages.render settings for engines that read an image), langs, ocr, force_ocr.
"""

import json
import multiprocessing
import os
import sys
from argparse import ArgumentParser

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(HERE, "..", "docparse"))

from benchmate import benchserve  # noqa: E402

import client  # noqa: E402  (benchmarks/docparse)
import pages  # noqa: E402
import pipelines  # noqa: E402


def answer(rec: dict, items: list[dict], concurrency: int, warmup: int):
    """(results, wall seconds, start epoch, end epoch): pages rendered first, then warmup pages, then timed."""
    import time

    engine = pipelines.ENGINES[rec["engine"]]
    if engine.needs == "image":
        jobs = [(pages.file_of(item), pages.render(item, rec.get("render"))) for item in items]
    else:
        jobs = [(pages.file_of(item), None) for item in items]
    # spawned workers import pipelines.py by name: put this folder on their path (voir, not main.py, is their
    # __main__)
    os.environ["PYTHONPATH"] = os.pathsep.join(p for p in (HERE, os.environ.get("PYTHONPATH")) if p)
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(concurrency, initializer=pipelines._init, initargs=(rec["engine"], rec)) as pool:
        if warmup:
            pool.map(pipelines._one, [jobs[i % len(jobs)] for i in range(warmup)], chunksize=1)
        t0, start = time.time(), time.perf_counter()
        results = pool.map(pipelines._one, jobs, chunksize=1)
        wall = time.perf_counter() - start
    return results, wall, t0, t0 + wall


def run(spec: dict, *, output_dir: str, name: str, pass_name: str) -> None:
    # the docparse client's pass logic, with this process as the engine
    client.answer = lambda base_url, model, rec, items, concurrency, warmup, out_dir: answer(rec, items, concurrency, warmup)
    rec = client.recipe()
    rec.setdefault("output", "markdown")
    client.recipe = lambda: rec
    client.run("", "", spec, output_dir=output_dir, name=name, pass_name=pass_name, push=benchserve.push_metric)


def main(argv):
    parser = ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--pass", dest="pass_name", required=True)
    args, _ = parser.parse_known_args(argv[1:])
    spec = benchserve.client_passes("all", args.pass_name)[0]
    benchserve.current_pass = spec["name"]
    folder, name = benchserve._pass_output("docparse", False)
    _, monitor = benchserve.make_observer()
    with monitor():
        run(spec["docparse"] or {}, output_dir=folder, name=name, pass_name=spec["name"])
    print(json.dumps({"pass": spec["name"], "done": True}))


if __name__ == "__main__":
    main(sys.argv)
