"""Vendor clients driving the served model; runs in the docparse-sdk venv (toolenv.py), called by client.py.

    python sdk_run.py --adapter mineru|paddleocr_vl --base-url http://127.0.0.1:8000 --model model \
        --concurrency 8 --warmup 8 --out answers.jsonl [--options JSON] page.png ...

    mineru         mineru-vl-utils' MinerUClient (http-client backend): its two-step extraction (the model finds
                   the layout on a thumbnail, then reads each block crop), json2md
    paddleocr_vl   paddleocr's PaddleOCRVL pipeline (pipeline_version from --options, default v1.6): PP-DocLayoutV3
                   layout on the CPU (paddlepaddle CPU wheel on aarch64), one request per region to the server,
                   the pipeline's own Markdown

Pages run ``concurrency`` at a time (threads; each SDK call is blocking HTTP). One JSON line per page in --out
({reply, error, e2el_s, completion_tokens: null}); the last stdout line is {wall_s, start_time, end_time}.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor


def mineru(base_url: str, model: str, options: dict):
    from mineru_vl_utils import MinerUClient
    from mineru_vl_utils.post_process import json2md
    from PIL import Image

    client = MinerUClient(backend="http-client", server_url=base_url, model_name=model, use_tqdm=False,
                          **options)

    def page(path: str) -> str:
        blocks = client.two_step_extract(Image.open(path).convert("RGB"))
        return json2md([dict(b) for b in blocks])

    return page


def paddleocr_vl(base_url: str, model: str, options: dict):
    from paddleocr import PaddleOCRVL

    pipeline = PaddleOCRVL(pipeline_version=options.get("pipeline_version", "v1.6"), vl_rec_backend="vllm-server",
                           vl_rec_server_url=f"{base_url}/v1", vl_rec_api_model_name=model, device="cpu")
    lock = threading.Lock()  # the layout model is not thread safe; the VLM requests of one page run in parallel

    def page(path: str) -> str:
        with lock:
            (res,) = list(pipeline.predict(path))
        return res._to_markdown(pretty=True)["markdown_texts"]

    return page


ADAPTERS = {"mineru": mineru, "paddleocr_vl": paddleocr_vl}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", choices=sorted(ADAPTERS), required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--options", default="{}")
    parser.add_argument("pages", nargs="+")
    args = parser.parse_args()
    page = ADAPTERS[args.adapter](args.base_url, args.model, json.loads(args.options))

    def one(path):
        start = time.perf_counter()
        try:
            reply, error = page(path), None
        except Exception as exc:  # noqa: BLE001 - one failed page is one empty page
            reply, error = "", f"{type(exc).__name__}: {exc}"
        return {"reply": reply, "error": error, "e2el_s": time.perf_counter() - start, "ttft_s": None,
                "prompt_tokens": None, "completion_tokens": None, "finish_reason": None}

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        if args.warmup:
            list(pool.map(one, [args.pages[i % len(args.pages)] for i in range(args.warmup)]))
        t0, start = time.time(), time.perf_counter()
        results = list(pool.map(one, args.pages))
        wall = time.perf_counter() - start
    with open(args.out, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"wall_s": wall, "start_time": t0, "end_time": t0 + wall}))


if __name__ == "__main__":
    main()
