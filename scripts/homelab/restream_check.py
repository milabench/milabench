#!/usr/bin/env python
"""Check qb_launch.py's restream (QB_RESTREAM=1) against a real server: the same prompts at temperature 0, once
straight to the server (plain, non-streamed) and once through QualityBench's proxy with restream on. The rebuilt
completion must match the plain one: content, reasoning, finish_reason, usage present.

Run with QualityBench's own interpreter, one request at a time (it borrows one server slot for ~1-2 min):

  /opt/milabench/data/benchserve-tools/qualitybench/bin/python scripts/homelab/restream_check.py \
      http://127.0.0.1:8000 [--model NAME] [--max-tokens 200] [--thinking on|off]

Exit 0 when every prompt matches. Greedy decoding on a batched server is not always bit-exact (a different
batch makes different float sums), so a mismatch is printed in full: a late divergence in otherwise equal text
is batching, an empty, cut or missing field is a bug.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "benchmate", "benchmate",
                                "benchserve"))
import qb_launch  # noqa: E402

PROMPTS = [
    "What is 17 * 23? Answer with the number only.",
    "Write a Python function is_prime(n) that returns True when n is prime. Code only.",
    "Name three primary colors, comma separated.",
    "Explain in two sentences why the sky is blue.",
]


def fields(doc):
    ch = (doc.get("choices") or [{}])[0]
    m = ch.get("message") or {}
    return {"content": m.get("content") or "", "reasoning": m.get("reasoning_content") or m.get("reasoning") or "",
            "finish_reason": ch.get("finish_reason"), "usage": bool(doc.get("usage"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("endpoint")
    ap.add_argument("--model")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--thinking", choices=["on", "off"], default=None)
    a = ap.parse_args()

    qb_launch._reply_window(600)
    qb_launch._restream()
    from qualitybench import http as qhttp
    from qualitybench import proxy

    up = a.endpoint.rstrip("/")
    up = up[:-3] if up.endswith("/v1") else up
    direct = qhttp.Client(up)
    model = a.model or direct.get("/v1/models")["data"][0]["id"]
    print(f"model {model} at {up}")
    bad = 0
    with proxy.Proxy(up, thinking=a.thinking) as px:
        via = qhttp.Client(px.url)
        for p in PROMPTS:
            msgs = [{"role": "user", "content": p}]
            kw = {"max_tokens": a.max_tokens, "temperature": 0}
            body = {"model": model, "messages": msgs, **kw}
            if a.thinking:
                body["chat_template_kwargs"] = {"enable_thinking": a.thinking == "on"}
            plain = fields(direct.post("/v1/chat/completions", body))
            rebuilt = fields(via.chat(model, msgs, **kw))
            same = plain == rebuilt
            bad += not same
            print(f"{'OK  ' if same else 'DIFF'} {p[:50]!r}  finish={rebuilt['finish_reason']} usage={rebuilt['usage']} "
                  f"chars={len(rebuilt['content'])}+{len(rebuilt['reasoning'])}")
            if not same:
                print("  plain:  ", json.dumps(plain)[:2000])
                print("  rebuilt:", json.dumps(rebuilt)[:2000])
    print("all match" if not bad else f"{bad} of {len(PROMPTS)} differ")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
