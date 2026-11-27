#!/usr/bin/env python
"""Release date of every model the report scores, for its quality-over-time chart: release_dates.json.

For each model's weights (the repos the report lists for it), follow the Hugging Face model card's base_model back
to the original repo (a quantization or GGUF points at what it quantized), and take that repo's creation date: the
day the original weights were published. Run it when new models are added; the report only reads the JSON.

  python3 scripts/homelab/release_dates.py <run dir>      (HF_TOKEN or $MILABENCH_BASE/data/token for gated repos)
"""
import json
import os
import sys
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "release_dates.json")


def token():
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    path = os.path.join(os.environ.get("MILABENCH_BASE", "/opt/milabench"), "data", "token")
    return open(path).read().strip() if os.path.exists(path) else None


def info(repo, tok, cache):
    if repo not in cache:
        req = urllib.request.Request(f"https://huggingface.co/api/models/{repo}")
        if tok:
            req.add_header("Authorization", f"Bearer {tok}")
        try:
            cache[repo] = json.load(urllib.request.urlopen(req, timeout=30))
        except (urllib.error.URLError, ValueError, TimeoutError):
            cache[repo] = None
    return cache[repo]


def origin(repo, tok, cache, depth=0):
    """The repo at the root of the base_model chain, and its info."""
    d = info(repo, tok, cache)
    if d is None or depth > 5:
        return repo, d
    card = d.get("cardData") or {}
    base = card.get("base_model")
    if isinstance(base, list):
        base = base[0] if base else None
    # only through quantizations (and GGUF conversions): a fine-tune is a release of its own
    quant = card.get("base_model_relation") == "quantized" or "gguf" in (d.get("tags") or []) or any(
        t in repo.lower() for t in ("gguf", "awq", "fp8", "nvfp4", "w4a16", "w8a8", "int4", "int8", "mxfp4", "gptq", "bnb", "-4bit"))
    if base and base != repo and quant:
        root, rd = origin(base, tok, cache, depth + 1)
        if rd is not None:
            return root, rd
    return repo, d


def main(run_dir):
    sys.path.insert(0, HERE)
    import report_html as h

    rows = json.load(open(os.path.join(run_dir, "report", "report.json")))
    rows = rows["rows"] if isinstance(rows, dict) else rows
    models = h.model_rows(rows)
    known = json.load(open(OUT)) if os.path.exists(OUT) else {}
    tok, cache = token(), {}
    for m in models:
        key = h.base_weights(m)
        if key in known or not m.get("models"):
            continue
        # the served weights first: a drafter (MTP/DFlash "-assistant") or a tokenizer repo comes after them
        root, d = origin(m["models"][0], tok, cache)
        if d is None or not d.get("createdAt"):
            print(f"{key}: no date ({m['models'][0]})")
            continue
        known[key] = {"repo": root, "date": d["createdAt"][:10]}
        print(f"{key}: {known[key]['date']} ({root})")
    json.dump(dict(sorted(known.items())), open(OUT, "w"), indent=1)
    print(f"{len(known)} models -> {OUT}")


if __name__ == "__main__":
    main(sys.argv[1])
