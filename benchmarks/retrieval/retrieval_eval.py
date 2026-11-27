"""``retrieval`` client passes: retrieval quality next to the speed passes, on the same server.

    passes:
      - name: retrieval
        retrieval:
          mode: embed               # embed: /v1/embeddings, exact cosine search over the whole corpus
                                    # rerank: /v1/rerank of a BM25 top-k (first stage computed here, no model)
          datasets: [scifact, nfcorpus]
          query_prompt: "Instruct: {task}\\nQuery: {query}"   # {task}: the dataset's instruction, {query}
          doc_prompt: "{title} {text}"                        # {title}, {text}
          batch: 64                 # documents per /v1/embeddings request
          workers: 4                # requests in flight
          rerank_top_k: 100         # rerank: BM25 candidates per query

Datasets: BEIR test splits as MTEB ships them (mteb/<name> on the Hugging Face hub), pinned revisions, the whole
corpus (no subsampling: the scores are comparable to MTEB's). Metrics: nDCG@10 (trec_eval's linear gain, as
BEIR / MTEB), recall@100, MRR@10. For rerank, the BM25 first stage is scored too (bm25_ndcg@10): the reranker's
gain over the same candidates. This BM25 is a plain one (lowercase word tokens, no stemming), weaker than
BEIR's Elasticsearch baseline; it is the same for every reranker.

One ``eval_summary`` event per dataset (suite "retrieval"), the per-query scores in
``<runs>/<run>/<bench>.retrieval.json``. Embedding mode also times the corpus indexing (index_docs_per_s): real
documents of real lengths, at ``batch`` x ``workers``.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

# BEIR test splits from MTEB's copies, pinned
DATASETS = {
    "scifact": ("mteb/scifact", "cf10ab6856b15b0e670ef8ae5dae4e266c12d035"),
    "nfcorpus": ("mteb/nfcorpus", "52ac3f19d3449632d9f00aab0ad34a110fc03816"),
}

# MTEB's retrieval instructions for these tasks (the ones Qwen3-Embedding / e5-mistral / harrier evaluate with)
INSTRUCTIONS = {
    "scifact": "Given a scientific claim, retrieve documents that support or refute the claim",
    "nfcorpus": "Given a question, retrieve relevant documents that best answer the question",
}

FILES = ("corpus.jsonl", "queries.jsonl", "qrels/test.jsonl")


def _download(name):
    from huggingface_hub import hf_hub_download

    repo, revision = DATASETS[name]
    return {f: hf_hub_download(repo, f, repo_type="dataset", revision=revision) for f in FILES}


def prepare(spec: dict):
    """Download the datasets of a retrieval pass into the HF cache (the runs are offline)."""
    for name in spec.get("datasets") or ["scifact"]:
        paths = _download(name)
        print(f"retrieval dataset {name}: {paths['corpus.jsonl']}")


def _jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load(name):
    paths = _download(name)  # cached: no network once prepared
    corpus = {str(d["_id"]): (d.get("title") or "", d.get("text") or "") for d in _jsonl(paths["corpus.jsonl"])}
    qrels = defaultdict(dict)
    for row in _jsonl(paths["qrels/test.jsonl"]):
        score = int(float(row["score"]))
        if score > 0:
            qrels[str(row["query-id"])][str(row["corpus-id"])] = score
    queries = {str(q["_id"]): q["text"] for q in _jsonl(paths["queries.jsonl"]) if str(q["_id"]) in qrels}
    return corpus, queries, dict(qrels)


# --- metrics -------------------------------------------------------------------------------------------


def ndcg_at(ranked, rels, k=10):
    dcg = sum(rels.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted(rels.values(), reverse=True)[:k]
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def recall_at(ranked, rels, k=100):
    return len(set(ranked[:k]) & set(rels)) / len(rels) if rels else 0.0


def mrr_at(ranked, rels, k=10):
    for i, d in enumerate(ranked[:k]):
        if d in rels:
            return 1.0 / (i + 1)
    return 0.0


def score_run(run: dict, qrels: dict) -> tuple[dict, dict]:
    per_query = {q: {"ndcg@10": ndcg_at(r, qrels[q]), "recall@100": recall_at(r, qrels[q]), "mrr@10": mrr_at(r, qrels[q])}
                 for q, r in run.items()}
    n = max(1, len(per_query))
    means = {m: sum(s[m] for s in per_query.values()) / n for m in ("ndcg@10", "recall@100", "mrr@10")}
    return means, per_query


# --- BM25 first stage (rerank mode) ---------------------------------------------------------------------

_TOKEN = re.compile(r"\w+")


def _tokens(text):
    return _TOKEN.findall(text.lower())


def bm25_top(corpus: dict, queries: dict, k: int, k1: float = 0.9, b: float = 0.4) -> dict:
    """Okapi BM25 (Anserini's k1 / b) over the whole corpus: query id -> the top-k doc ids."""
    ids = list(corpus)
    docs = [Counter(_tokens(f"{t} {x}")) for t, x in (corpus[i] for i in ids)]
    lengths = [sum(d.values()) for d in docs]
    avg = sum(lengths) / max(1, len(lengths))
    df = Counter(term for d in docs for term in d)
    postings = defaultdict(list)
    for j, d in enumerate(docs):
        for term, tf in d.items():
            postings[term].append((j, tf))
    n = len(docs)
    run = {}
    for qid, text in queries.items():
        scores = defaultdict(float)
        for term in set(_tokens(text)):
            if term not in df:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            for j, tf in postings[term]:
                scores[j] += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * lengths[j] / avg))
        top = sorted(scores.items(), key=lambda x: -x[1])[:k]
        run[qid] = [ids[j] for j, _ in top]
    return run


# --- server calls ---------------------------------------------------------------------------------------


def _post(url, payload, timeout=3600):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def embed_all(base, model, texts, batch, workers):
    """Embeddings of ``texts`` (normalized float32 matrix) and the wall time / prompt tokens it took."""
    import numpy as np

    chunks = [texts[i:i + batch] for i in range(0, len(texts), batch)]

    def one(chunk):
        out = _post(f"{base}/v1/embeddings", {"model": model, "input": chunk, "truncate_prompt_tokens": -1})
        data = sorted(out["data"], key=lambda d: d["index"])
        return [d["embedding"] for d in data], (out.get("usage") or {}).get("prompt_tokens", 0)

    start = time.perf_counter()
    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(one, chunks))
    elapsed = time.perf_counter() - start
    vectors = np.asarray([v for vs, _ in results for v in vs], dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    return vectors, elapsed, sum(t for _, t in results)


def rerank_all(base, model, queries, candidates, corpus, doc_prompt, workers):
    def one(qid):
        docs = candidates[qid]
        if not docs:
            return qid, []
        texts = [doc_prompt.format(title=corpus[d][0], text=corpus[d][1]).strip() for d in docs]
        out = _post(f"{base}/v1/rerank", {"model": model, "query": queries[qid], "documents": texts,
                                          "truncate_prompt_tokens": -1})
        order = sorted(out["results"], key=lambda r: -r["relevance_score"])
        return qid, [docs[r["index"]] for r in order]

    start = time.perf_counter()
    with ThreadPoolExecutor(workers) as pool:
        run = dict(pool.map(one, list(queries)))
    return run, time.perf_counter() - start


# --- the pass -------------------------------------------------------------------------------------------


def run(base: str, model: str, spec: dict, *, name: str, output: str | None, push) -> list[dict]:
    mode = spec.get("mode", "embed")
    query_prompt = spec.get("query_prompt", "{query}")
    doc_prompt = spec.get("doc_prompt", "{title} {text}")
    workers = int(spec.get("workers", 4))
    summaries, details = [], {}
    for dataset in spec.get("datasets") or ["scifact"]:
        corpus, queries, qrels = load(dataset)
        summary = {"name": name, "suite": "retrieval", "task": dataset, "mode": mode, "samples": len(queries),
                   "corpus_docs": len(corpus), "results": output}
        if mode == "embed":
            import numpy as np

            ids = list(corpus)
            doc_texts = [doc_prompt.format(title=t, text=x).strip() for t, x in (corpus[i] for i in ids)]
            qids = list(queries)
            task = INSTRUCTIONS.get(dataset, "")
            q_texts = [query_prompt.format(task=task, query=queries[q]) for q in qids]
            batch = int(spec.get("batch", 64))
            docs, index_s, index_tokens = embed_all(base, model, doc_texts, batch, workers)
            qv, query_s, _ = embed_all(base, model, q_texts, batch, workers)
            scores = qv @ docs.T
            k = min(100, len(ids))
            top = np.argpartition(-scores, k - 1, axis=1)[:, :k]
            run_ = {}
            for row, qid in enumerate(qids):
                order = top[row][np.argsort(-scores[row, top[row]])]
                run_[qid] = [ids[j] for j in order]
            means, per_query = score_run(run_, qrels)
            summary.update({"scores": means, "index_seconds": index_s, "index_docs_per_s": len(ids) / index_s,
                            "index_tokens_per_s": index_tokens / index_s, "query_seconds": query_s,
                            "embedding_dim": int(docs.shape[1])})
        else:
            k = int(spec.get("rerank_top_k", 100))
            candidates = bm25_top(corpus, queries, k)
            bm25, _ = score_run(candidates, qrels)
            run_, rerank_s = rerank_all(base, model, queries, candidates, corpus, doc_prompt, workers)
            means, per_query = score_run(run_, qrels)
            summary.update({"scores": means, "first_stage": f"bm25 top-{k}",
                            "bm25": bm25, "rerank_seconds": rerank_s,
                            "rerank_pairs_per_s": sum(len(c) for c in candidates.values()) / rerank_s})
        print("[eval]", json.dumps(summary), flush=True)
        push(eval_summary=summary)
        summaries.append(summary)
        details[dataset] = {"summary": summary, "per_query": per_query}
    if output:
        os.makedirs(os.path.dirname(output), exist_ok=True)
        with open(output, "w") as f:
            json.dump(details, f, indent=1)
    return summaries
