"""FineWeb subsets for pretraining benchmarks.

FineWeb is ~15T tokens; benchmarks only need enough text to keep the trainer
fed for warmup plus a few minutes of steady state. Instead of downloading
whole shards we read the leading parquet row groups (~1000 docs / ~2MB each)
of the shard files over HTTP range requests until ``samples`` documents are
fetched. Docs average ~660 tokens with GPT-2 or Qwen3.x tokenizers.

Output layout (consumable by ``datasets``, pyarrow, polars, ...)::

    <dest>/
        part-00000.parquet ...   # columns: text, id, token_count
        manifest.json            # docs, tokens, source revision/files
        .hf_cache/               # Arrow cache built by ``load_fineweb``

Typical use from a benchmark's prepare.py::

    from benchmate.fineweb import prepare_fineweb

    prepare_fineweb(data_dir / "fineweb" / "sample-10BT", samples=200_000)
"""

from __future__ import annotations

import json
import os
import shutil
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path

REPO_ID = "HuggingFaceFW/fineweb"

# Subset name -> shard glob inside the dataset repo. The samples are random
# draws over all CommonCrawl dumps, so their first row groups are already
# diverse.
SUBSETS = {
    "sample-10BT": "sample/10BT/*.parquet",
    "sample-100BT": "sample/100BT/*.parquet",
    "sample-350BT": "sample/350BT/*.parquet",
}

COLUMNS = ["text", "id", "token_count"]

_DOCS_PER_PART = 200_000
_ROW_GROUPS_PER_CHUNK = 16
_MANIFEST = "manifest.json"


def read_manifest(dest) -> dict | None:
    path = Path(dest) / _MANIFEST
    if not path.exists():
        return None
    return json.loads(path.read_text())


def prepare_fineweb(
    dest,
    samples: int,
    subset: str = "sample-10BT",
    revision: str = "main",
    token: str | None = None,
    warm_cache: bool = True,
    workers: int = 16,
) -> dict:
    """Materialize the first ``samples`` documents of a FineWeb subset into ``dest``.

    Idempotent: returns immediately if ``dest`` already holds at least
    ``samples`` documents of the same subset, so benchmarks sharing ``dest``
    get the largest subset any of them asked for. Writes into ``<dest>.tmp``
    and swaps it in at the end so an interrupted prepare never leaves a
    partial dataset that looks complete.
    """
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, HfFileSystem

    dest = Path(dest)
    existing = read_manifest(dest)
    if existing and existing["subset"] == subset and existing["docs"] >= samples:
        print(f"FineWeb {subset}: {existing['docs']:,} docs already at {dest}")
        return existing

    token = token or os.environ.get("MILABENCH_HF_TOKEN") or os.environ.get("HF_TOKEN")
    sha = HfApi(token=token).dataset_info(REPO_ID, revision=revision).sha
    fs = HfFileSystem(token=token)
    files = sorted(fs.glob(f"datasets/{REPO_ID}@{sha}/{SUBSETS[subset]}"))
    if not files:
        raise RuntimeError(f"No parquet files found for FineWeb subset {subset!r}")

    tmp = dest.with_name(dest.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)

    print(f"Fetching {samples:,} docs of {REPO_ID}/{subset}@{sha[:8]} -> {dest}")
    total_tokens = total_docs = part_docs = 0
    parts, used_files = [], []
    writer = None
    chunks = _row_group_chunks(fs, files)
    with ThreadPoolExecutor(workers) as pool:
        # Bounded read-ahead, consumed in order so the subset is always the
        # same dataset prefix regardless of network timing.
        pending = deque(
            pool.submit(_read_chunk, fs, *c) for c in islice(chunks, 2 * workers)
        )
        while pending and total_docs < samples:
            file, table = pending.popleft().result()
            for c in islice(chunks, 1):
                pending.append(pool.submit(_read_chunk, fs, *c))

            table = table.slice(0, samples - total_docs)
            if writer is None or part_docs >= _DOCS_PER_PART:
                if writer is not None:
                    writer.close()
                parts.append(f"part-{len(parts):05d}.parquet")
                writer = pq.ParquetWriter(tmp / parts[-1], table.schema, compression="zstd")
                part_docs = 0
            writer.write_table(table)
            part_docs += table.num_rows
            total_docs += table.num_rows
            total_tokens += pc.sum(table.column("token_count")).as_py()
            name = file.split(f"@{sha}/", 1)[-1]
            if name not in used_files:
                used_files.append(name)
            print(f"  {total_docs:,} / {samples:,} docs", flush=True)
        for fut in pending:
            fut.cancel()
    writer.close()

    if total_docs < samples:
        print(f"Warning: {subset} exhausted at {total_docs:,} < {samples:,} docs")

    manifest = {
        "repo_id": REPO_ID,
        "subset": subset,
        "revision": sha,
        "source_files": used_files,
        "docs": total_docs,
        "tokens": total_tokens,
        "token_unit": "gpt2",
        "parts": parts,
    }
    (tmp / _MANIFEST).write_text(json.dumps(manifest, indent=2))

    shutil.rmtree(dest, ignore_errors=True)
    tmp.rename(dest)
    print(f"FineWeb {subset}: {total_docs:,} docs / {total_tokens:,} tokens at {dest}")

    if warm_cache:
        load_fineweb(dest)
    return manifest


def _row_group_chunks(fs, files):
    """Yield ``(file, metadata, row_groups)`` in dataset order.

    Footers are read once per file and handed to the readers so each chunk
    only fetches its column chunks.
    """
    import pyarrow.parquet as pq

    for file in files:
        with fs.open(file, "rb") as f:
            metadata = pq.ParquetFile(f).metadata
        n = metadata.num_row_groups
        for start in range(0, n, _ROW_GROUPS_PER_CHUNK):
            yield file, metadata, list(range(start, min(start + _ROW_GROUPS_PER_CHUNK, n)))


def _read_chunk(fs, file, metadata, row_groups):
    import pyarrow.parquet as pq

    with fs.open(file, "rb") as f:
        table = pq.ParquetFile(f, metadata=metadata).read_row_groups(row_groups, columns=COLUMNS)
    return file, table


def load_fineweb(dest, split: str = "train"):
    """Load a prepared subset as a map-style ``datasets.Dataset``.

    The Arrow cache is pinned inside ``dest`` so it is shared between prepare
    and run regardless of HF_HOME, and ranks never race to build it.
    """
    from datasets import load_dataset

    dest = Path(dest)
    parts = read_manifest(dest)["parts"]
    return load_dataset(
        "parquet",
        data_files=[str(dest / p) for p in parts],
        split=split,
        cache_dir=str(dest / ".hf_cache"),
    )
