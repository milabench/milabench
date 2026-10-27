import glob
import json
import os
import time
from contextvars import ContextVar


class LocalMetricStore:
    """JSONL metric store with one file per rank.

    ``path`` is a base name such as ``runs/metrics.jsonl``; each rank appends
    to its own ``runs/metrics.<rank>.jsonl`` so no two processes ever write to
    the same file (append atomicity is not guaranteed on NFS/Lustre).
    ``read()`` merges the files of all ranks.
    """

    def __init__(self, path, rank=None):
        self.rank = int(os.getenv("RANK", 0)) if rank is None else rank
        self.base = str(path)
        self.stem = self.base[: -len(".jsonl")] if self.base.endswith(".jsonl") else self.base
        self.path = f"{self.stem}.{self.rank}.jsonl"
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._file = open(self.path, "a", buffering=1, encoding="utf-8")

    def push(self, **metrics):
        """Append one record; ``rank`` and ``time`` are added if absent."""
        metrics.setdefault("rank", self.rank)
        metrics.setdefault("time", time.time())
        self._file.write(json.dumps(metrics, default=str) + "\n")

    def files(self):
        return sorted(glob.glob(f"{glob.escape(self.stem)}.*.jsonl"))

    def read(self):
        """Return the records of all ranks, ordered by time."""
        records = []
        for path in self.files():
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        records.sort(key=lambda r: r.get("time", 0))
        return records

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


_store = ContextVar("milabench_local_metric_store", default=None)


def default_path():
    runs = os.getenv("MILABENCH_DIR_RUNS", ".")
    return os.getenv("MILABENCH_LOCAL_METRICS", os.path.join(runs, "metrics.jsonl"))


def init_store(path=None, rank=None):
    """Create the global store (replacing any previous one) and return it."""
    old = _store.get()
    if old is not None:
        old.close()
    store = LocalMetricStore(path or default_path(), rank=rank)
    _store.set(store)
    return store


def get_store():
    """Return the global store, creating it from the environment if needed."""
    store = _store.get()
    if store is None:
        store = init_store()
    return store


def ranked_push(**metrics):
    get_store().push(**metrics)
