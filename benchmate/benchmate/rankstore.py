import glob
import json
import os
import socket
import time
from contextvars import ContextVar


def _hostname():
    """Short hostname, or None if it cannot be resolved."""
    try:
        return socket.gethostname().split(".")[0] or None
    except Exception:
        return None


def _benchname():
    """Name of the benchmark being run (from milabench's config), None outside milabench."""
    try:
        return json.loads(os.environ["MILABENCH_CONFIG"]).get("name") or None
    except Exception:
        return None


class LocalMetricStore:
    """JSONL metric store with one file per rank.

    ``path`` is a base name such as ``runs/<run>/metrics.<bench>.jsonl``; each
    rank appends to its own ``metrics.<bench>.<host>.<rank>.jsonl`` (without
    ``<host>`` when the hostname cannot be resolved) so no two processes ever write to
    the same file (append atomicity is not guaranteed on NFS/Lustre).
    ``read()`` merges the files of all ranks.
    """

    def __init__(self, path, rank=None, host=None):
        self.rank = int(os.getenv("RANK", 0)) if rank is None else rank
        self.host = _hostname() if host is None else host
        self.base = str(path)
        self.stem = self.base[: -len(".jsonl")] if self.base.endswith(".jsonl") else self.base
        tag = f"{self.host}.{self.rank}" if self.host else str(self.rank)
        self.path = f"{self.stem}.{tag}.jsonl"
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._file = open(self.path, "a", buffering=1, encoding="utf-8")

    def push(self, **metrics):
        """Append one record; ``rank`` and ``time`` are added if absent."""
        metrics.setdefault("time", time.time())
        self._file.write(json.dumps(metrics, default=str) + "\n")

    def files(self):
        return sorted(glob.glob(f"{glob.escape(self.stem)}.*.jsonl"))

    def _fields_of(self, path):
        """Fields encoded in a file name ``<stem>.<host>.<rank>.jsonl`` (no host in ``<stem>.<rank>.jsonl``)."""
        tag = path[len(self.stem) + 1 : -len(".jsonl")]
        host, _, rank = tag.rpartition(".")
        fields = {"host": host} if host else {}
        if rank.isdigit():
            fields["rank"] = int(rank)
        return fields

    def read(self):
        """Return the records of all ranks, ordered by time."""
        records = []
        for path in self.files():
            with open(path, "r", encoding="utf-8") as f:
                fields = self._fields_of(path)
                for line in f:
                    line = line.strip()
                    if line:
                        record = json.loads(line)
                        for key, value in fields.items():
                            record.setdefault(key, value)
                        records.append(record)
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
    # the run's own directory (runs/<run_name>), not the shared runs/ base
    runs = os.getenv("MILABENCH_DIR_RUN") or os.getenv("MILABENCH_DIR_RUNS", ".")
    name = _benchname()
    base = f"metrics.{name}.jsonl" if name else "metrics.jsonl"
    return os.getenv("MILABENCH_LOCAL_METRICS", os.path.join(runs, base))


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
