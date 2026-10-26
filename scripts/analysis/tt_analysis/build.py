"""Raw sources -> tidy tables (CSV) in ``data/tables/``.

Tables: runs, steps, ranks, node_health, comm, gpu_monitor. Each row carries
``cluster`` and ``synthetic``; runs keep failed attempts (``status``).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .health import parse_nccl_tests, parse_srygaard, parse_tt_health
from .runs import data_files, parse_run
from .topology import load_topology

# Traffic isolation of a run (comparisons only within one level):
#   alone      - the cluster is ours (window campaign): no traffic from other users anywhere;
#   reserved   - every node in our reservation and on one switch: other users' traffic never
#                shares our links;
#   shared     - the run crosses the spine, or its nodes are not reserved: other users' traffic
#                may share its links.
ISOLATION_LABEL = {"alone": "alone on the cluster", "reserved": "reserved, one switch",
                   "shared": "shared links (other users' traffic possible)"}


def isolation(row, reserved_jobs: set[str]) -> str:
    if str(row.get("campaign", "")).startswith("window"):
        return "alone"
    job = str(row.get("job_id", "")).removesuffix(".0")
    if job in reserved_jobs and row.get("n_switches") == 1:
        return "reserved"
    return "shared"


# Settings that must match for two runs to be repetitions of each other.
REPEAT_KEY = ["cluster", "campaign", "model", "n_layers", "seq_len", "mb_seqs", "grad_accum",
              "adam_moments", "precision", "ep", "gpus", "family", "nodes_per_leaf", "bench"]


def _sources(raw: Path, synthetic: Path | None):
    yield from ((d, False) for d in sorted(raw.iterdir()) if d.is_dir())
    if synthetic and synthetic.exists():
        yield from ((d, True) for d in sorted(synthetic.iterdir()) if d.is_dir())


def build(raw: Path, synthetic: Path | None, out: Path) -> dict[str, pd.DataFrame]:
    runs, steps, ranks, nodes, comm, monitor, ep = [], [], [], [], [], [], []
    for cdir, is_syn in _sources(raw, synthetic):
        cluster = cdir.name
        leaf_of = load_topology(cdir / "topology.txt")
        excluded_by_job = {}
        reserved_jobs = set()
        if (cdir / "reservations.csv").exists():  # jobs that ran in one of our reservations
            reserved_jobs = set(pd.read_csv(cdir / "reservations.csv").job_id.astype(str))
        for hdir in sorted((cdir / "health").glob("*")):
            n, c, excl = parse_tt_health(hdir, cluster, leaf_of)
            nodes += n
            comm += c
            excluded_by_job[hdir.name] = excl
        for sdir in sorted((cdir / "srygaard").glob("*")):
            n, c = parse_srygaard(sdir, cluster, leaf_of, synthetic=is_syn)
            nodes += n
            comm += c
        for f in sorted((cdir / "nccl_tests").glob("*/*.txt")):
            comm += parse_nccl_tests(f, cluster, leaf_of)
        for f in data_files(cdir):
            parsed = parse_run(f, cluster, leaf_of, synthetic=is_syn)
            if parsed is None:
                continue
            row, s, r, e = parsed
            ep += e
            if row.get("excluded_nodes") is None and (excl := excluded_by_job.get(str(row.get("job_id")))):
                row["excluded_nodes"] = ";".join(f"{h}:{why}" for h, why in excl.items())
            if row["status"] == "completed":
                row["isolation"] = isolation(row, reserved_jobs)
            runs.append(row)
            steps += s
            ranks += r
        for m in sorted((cdir / "dcgm").glob("*.csv")):
            df = pd.read_csv(m)
            df["cluster"], df["source"], df["synthetic"] = cluster, m.stem, is_syn
            monitor.append(df)

    tables = {
        "runs": pd.DataFrame(runs),
        "steps": pd.DataFrame(steps),
        "ranks": pd.DataFrame(ranks),
        "node_health": pd.DataFrame(nodes),
        "comm": pd.DataFrame(comm),
        "gpu_monitor": pd.concat(monitor) if monitor else pd.DataFrame(),
        "ep_groups": pd.DataFrame(ep),
    }
    r = tables["runs"]
    if not r.empty:
        r = r.sort_values("started").reset_index(drop=True)
        done = r["status"] == "completed"
        key = r[REPEAT_KEY].map(str).agg("|".join, axis=1)
        rep = r[done].groupby(key[done]).cumcount() + 1
        r["repetition"] = r["repetition"].fillna(rep)
        tables["runs"] = r
    out.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(out / f"{name}.csv", index=False)
    return tables


def load(out: Path) -> dict[str, pd.DataFrame]:
    tables = {}
    for name in ("runs", "steps", "ranks", "node_health", "comm", "gpu_monitor", "ep_groups"):
        path = out / f"{name}.csv"
        try:
            tables[name] = pd.read_csv(path)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            tables[name] = pd.DataFrame()
    return tables
