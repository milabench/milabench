"""Single-node, CPU-only checks of the per-rank metric logging (no GPU, no torch, no allocation).

Ranks are simulated with one subprocess per RANK value.
"""
import json
import multiprocessing as mp
import os
import time

import pytest

from benchmate.rankstore import LocalMetricStore


class FakeEvent:
    def __init__(self, enable_timing=True):
        self.t = None

    def record(self):
        self.t = time.perf_counter()

    def synchronize(self):
        pass

    def elapsed_time(self, other):
        return (other.t - self.t) * 1000


def _rank_worker(rank, base, steps, batch):
    os.environ["RANK"] = str(rank)
    os.environ["MILABENCH_LOCAL_METRICS"] = base
    os.environ["VOIR_EARLYSTOP_COUNT"] = "1000"

    from benchmate.metrics import ManualTimedIterator

    pushed = []
    timer = ManualTimedIterator(
        [0] * steps,
        event_fn=FakeEvent,
        rank=rank,
        push=lambda **kw: pushed.append(kw),
        device=None,
        earlystop=1000,
        batch_size_fn=lambda b: 0,
    )
    for _ in timer:
        time.sleep(0.005)
        timer.step(batch_override=batch)
    timer._push()

    with open(f"{base}.rank0_push.{rank}.json", "w") as f:
        json.dump(pushed, f, default=str)


def _run_ranks(tmp_path, world=4, steps=6, batch=128):
    base = str(tmp_path / "metrics.jsonl")
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_rank_worker, args=(r, base, steps, batch)) for r in range(world)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(60)
        assert p.exitcode == 0
    return base


def test_store_one_file_per_rank_and_merged_read(tmp_path):
    base = str(tmp_path / "m.jsonl")
    stores = [LocalMetricStore(base, rank=r) for r in range(3)]
    for i in range(4):
        for s in stores:
            s.push(step_time=0.1 * i, time=i + s.rank / 10)
    for s in stores:
        s.close()

    assert len(stores[0].files()) == 3
    records = stores[0].read()
    assert len(records) == 12
    assert [r["time"] for r in records] == sorted(r["time"] for r in records)
    assert {r["rank"] for r in records} == {0, 1, 2}


def test_only_rank0_pushes_rate(tmp_path):
    base = _run_ranks(tmp_path)
    for rank in range(4):
        with open(f"{base}.rank0_push.{rank}.json") as f:
            pushed = json.load(f)
        rates = [p for p in pushed if "rate" in p]
        if rank == 0:
            assert rates, "rank 0 must report rate through the main push"
        else:
            assert not rates, f"rank {rank} must not report rate through the main push"


def test_workers_write_step_time_to_their_own_file(tmp_path):
    base = _run_ranks(tmp_path)
    store = LocalMetricStore(base, rank=0)
    store.close()
    names = [os.path.basename(f) for f in store.files() if ".rank0_push." not in f]
    assert names == [f"metrics.{r}.jsonl" for r in range(4)]

    records = store.read()
    by_rank = {}
    for r in records:
        by_rank.setdefault(r["rank"], []).append(r)
    for rank in (1, 2, 3):
        assert rank in by_rank, f"rank {rank} wrote nothing"
        assert any("step_time" in r for r in by_rank[rank])
        assert all(r["rank"] == rank for r in by_rank[rank])


# --------------------------------------------------------------------------------------
# BENCHMATE_RECORD_FINE=1: every fine grained timer (next / work / step), the overhead,
# the loader init time, laps and totals must reach either the main push (rank 0) or the
# rank store (workers).
# --------------------------------------------------------------------------------------

STEPS = 6


def _fine_worker(rank, base, kind, steps, batch):
    os.environ["RANK"] = str(rank)
    os.environ["MILABENCH_LOCAL_METRICS"] = base
    os.environ["BENCHMATE_RECORD_FINE"] = "1"
    os.environ["BENCHMATE_RECORD_OVERHEAD"] = "1"
    os.environ["BENCHMATE_RECORD_LAPS"] = "1"

    import benchmate.metrics as metrics

    pushed = []
    cls = metrics.ManualTimedIterator if kind == "manual" else metrics.TimedIterator
    timer = cls(
        list(range(steps)),
        event_fn=FakeEvent,
        rank=rank,
        push=lambda **kw: pushed.append(kw),
        device=None,
        earlystop=1000,
        batch_size_fn=lambda b: batch,
    )
    for _ in timer:
        time.sleep(0.003)
        if kind == "manual":
            timer.step(batch_override=batch)

    with open(f"{base}.push.{rank}.json", "w") as f:
        json.dump(pushed, f, default=str)


def _run_fine(tmp_path, kind, world=3):
    base = str(tmp_path / "metrics.jsonl")
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_fine_worker, args=(r, base, kind, STEPS, 128)) for r in range(world)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(60)
        assert p.exitcode == 0
    pushed = {}
    for r in range(world):
        with open(f"{base}.push.{r}.json") as f:
            pushed[r] = json.load(f)
    stored = {}
    for rec in LocalMetricStore(base, rank=0).read():
        stored.setdefault(rec["rank"], []).append(rec)
    return pushed, stored


def _names(records, key):
    return [r for r in records if key in r]


@pytest.mark.parametrize("kind", ["timed", "manual"])
def test_fine_rank0_main_push_has_everything(tmp_path, kind):
    pushed, _ = _run_fine(tmp_path, kind)
    rank0 = pushed[0]

    rates = _names(rank0, "rate")
    assert len(rates) == STEPS
    assert [r["batch_id"] for r in rates] == list(range(STEPS))
    assert all(r["rate"] > 0 and r["elapsed"] > 0 for r in rates)

    works = _names(rank0, "work")
    assert sorted(w["batch_id"] for w in works) == list(range(STEPS)), "one `work` timer per step"
    assert all(w["work"] >= 0 and w["units"] == "s" for w in works)

    assert _names(rank0, "__iter__"), "loader init time"
    assert _names(rank0, "overhead"), "BENCHMATE_RECORD_OVERHEAD=1"
    assert _names(rank0, "total_elapsed") or kind == "manual"
    laps = {k for r in rank0 for k in ("iter_create", "iter_start", "iter_end") if k in r}
    assert laps >= {"iter_create", "iter_start"}


def test_fine_timed_rank0_has_next_and_step_timers(tmp_path):
    pushed, _ = _run_fine(tmp_path, "timed")
    rank0 = pushed[0]
    # the first batch is prefetched (BENCHMATE_BATCH_FETCH) so it has no `next` timer
    assert sorted(e["batch_id"] for e in _names(rank0, "next")) == list(range(1, STEPS))
    assert sorted(e["batch_id"] for e in _names(rank0, "step")) == list(range(STEPS))


def test_fine_manual_rank0_has_step_timer(tmp_path):
    pushed, _ = _run_fine(tmp_path, "manual")
    assert sorted(e["batch_id"] for e in _names(pushed[0], "step")) == list(range(STEPS))


@pytest.mark.parametrize("kind", ["timed", "manual"])
def test_fine_workers_store_has_everything(tmp_path, kind):
    _, stored = _run_fine(tmp_path, kind)
    for rank in (1, 2):
        recs = stored[rank]
        assert all(r["rank"] == rank for r in recs)

        step_times = _names(recs, "step_time")
        assert [r["batch_id"] for r in step_times] == list(range(STEPS))
        assert all(r["step_time"] > 0 for r in step_times)
        assert not _names(recs, "rate"), "workers do not report rate"

        works = _names(recs, "work")
        assert sorted(w["batch_id"] for w in works) == list(range(STEPS)), "one `work` timer per step"
        assert _names(recs, "__iter__")
        assert _names(recs, "overhead")
        if kind == "timed":
            assert sorted(e["batch_id"] for e in _names(recs, "next")) == list(range(1, STEPS))
        assert sorted(e["batch_id"] for e in _names(recs, "step")) == list(range(STEPS))


@pytest.mark.parametrize("kind", ["timed", "manual"])
def test_fine_store_records_are_time_ordered_and_json_clean(tmp_path, kind):
    _, stored = _run_fine(tmp_path, kind)
    for recs in stored.values():
        assert all(isinstance(r["time"], float) for r in recs)
        # progress is a tuple in memory, must round trip through json
        for r in _names(recs, "progress"):
            assert len(r["progress"]) == 2


@pytest.mark.xfail(reason="rank 0 only goes to the main push, not to the per-rank store (docs/Design/metrics.rst level 2)", strict=True)
def test_rank0_also_in_store(tmp_path):
    _, stored = _run_fine(tmp_path, "timed")
    assert 0 in stored
