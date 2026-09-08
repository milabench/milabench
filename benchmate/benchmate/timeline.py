from __future__ import annotations

import json
import math
import os
import statistics
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    Boolean,
    create_engine,
    func,
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.pool import NullPool

if False:
    @dataclass
    class RequestFuncOutput:
        """The output of the request function including metrics."""

        generated_text: str = ""
        success: bool = False
        latency: float = 0.0
        output_tokens: int = 0
        ttft: float = 0.0  # Time to first token
        itl: list[float] = field(default_factory=list)  # list of inter-token latencies
        tpot: float = 0.0  # avg next-token latencies
        prompt_len: int = 0
        error: str = ""
        start_time: float = 0.0


@dataclass
class TimelineConfig:
    num_buckets: int | None = 30
    bucket_duration: float | None = None

    input_token_weight: float = 1.0
    output_token_weight: float = 5.0

    latency_percentiles: tuple[float, ...] = (0.5, 0.90, 0.95, 0.99)
    track_latency: bool = True


# ---------------------------------------------------------------------------
#  SQLAlchemy persistence — store raw results once, replay with different configs
# ---------------------------------------------------------------------------

_Base = declarative_base()


class _Run(_Base):
    __tablename__ = "runs"

    run_id      = Column(Integer, primary_key=True, autoincrement=True)
    created_at  = Column(DateTime, default=datetime.utcnow, nullable=False)
    description = Column(String(512))
    config_json = Column(JSON)

    requests = relationship("_Request", back_populates="run", cascade="all, delete-orphan")


class _Request(_Base):
    __tablename__ = "requests"

    id             = Column(Integer, primary_key=True, autoincrement=True)
    run_id         = Column(Integer, ForeignKey("runs.run_id"), nullable=False, index=True)
    request_id     = Column(String(128))
    start_time     = Column(Float)
    latency        = Column(Float)
    prompt_len     = Column(Integer)
    output_tokens  = Column(Integer)
    ttft           = Column(Float)
    tpot           = Column(Float)
    itl_json       = Column(JSON)
    success        = Column(Boolean)
    error          = Column(Text)
    prompt         = Column(Text)
    generated_text = Column(Text)

    run = relationship("_Run", back_populates="requests")


def _default_db_path() -> Path:
    explicit = os.environ.get("MILABENCH_TIMELINE_DB")
    if explicit:
        return Path(explicit)

    runs_dir = os.environ.get("MILABENCH_DIR_RUNS")
    run_name = None
    cfg_raw = os.environ.get("MILABENCH_CONFIG")
    if cfg_raw:
        try:
            run_name = json.loads(cfg_raw).get("run_name")
        except json.JSONDecodeError:
            pass

    if runs_dir and run_name:
        if run_name[0] in ("/", ".", "~"):
            return Path(run_name).expanduser() / "benchmark_results.db"
        return Path(runs_dir) / run_name / "benchmark_results.db"
    if runs_dir:
        return Path(runs_dir) / "benchmark_results.db"
    return Path("benchmark_results.db")



class ResultStore:
    """Single entry-point for benchmark result persistence."""

    def __init__(self, db_path: str | Path | None = None):
        if db_path is None:
            db_path = _default_db_path()
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        url = f"sqlite:///{db_path}"
        # NullPool: a QueuePool's fixed capacity (default 5 + 10 overflow)
        # can be exhausted by a burst of concurrent requests against a
        # single shared, cached ResultStore (the dev dashboard server keeps
        # one per db path) and then hangs for pool_timeout before raising.
        # SQLite connections are cheap to open/close and don't benefit much
        # from pooling anyway, so just skip pooling instead of tuning its
        # size — the standard recommendation for SQLite + threaded servers.
        self.engine = create_engine(url, poolclass=NullPool)
        _Base.metadata.create_all(self.engine)
        self._migrate_schema()
        self.Session = sessionmaker(bind=self.engine)

    def _migrate_schema(self):
        """create_all() only creates missing tables, it never adds columns
        to a table that already exists — so a db written before a column
        was added to _Request (e.g. request_id) breaks every ORM query,
        since the ORM always selects every mapped column. Add whatever is
        missing (nullable, no data touched) so older dbs keep working.
        """
        from sqlalchemy import inspect, text

        inspector = inspect(self.engine)
        if _Request.__tablename__ not in inspector.get_table_names():
            return

        existing = {col["name"] for col in inspector.get_columns(_Request.__tablename__)}
        missing = [c for c in _Request.__table__.columns if c.name not in existing]
        if not missing:
            # No write-transaction needed — this is the common case on every
            # request after the first migration ever runs. Opening one
            # unconditionally here was acquiring SQLite's single writer lock
            # on every request; with several endpoints (buckets/report/
            # gantt/requests) firing concurrently on page load, that raced
            # and intermittently failed with "database is locked".
            return

        with self.engine.begin() as conn:
            for column in missing:
                col_type = column.type.compile(self.engine.dialect)
                conn.execute(text(
                    f'ALTER TABLE {_Request.__tablename__} ADD COLUMN "{column.name}" {col_type}'
                ))

    def save(
        self,
        outputs: list[dict],
        description: str = "",
        config: TimelineConfig | None = None,
    ) -> int:
        """Persist request metrics. Counts are stored; prompt/generated text are not."""
        config_json = asdict(config) if config else None

        run = _Run(description=description, config_json=config_json)
        for o in outputs:
            if not isinstance(o, dict):
                o = asdict(o)
            run.requests.append(_Request(
                request_id     = o.get("request_id"),
                start_time     = o.get("start_time"),
                latency        = o.get("latency"),
                prompt_len     = o.get("prompt_len"),
                output_tokens  = o.get("output_tokens"),
                ttft           = o.get("ttft"),
                tpot           = o.get("tpot"),
                itl_json       = o.get("itl", []),
                success        = bool(o.get("success", False)),
                error          = o.get("error", ""),
                prompt         = None,
                generated_text = None,
            ))

        with self.Session() as session:
            session.add(run)
            session.commit()
            run_id = run.run_id

        db_path = self.engine.url.database
        print(
            f"[timeline] saved {len(outputs)} requests -> {db_path} (run_id={run_id})",
            flush=True,
        )
        return run_id

    def load(self, run_id: int | None = None) -> list[dict]:
        """Load results for a run. Loads the latest run when run_id is None."""
        with self.Session() as session:
            if run_id is None:
                run_id = session.query(func.max(_Run.run_id)).scalar()
                if run_id is None:
                    return []

            rows = (
                session.query(_Request)
                .filter(_Request.run_id == run_id)
                .order_by(_Request.start_time)
                .all()
            )
            return [
                {
                    "request_id":     r.request_id or "",
                    "start_time":     r.start_time,
                    "latency":        r.latency,
                    "prompt_len":     r.prompt_len,
                    "output_tokens":  r.output_tokens,
                    "ttft":           r.ttft,
                    "tpot":           r.tpot,
                    "itl":            r.itl_json if r.itl_json else [],
                    "success":        bool(r.success),
                    "error":          r.error or "",
                }
                for r in rows
            ]

    def list_runs(self) -> list[dict]:
        with self.Session() as session:
            rows = (
                session.query(
                    _Run.run_id,
                    _Run.created_at,
                    _Run.description,
                    func.count(_Request.id).label("num_requests"),
                )
                .outerjoin(_Request)
                .group_by(_Run.run_id)
                .order_by(_Run.run_id)
                .all()
            )
            return [
                {
                    "run_id": r.run_id,
                    "created_at": str(r.created_at),
                    "description": r.description or "",
                    "num_requests": r.num_requests,
                }
                for r in rows
            ]


# ---------------------------------------------------------------------------
#  Bucket-count heuristic
# ---------------------------------------------------------------------------

# Minimum completions per bucket for a percentile to be stable.
# p99 of 128 samples is just the max — useless.  You need multiple
# waves of C completions per bucket for extreme percentiles.
SAMPLES_FOR_PERCENTILE = {
    0.50: 20,
    0.90: 50,
    0.95: 100,
    0.99: 500,
}


def _min_samples_for(percentile: float) -> int:
    """How many completions a bucket needs for a given percentile."""
    for p in sorted(SAMPLES_FOR_PERCENTILE):
        if percentile <= p:
            return SAMPLES_FOR_PERCENTILE[p]
    return SAMPLES_FOR_PERCENTILE[0.99]


def suggest_num_buckets(
    num_requests: int,
    concurrency: int = 1,
    highest_percentile: float = 0.99,
    min_buckets: int = 10,
    max_buckets: int = 200,
) -> int:
    """Suggest a bucket count given request count and concurrency.

    Requests complete in waves of ~concurrency.  Each bucket needs
    enough completions for the requested percentile to be stable:

        p50   — ~20  completions/bucket  (~0.2 waves @ C=128)
        p90   — ~50  completions/bucket  (~0.4 waves @ C=128)
        p99   — ~500 completions/bucket  (~4   waves @ C=128)

    With C=128 and N=3000 that is 23 waves total:
        p50 → up to 115 buckets  (23 / 0.2)
        p99 → up to   5 buckets  (23 / 4)

    Note: ITL is denser (output_tokens values per job) so its
    percentiles are more stable than TTFT/TPOT/E2E at the same
    bucket count.
    """
    min_per_bucket = _min_samples_for(highest_percentile)
    ideal = num_requests // min_per_bucket
    return max(min_buckets, min(max_buckets, ideal))


def suggest_num_requests(
    num_buckets: int = 30,
    concurrency: int = 1,
    highest_percentile: float = 0.99,
) -> dict:
    """How many requests the benchmark should send.

    Returns a dict with the minimum request count needed and the
    number of waves (batches of C concurrent requests) that implies.

    Example with C=128, B=30, targeting p99:
        min_per_bucket = 500
        num_requests   = 30 * 500 = 15000
        num_waves      = 15000 / 128 ≈ 117
    """
    min_per_bucket = _min_samples_for(highest_percentile)
    total = num_buckets * min_per_bucket
    num_waves = max(1, total // concurrency) if concurrency else total

    return {
        "num_requests": total,
        "num_waves": num_waves,
        "completions_per_bucket": min_per_bucket,
        "concurrency": concurrency,
        "num_buckets": num_buckets,
        "highest_percentile": highest_percentile,
    }


@dataclass
class Job:
    start: float
    end: float


class Timeline:
    def __init__(self, jobs: list[Job]):
        self.jobs: list[Job] = jobs 

    def pending(self, start, end=None):
        actively_running = []

        for job in self.jobs:
            if job.start <= start and job.end > start:
                actively_running.append(job)
        
        return job


@dataclass
class Worker:
    worker_id: int
    active_job = None
    job_count: int = 0

    def set_job(self, job):
        self.active_job = job

        if job is not None:
            self.job_count += 1
            self.active_job.worker = self.worker_id
            self.active_job.batch_id = self.job_count

    def end(self):
        if self.active_job:
            return self.active_job.end
        return 0


class JobAdapter:
    def __init__(self, dat):
        self.data = dat
        self.start = self.data["start_time"]
        self.end = self.start + self.data["latency"]
        self.pct = None
        self.worker = None
        self.accounted = 0
        self.batch_id = None

    def completion_percentage(self, start, end):
        total_time = self.end - self.start

        work_time_included_in_sampling = min(end, self.end) - max(start, self.start)
        
        percentage_done = max(min(work_time_included_in_sampling / total_time, 0), 1)
        return percentage_done

    def total_token(self, config: TimelineConfig | None = None):
        if config is None:
            total = self.data['output_tokens'] + self.data['prompt_len']
        else:
            total = (
                self.data['prompt_len'] * config.input_token_weight
                + self.data['output_tokens'] * config.output_token_weight
            ) / (config.input_token_weight + config.output_token_weight)

        if self.pct:
            return total * self.pct

        return total

    def token_per_second(self, config: TimelineConfig | None = None):
        if config is None:
            total = self.data['output_tokens'] + self.data['prompt_len']
        else:
            total = (
                self.data['prompt_len'] * config.input_token_weight
                + self.data['output_tokens'] * config.output_token_weight
            ) / (config.input_token_weight + config.output_token_weight)

        elapsed = self.data["latency"]
        return total / elapsed

    def __repr__(self):
        data = self.__json__()
        data.pop("generated_text", None)
        args = ", ".join([f"{k}={v}" for k, v in data.items()])
        return f"Job({args})"

    def __json__(self):
        return {
            **self.data,
            "start": self.start,
            "end": self.end,
            "worker": self.worker,
            "batch_id": self.batch_id
        }

def convert(obj):
    if isinstance(obj, dict):
        return obj
    return asdict(obj)


@dataclass
class PartialJob:
    job: None
    pct: float


def _percentile_sorted(sorted_values, pct):
    """Compute a single percentile from an already-sorted list."""
    if not sorted_values:
        return 0.0
    idx = pct * (len(sorted_values) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = idx - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def _latency_stats(values: list[float], percentiles: tuple[float, ...]) -> dict:
    if not values:
        return {}
    sv = sorted(values)
    out = {
        "mean": statistics.mean(sv),
        "min": sv[0],
        "max": sv[-1],
    }
    for p in percentiles:
        out[f"p{int(p * 100)}"] = _percentile_sorted(sv, p)
    return out


@dataclass
class Bucket:
    start: float
    end: float
    jobs: list[PartialJob] = None
    tokens: int = 0
    input_tokens: float = 0
    output_arrivals: int = 0
    # False for the "fake" context buckets a trim window adds outside
    # itself (see TimelineProcessor.method_2) — visualization-only, never
    # part of the official N in-window samples.
    in_window: bool = True

    def overlap(self, job):
        return max(0, min(job.end, self.end) - max(job.start, self.start))

    def active_jobs(self):
        return len(self.jobs)

    def active_jobs_pct(self):
        acc = 0
        bucket_duration = self.end - self.start
        for partial in self.jobs:
            acc += self.overlap(partial.job) / bucket_duration
        return acc

    def has_started_in_bucket(self, job):
        return self.start <= job.start <= self.end
    
    def has_finished_in_bucket(self, job):
        return self.start <= job.end <= self.end

    def ran_in_bucket(self, job):
        return job.start < self.start and job.end > self.end

    def _for_each(self, cond):
        acc = 0
        for partial in self.jobs:
            if cond(partial.job):
                acc += 1
        return acc

    def start_job_count(self):
        return self._for_each(self.has_started_in_bucket)

    def finished_job_count(self):
        return self._for_each(self.has_finished_in_bucket)
    
    def ran_through_job_count(self):
        return self._for_each(self.ran_in_bucket)

    def latency_summary(self, percentiles: tuple[float, ...]) -> dict:
        """Aggregate latency metrics based on when each event occurs.

        - TTFT: included if the first token arrived inside this bucket.
        - ITL:  each inter-token latency is included if the token it
          produced arrived inside this bucket.
        """
        ttfts = []
        itls = []

        for partial in self.jobs:
            job = partial.job
            data = job.data

            ttft = data.get("ttft", 0)
            first_token_time = job.start + ttft

            if ttft and self.start <= first_token_time <= self.end:
                ttfts.append(ttft)

            itl_list = data.get("itl")
            if itl_list:
                t = first_token_time
                for gap in itl_list:
                    t += gap
                    if t > self.end:
                        break
                    if t >= self.start:
                        itls.append(gap)

        result = {}
        for name, vals in (("ttft", ttfts), ("itl", itls)):
            stats = _latency_stats(vals, percentiles)
            for stat_name, v in stats.items():
                result[f"{name}_{stat_name}"] = v
        return result


class TimelineProcessor:
    def __init__(self, config: TimelineConfig | None = None):
        self.config = config or TimelineConfig()
        self.finished_jobs = []
        self.start = 0
        self.end = 0
        self.k = 0
        self.total = 0
        self.step = 0
        self.output = []
        self.avg = 0
        self.avg_instant = 0

    def _make_buckets(self):
        """Build buckets from config: either fixed count or fixed duration."""
        cfg = self.config
        duration = self.end - self.start

        if cfg.bucket_duration is not None:
            num = max(1, int(duration / cfg.bucket_duration))
            step = cfg.bucket_duration
        else:
            num = cfg.num_buckets or 30
            step = duration / num

        self.step = step
        self.samples = [self.start + step * (i + 1) for i in range(num)]
        return num

    def __call__(self, outputs: list[RequestFuncOutput], number=None, persist=True, window=None):
        if number is not None:
            self.config.num_buckets = number

        if not outputs:
            return []

        jobs = [JobAdapter(convert(l)) for l in outputs]
        jobs.sort(key=lambda item: item.start)

        if persist:
            self.save_normalized_data(jobs)

        return self.method_2(jobs, window=window)

    def save_normalized_data(self, jobs, db_path=None, description=""):
        outputs = [j.data for j in jobs]
        store = ResultStore(db_path)
        store.save(outputs, description=description, config=self.config)

    def method_2(self, jobs: list[Job], window: tuple[float, float] | None = None):
        cfg = self.config
        start = jobs[0].start
        for job in jobs:
            job.start -= start
            job.end -= start

            self.start = min(job.start, self.start)
            self.end = max(job.end, self.end)

        # Full extent of the (normalized) job list — always needed as the
        # outer bound for the "fake" context buckets below, regardless of
        # whether a window narrows where the N *official* buckets go.
        run_start, run_end = self.start, self.end

        # A degenerate/empty window (e.g. a "trim everything" sentinel where
        # a concurrency threshold cuts the whole run) has no steady-state
        # region left to place N clean buckets in. Fall back to spreading
        # cfg.num_buckets across the whole run so there's still something to
        # look at, with none of them counted as official.
        if window is not None and window[1] <= window[0]:
            window = None
            no_official_buckets = True
        else:
            no_official_buckets = False

        if window is not None:
            # The N buckets milabench actually samples from live INSIDE the
            # trim window, not wherever they happen to fall in a full-run
            # grid — the window drives their placement and width directly,
            # so all N are genuinely clean steady-state samples instead of
            # some being discarded after the fact.
            self.start, self.end = window

        number = self._make_buckets()

        buckets = [
            Bucket(self.start + i * self.step, self.start + (i + 1) * self.step, [])
            for i in range(number)
        ]
        if no_official_buckets:
            for b in buckets:
                b.in_window = False

        if window is not None:
            # "Fake" buckets: exactly one lump on each side covering
            # whatever's left of the run outside the trim window — pure
            # visualization context (so ramp-up/down isn't just cut off the
            # edge of the chart), never included in this method's
            # aggregate/report inputs. Not tiled at the official buckets'
            # width: that region can be an arbitrary, unrelated size, so one
            # bucket per side is however wide it needs to be.
            pre = [Bucket(run_start, self.start, [])] if self.start > run_start else []
            post = [Bucket(self.end, run_end, [])] if self.end < run_end else []

            for b in pre + post:
                b.in_window = False

            buckets = pre + buckets + post

        for job in jobs:
            for bucket in buckets:
                if job.start <= bucket.end and job.end >= bucket.start:
                    raw_total = job.end - job.start

                    if raw_total == 0:
                        print("Malformed job: ", job)

                    total = max(raw_total, 0.001)
                    overlap = max(0, min(job.end, bucket.end) - max(job.start, bucket.start))
                    bucket.jobs.append(PartialJob(job, overlap/total))

        for bucket in buckets:
            for partial in bucket.jobs:
                bucket.tokens += partial.job.total_token(cfg) * partial.pct
                partial.job.accounted += partial.pct

        for job in jobs:
            if job.accounted < 0.9999999:
                print("WARNING: Unaccounted job", job.accounted, job.start, job.end)

        # Input/output split: prompt tokens are only actually being processed
        # during prefill (start -> start+ttft), not smeared across decode too
        # — prompt_len/ttft is the local prefill rate. Output tokens have a
        # known exact arrival time per token (ttft, then +itl per token), so
        # count real arrivals per bucket instead of approximating.
        for job in jobs:
            ttft = job.data.get("ttft", 0) or 0
            prefill_end = job.start + ttft
            prefill_window = max(prefill_end - job.start, 0.001)

            for bucket in buckets:
                if job.start <= bucket.end and job.end >= bucket.start:
                    overlap = max(0, min(prefill_end, bucket.end) - max(job.start, bucket.start))
                    bucket.input_tokens += job.data.get("prompt_len", 0) * (overlap / prefill_window)

            output_tokens = job.data.get("output_tokens", 0) or 0
            if output_tokens > 0:
                def arrival_bucket(t):
                    # Buckets aren't necessarily uniform width (the "fake"
                    # context lumps outside a trim window can be much wider
                    # than the official buckets), so this can't be a direct
                    # index computation off a fixed step — scan for the
                    # bucket that actually contains t.
                    for b in buckets:
                        if b.start <= t <= b.end:
                            return b
                    return buckets[-1] if t > buckets[-1].end else buckets[0]

                t = prefill_end
                arrival_bucket(t).output_arrivals += 1
                for gap in (job.data.get("itl") or []):
                    t += gap
                    arrival_bucket(t).output_arrivals += 1

        self.output = []
        self.avg = 0
        for bucket in buckets:
            rate = bucket.tokens / (bucket.end - bucket.start)
            self.avg += rate
            entry = {
                "time": bucket.end,
                "start": bucket.start,
                "rate": rate,
                "input_rate": bucket.input_tokens / (bucket.end - bucket.start),
                "output_rate": bucket.output_arrivals / (bucket.end - bucket.start),
                "active_jobs": bucket.active_jobs(),
                "start_job": bucket.start_job_count(),
                "finished_job": bucket.finished_job_count(),
                "ran_through": bucket.ran_through_job_count(),
                "active_jobs_pct": bucket.active_jobs_pct(),
                "in_window": bucket.in_window,
            }
            if cfg.track_latency:
                entry.update(bucket.latency_summary(cfg.latency_percentiles))
            self.output.append(entry)
        
        return self.output

    def _on_time_change(self, now, workers):
        cfg = self.config
        while self.k < len(self.samples) and self.samples[self.k] <= now:
            start = self.step * self.k
            end   = start + self.step

            token = 0
            elapsed = self.step

            self.finished_jobs.sort(key=lambda job: job.end)

            while self.finished_jobs and self.finished_jobs[0].end <= end:
                job = self.finished_jobs.pop(0)
                token += job.total_token(cfg)
                job.accounted = True

            instant = 0
            for w in workers:
                if w.active_job:
                    instant += w.active_job.token_per_second(cfg)

            throughput = token / elapsed if elapsed > 0 else 0

            self.avg += throughput
            self.avg_instant += instant

            self.output.append({
                "rate": throughput,
                "time": self.samples[self.k],
                "instant": instant,
            })

            self.k += 1

    def _on_job_ended(self, job, workers):
        if job is None:
            end_time = self.end
        else:
            end_time = job.end
            self.finished_jobs.append(job)

        self._on_time_change(end_time, workers)

    def method_1(self, jobs: list[RequestFuncOutput]):
        start = jobs[0].start
        for job in jobs:
            job.start -= start
            job.end -= start

            self.start = min(job.start, self.start)
            self.end = max(job.end, self.end)

        self._make_buckets()
    
        workers = []
        for job in jobs:
            # Sort by workers tht will finish first
            workers.sort(key=lambda w: w.end())

            for worker in workers:
                # worker is going to finish 
                if worker.end() < job.start:
                    jb = worker.active_job
                    self._on_job_ended(jb, workers)
                    worker.set_job(job)
                    break

            else:
                w = Worker(len(workers))
                workers.append(w)
                w.set_job(job)

        workers.sort(key=lambda w: w.end())
        for worker in workers:
            jb = worker.active_job
            self._on_job_ended(jb, workers)
            worker.set_job(None)
        
        self._on_job_ended(None, workers)
            

        for job in jobs:
            if job.accounted < 0.999:
                print("MISSING")

        return jobs


# ---------------------------------------------------------------------------
#  Report helpers — vLLM-style whole-run aggregate vs bucket-rollup aggregate
# ---------------------------------------------------------------------------

def _dist_stats(values: list[float], percentiles=(0.5, 0.9, 0.95, 0.99)) -> dict | None:
    """Mean/std/median/percentiles over a raw distribution (no unit scaling).
    std is the population standard deviation (ddof=0), matching vLLM's own
    np.std(...) usage in calculate_metrics.
    """
    if not values:
        return None
    sv = sorted(values)
    n = len(sv)
    mean = sum(sv) / n
    variance = sum((x - mean) ** 2 for x in sv) / n
    return {
        "mean": mean,
        "std": variance ** 0.5,
        "median": _percentile_sorted(sv, 0.5),
        "percentiles": [(p, _percentile_sorted(sv, p)) for p in percentiles],
    }


def _milabench_style_stats(values: list[float], percentiles=(0.5, 0.9, 0.95, 0.99)) -> dict | None:
    """Distribution over the raw per-bucket `rate` sample stream that
    benchmarks/vllm/main.py pushes to milabench (`for sampled_obs in
    timeline(...): push_metric(**sampled_obs)`).

    This is NOT milabench's summary.py::_metrics() aggregation (sort, drop
    min/max as outliers, then summarize) — that pipeline only ever runs on
    metrics tagged `task="train"` (see aggregate()'s `k == "rate"` rename),
    and these bucket-rate pushes carry no task tag, so they never actually
    reach it. As things stand, milabench collects this stream but doesn't
    aggregate it at all, so this reports the plain, untrimmed distribution
    instead of pretending an aggregation step happens that doesn't.
    """
    xs = sorted(v for v in values if v is not None)
    if not xs:
        return None
    n = len(xs)
    mean = sum(xs) / n
    variance = sum((x - mean) ** 2 for x in xs) / n
    return {
        "mean": mean,
        "std": variance ** 0.5,
        "median": _percentile_sorted(xs, 0.5),
        "percentiles": [(p, _percentile_sorted(xs, p)) for p in percentiles],
        "min": xs[0],
        "max": xs[-1],
        "n": n,
    }


def _prefill_rates(rows: list[dict]) -> list[float]:
    """Per-request prefill throughput: prompt_len / ttft (the prefill phase's
    own duration), not a wall-clock system throughput.
    """
    rates = []
    for r in rows:
        ttft = r.get("ttft") or 0
        if ttft > 0:
            rates.append((r.get("prompt_len") or 0) / ttft)
    return rates


def vllm_style_report(
    outputs: list[dict],
    percentiles: tuple[float, ...] = (0.5, 0.9, 0.95, 0.99),
) -> dict:
    """Whole-run aggregate mirroring vllm/benchmarks/serve.py's
    calculate_metrics: every metric collapsed into one number for the
    entire run. Kept alongside bucket_aggregate_report() so the two
    computation styles are directly comparable on the same data.
    """
    rows = [convert(o) for o in outputs]
    successful = [r for r in rows if r.get("success")]
    failed = len(rows) - len(successful)
    if not successful:
        return {"completed": 0, "failed": failed}

    min_start = min(r["start_time"] for r in successful)
    max_end = max(r["start_time"] + (r.get("latency") or 0) for r in successful)
    dur_s = max(max_end - min_start, 1e-9)

    total_input = sum(r.get("prompt_len") or 0 for r in successful)
    total_output = sum(r.get("output_tokens") or 0 for r in successful)

    ttfts = [r.get("ttft") or 0 for r in successful]
    tpots = []
    for r in successful:
        out_len = r.get("output_tokens") or 0
        if out_len > 1:
            tpots.append(((r.get("latency") or 0) - (r.get("ttft") or 0)) / (out_len - 1))
    itls = []
    for r in successful:
        itls.extend(r.get("itl") or [])
    e2els = [r.get("latency") or 0 for r in successful]

    # Per-second histogram, same idea as serve.py: only used to derive two
    # peak scalars, then discarded.
    n_sec = int(math.ceil(max_end - min_start)) + 1
    tokens_per_second = [0] * n_sec
    concurrent_per_second = [0] * n_sec
    for r in successful:
        start = r["start_time"]
        cur = start + (r.get("ttft") or 0)
        token_times = [cur]
        for gap in (r.get("itl") or []):
            cur += gap
            token_times.append(cur)
        for t in token_times:
            idx = int(t - min_start)
            if 0 <= idx < n_sec:
                tokens_per_second[idx] += 1
        start_sec = max(0, int(start - min_start))
        end_sec = min(n_sec - 1, int(start + (r.get("latency") or 0) - min_start))
        for s in range(start_sec, end_sec + 1):
            concurrent_per_second[s] += 1

    return {
        "completed": len(successful),
        "failed": failed,
        "dur_s": dur_s,
        "total_input": total_input,
        "total_output": total_output,
        "request_throughput": len(successful) / dur_s,
        "output_throughput": total_output / dur_s,
        "total_token_throughput": (total_input + total_output) / dur_s,
        "max_output_tokens_per_s": max(tokens_per_second) if tokens_per_second else 0,
        "max_concurrent_requests": max(concurrent_per_second) if concurrent_per_second else 0,
        "prefill": _dist_stats(_prefill_rates(successful), percentiles),
        "ttft": _dist_stats(ttfts, percentiles),
        "tpot": _dist_stats(tpots, percentiles),
        "itl": _dist_stats(itls, percentiles),
        "e2el": _dist_stats(e2els, percentiles),
    }


def apply_ramp_trim(buckets: list[dict], threshold: float | None) -> dict:
    """Drop leading/trailing buckets whose active_jobs <= threshold — the
    ramp-up/ramp-down periods before/after steady state. Only trims
    contiguous runs at the two ends; a mid-run dip is real behavior, not a
    ramp artifact, so it is left alone. Returns the kept buckets plus the
    (start, end) time window they cover, so the same window can also
    restrict which raw requests feed a whole-run report.
    """
    if threshold is None or not buckets:
        return {"buckets": buckets, "window": None, "trimmed_start": 0, "trimmed_end": 0}

    start_idx = 0
    while start_idx < len(buckets) and buckets[start_idx]["active_jobs"] <= threshold:
        start_idx += 1
    end_idx = len(buckets) - 1
    while end_idx >= start_idx and buckets[end_idx]["active_jobs"] <= threshold:
        end_idx -= 1

    if start_idx > end_idx:
        return {"buckets": [], "window": None, "trimmed_start": len(buckets), "trimmed_end": 0}

    # Bucket dicts only carry `time` (the bucket's end); the step is uniform
    # so the start of the kept window is recovered from it.
    step = buckets[1]["time"] - buckets[0]["time"] if len(buckets) > 1 else buckets[0]["time"]
    kept = buckets[start_idx:end_idx + 1]
    return {
        "buckets": kept,
        "window": (start_idx * step, kept[-1]["time"]),
        "trimmed_start": start_idx,
        "trimmed_end": len(buckets) - 1 - end_idx,
    }


def compute_launch_trim_window(
    outputs: list[dict],
    conc: float | None,
) -> tuple[float, float] | None:
    """Alternative ramp-trim strategy based on request launch/completion
    order instead of per-bucket concurrency:
      - window start = the EARLIEST completion among the first `conc`
        requests by start time (the initial wave, ~one per worker) — once
        any one of them finishes, the run is past its very first round.
      - window end = the LATEST start time across all requests — the moment
        the very last request is dispatched. In a closed-loop harness that
        always keeps `conc` requests in flight, a new request only launches
        when an earlier one finishes AND there's still backlog, so the last
        such dispatch is necessarily the last-launched request's own start:
        after that instant nothing replaces a finisher, so it's pure
        drain-down, not steady state. (Using the start of the (N-conc)th
        request by start order instead — "the last wave begins" rather than
        "the last wave finishes being dispatched" — cuts one or more waves
        too early.)
    Successful requests only. Returns None if there's no usable window
    (conc <= 0, no successful requests, or conc >= N).

    The window is expressed relative to the min start-time over ALL
    outputs (success + failed) — the same baseline the buckets' own time
    coordinates and the caller's request-filtering use. Baselining on
    successful-only here instead would silently shift the window whenever
    a failed request started earlier than the first successful one,
    desyncing it from the bucket/report filtering that reads it back.
    """
    successful = [r for r in outputs if r.get("success")]
    n = len(successful)
    if conc is None or conc <= 0 or n == 0:
        return None
    conc = int(conc)
    if conc >= n:
        return None

    min_start = min(r["start_time"] for r in outputs)
    spans = [
        (
            r["start_time"] - min_start,
            r["start_time"] - min_start + (r.get("latency") or 0),
        )
        for r in successful
    ]

    by_start = sorted(spans, key=lambda s: s[0])
    window_start = min(s[1] for s in by_start[:conc])
    window_end = by_start[-1][0]

    if window_start >= window_end:
        return None
    return (window_start, window_end)


def select_buckets_in_window(
    buckets: list[dict],
    window: tuple[float, float] | None,
) -> dict:
    """Keep only buckets whose center falls inside `window`. Works for any
    window, not just a bucket-aligned one — apply_ramp_trim's own window is
    always bucket-aligned (its edges come from bucket boundaries), but
    compute_launch_trim_window's generally isn't, since it's derived from
    exact request completion timestamps.
    """
    if window is None or not buckets:
        return {"buckets": buckets, "trimmed_start": 0, "trimmed_end": 0}

    window_start, window_end = window
    step = buckets[1]["time"] - buckets[0]["time"] if len(buckets) > 1 else buckets[0]["time"]

    kept_indices = [
        i for i, b in enumerate(buckets)
        if window_start <= (b["time"] - step / 2) <= window_end
    ]
    if not kept_indices:
        return {"buckets": [], "trimmed_start": len(buckets), "trimmed_end": 0}

    first, last = kept_indices[0], kept_indices[-1]
    return {
        "buckets": buckets[first:last + 1],
        "trimmed_start": first,
        "trimmed_end": len(buckets) - 1 - last,
    }


def bucket_aggregate_report(
    outputs: list[dict],
    num_buckets: int,
    input_weight: float,
    output_weight: float,
    percentiles: tuple[float, ...] = (0.5, 0.9, 0.95, 0.99),
    buckets: list[dict] | None = None,
) -> dict:
    """Rolls the SAME per-bucket algorithm (TimelineProcessor.method_2) back
    up into a single aggregate, restricted to successful requests — the
    exact basis vLLM's calculate_metrics uses — so the two whole-run numbers
    in vllm_style_report() and this one are a fair, line-by-line comparison
    instead of being computed two structurally different ways.

    `buckets`, when given, is used as-is instead of being recomputed here.
    Pass the caller's already-selected (e.g. ramp-trimmed) bucket list rather
    than leaving this function to build its own fresh N-bucket grid sized to
    `outputs`' own start/end span: that span is just the trimmed REQUEST
    set's extent, and a single request with an outsized latency (a long-tail
    straggler that legitimately started inside the kept window but finishes
    well after everyone else) stretches that from-scratch grid into a long
    fake near-empty tail of trailing buckets — one that never existed in
    whatever bucket grid the caller actually displays elsewhere. Reusing the
    caller's buckets keeps peak_bucket_*/milabench_rate consistent with
    whatever chart or window the caller is already showing.
    """
    rows = [convert(o) for o in outputs]
    successful = [r for r in rows if r.get("success")]
    failed = len(rows) - len(successful)
    if not successful:
        return {"completed": 0, "failed": failed}

    if buckets is None:
        config = TimelineConfig(
            num_buckets=num_buckets,
            input_token_weight=input_weight,
            output_token_weight=output_weight,
        )
        proc = TimelineProcessor(config)
        buckets = proc(successful, number=num_buckets, persist=False)

    total_input = sum(r.get("prompt_len") or 0 for r in successful)
    total_output = sum(r.get("output_tokens") or 0 for r in successful)
    # bucket width from consecutive bucket times, not buckets[0]["time"]: when
    # `buckets` is a caller-supplied slice (e.g. a ramp-trimmed window that
    # dropped leading buckets), buckets[0] is no longer the first bucket of
    # the run, so its absolute "time" isn't the bucket width. dur_s is the
    # KEPT WINDOW's own span (step * count), not buckets[-1]["time"], which
    # would be measured from the untrimmed run's t=0 and silently include
    # whatever got trimmed off the start.
    if len(buckets) > 1:
        step = buckets[1]["time"] - buckets[0]["time"]
    elif buckets:
        step = buckets[0]["time"]
    else:
        step = 1e-9
    dur_s = step * len(buckets) if buckets else 1e-9

    ttfts = [r.get("ttft") or 0 for r in successful]
    tpots = []
    for r in successful:
        out_len = r.get("output_tokens") or 0
        if out_len > 1:
            tpots.append(((r.get("latency") or 0) - (r.get("ttft") or 0)) / (out_len - 1))
    itls = []
    for r in successful:
        itls.extend(r.get("itl") or [])
    e2els = [r.get("latency") or 0 for r in successful]

    return {
        "completed": len(successful),
        "failed": failed,
        "dur_s": dur_s,
        "total_input": total_input,
        "total_output": total_output,
        "request_throughput": len(successful) / dur_s,
        "output_throughput": total_output / dur_s,
        "total_token_throughput": (total_input + total_output) / dur_s,
        "peak_bucket_input_rate": max((b["input_rate"] for b in buckets), default=0),
        "peak_bucket_output_rate": max((b["output_rate"] for b in buckets), default=0),
        "peak_bucket_active_jobs": max((b["active_jobs"] for b in buckets), default=0),
        "num_buckets": num_buckets,
        "bucket_duration": step,
        # Each bucket's `rate` is exactly the per-bucket sample
        # benchmarks/vllm/main.py pushes to milabench's metric stream in
        # production (`for sampled_obs in timeline(...): push_metric(**
        # sampled_obs)`). This is the plain distribution of that raw
        # stream, not an aggregation milabench itself performs.
        "milabench_rate": _milabench_style_stats([b["rate"] for b in buckets], percentiles),
        "prefill": _dist_stats(_prefill_rates(successful), percentiles),
        "ttft": _dist_stats(ttfts, percentiles),
        # TPOT/E2EL aren't localized per-bucket (the bucket method doesn't
        # produce a per-bucket breakdown for them), but their distribution
        # over THIS function's request set — the ramp-trimmed subset when
        # trim is on — is exactly what shows the trim's impact against
        # vLLM's report, which always uses every request.
        "tpot": _dist_stats(tpots, percentiles),
        "itl": _dist_stats(itls, percentiles),
        "e2el": _dist_stats(e2els, percentiles),
        "buckets": buckets,
    }


def timeline(
    outputs,
    number=None,
    config: TimelineConfig | None = None,
    persist: bool = True,
    description: str = "",
):
    if config is None:
        config = TimelineConfig()
    if number is not None:
        config.num_buckets = number

    proc = TimelineProcessor(config)
    if persist and outputs:
        jobs = [JobAdapter(convert(l)) for l in outputs]
        jobs.sort(key=lambda item: item.start)
        proc.save_normalized_data(jobs, description=description)
    proc(outputs, persist=False)
    return proc.output


def plot_timeline(jobs):
    import altair as alt

    base = alt.Chart(jobs).encode(
        y=alt.Y(
            "worker:O",
            title="Request",
            sort="-x"
        ),
        x=alt.X(
            "start:Q",
            title="Time (s)",
            axis=alt.Axis(format=".2f"),
        ),
        color="batch_id:O",
        x2="end:Q",
    )

    bars = base.mark_bar(height=12).properties(
        width=900,
        height=25 * len(set(jobs["worker"])),
        title="Request Timeline (Gantt)"
    )
    bars.save('chart.png', scale_factor=2)
    return bars


def main():
    from argparse import ArgumentParser
    import pandas as pd

    parser = ArgumentParser()
    parser.add_argument("file", type=str, nargs="?", default=None,
                        help="JSON file with raw outputs (or .db for SQLite)")
    parser.add_argument("--db", type=str, default=None,
                        help="SQLite database path to load results from")
    parser.add_argument("--run-id", type=int, default=None,
                        help="Run ID to load from the database (latest if omitted)")
    parser.add_argument("--list-runs", action="store_true",
                        help="List all runs in the database and exit")
    parser.add_argument("-n", "--num-buckets", type=int, default=None)
    parser.add_argument("--auto-buckets", action="store_true",
                        help="Automatically choose bucket count based on request count and concurrency")
    parser.add_argument("--concurrency", "-C", type=int, default=1,
                        help="Max concurrent requests (used by --auto-buckets)")
    parser.add_argument("--highest-percentile", type=float, default=0.99,
                        help="Target percentile for --auto-buckets (default 0.99)")
    parser.add_argument("--suggest-requests", action="store_true",
                        help="Print how many requests are needed for the given bucket/concurrency/percentile and exit")
    parser.add_argument("--bucket-duration", type=float, default=None,
                        help="Fixed bucket width in seconds (overrides -n)")
    parser.add_argument("--input-weight", type=float, default=1.0,
                        help="Weight applied to input/prompt tokens")
    parser.add_argument("--output-weight", type=float, default=5.0,
                        help="Weight applied to output/generated tokens")
    parser.add_argument("--no-latency", action="store_true",
                        help="Disable per-bucket latency stats")
    
    args = parser.parse_args()

    if args.suggest_requests:
        info = suggest_num_requests(
            num_buckets=args.num_buckets or 30,
            concurrency=args.concurrency,
            highest_percentile=args.highest_percentile,
        )
        print(f"  To get stable p{int(info['highest_percentile']*100)} "
              f"with {info['num_buckets']} buckets and C={info['concurrency']}:")
        print(f"    requests needed : {info['num_requests']}")
        print(f"    waves           : {info['num_waves']}")
        print(f"    completions/bucket: {info['completions_per_bucket']}")
        return

    if args.list_runs:
        db = args.db or args.file
        store = ResultStore(db)
        for run in store.list_runs():
            print(f"  run {run['run_id']:>4d}  {run['created_at']}  "
                  f"{run['num_requests']:>6d} requests  {run.get('description', '')}")
        return

    if args.db or (args.file and args.file.endswith(".db")):
        db = args.db or args.file
        store = ResultStore(db)
        outputs = store.load(run_id=args.run_id)
    elif args.file:
        with open(args.file, "r") as fp:
            outputs = json.load(fp)
    else:
        parser.error("Provide a JSON file, --db path, or --list-runs")

    num_buckets = args.num_buckets
    if num_buckets is None:
        if args.auto_buckets:
            num_buckets = suggest_num_buckets(
                len(outputs),
                concurrency=args.concurrency,
                highest_percentile=args.highest_percentile,
            )
            print(f"Auto-selected {num_buckets} buckets for "
                  f"{len(outputs)} requests (C={args.concurrency}, "
                  f"p{int(args.highest_percentile*100)})")
        else:
            num_buckets = 30

    config = TimelineConfig(
        num_buckets=num_buckets,
        bucket_duration=args.bucket_duration,
        input_token_weight=args.input_weight,
        output_token_weight=args.output_weight,
        track_latency=not args.no_latency,
    )

    proc = TimelineProcessor(config)

    jobs = [JobAdapter(convert(l)) for l in outputs]

    jobs.sort(key=lambda item: item.end)
    jobs.sort(key=lambda item: item.start)

    results = proc.method_2(jobs)

    for line in results:
        print(line)

    _ = proc.method_1(jobs)

    data = pd.DataFrame([job.__json__() for job in jobs])
    print(data)
    plot_timeline(data)


if __name__ == "__main__":
    main()