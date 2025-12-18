from __future__ import annotations

from dataclasses import dataclass, asdict
from collections import defaultdict

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
class Job:
    start: float
    end: float


class Timeline:
    def __init__(self, jobs: list[Jobs]):
        self.jobs: list[Jobs] = jobs 

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

    def set_job(self, job):
        self.active_job = job
        if job is not None:
            self.active_job.worker = self.worker_id

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

    def completion_percentage(self, start, end):
        total_time = self.end - self.start

        work_time_included_in_sampling = min(end, self.end) - max(start, self.start)
        
        percentage_done = max(min(work_time_included_in_sampling / total_time, 0), 1)
        return percentage_done

    def total_token(self):
        total = (self.data['output_tokens'] + self.data['prompt_len'])

        if self.pct:
            return total * self.pct

        return total

    def token_per_second(self):
        total = (self.data['output_tokens'] + self.data['prompt_len'])
        elapsed = self.data["latency"]
        return total / elapsed


def convert(obj):
    if isinstance(obj, dict):
        return obj
    return asdict(obj)



class TimelineProcessor:
    def __init__(self):
        self.finished_jobs = []
        self.start = 0
        self.end = 0
        self.k = 0
        self.total = 0
        self.step = 0
        self.output = []

    def _on_time_change(self, now, workers):
        next_sample_time = self.samples[self.k]

        if next_sample_time < now:
            token = 0
            elapsed = self.step
            for job in self.finished_jobs:
                token += job.total_token()

            # for w in workers:
            #     if w.active_job:
            #         pct = w.active_job.completion_percentage(max(now - elapsed, 0), now)
            #         tok = w.active_job.total_token()
            #         token += tok * pct
            #         w.active_job.pct = 1 - pct
            
            instant = 0
            for w in workers:
                if w.active_job:
                    instant += w.active_job.token_per_second()


            throughput = token / elapsed

            self.k += 1
            self.total += len(self.finished_jobs)
            self.finished_jobs = []
            self.output.append({
                "rate": throughput,
                "time": now,
                "instant": instant,
            })

    def _on_job_ended(self, job, workers):
        self.finished_jobs.append(job)
        self._on_time_change(job.end, workers)

    def _sample(self, number):
        self.step = (self.end - self.start) / (number + 1)
        self.samples = [self.start + self.step * (i + 1) for i in range(number)]

    def __call__(self, outputs: list[RequestFuncOutput], number=30):
        jobs = [JobAdapter(convert(l)) for l in outputs]
        jobs.sort(key=lambda item: item.start)

        start = jobs[0].start
        for job in jobs:
            job.start -= start
            job.end -= start

            self.start = min(job.start, self.start)
            self.end = max(job.end, self.end)

        self._sample(number)
    
        workers = []
        for job in jobs:
            # Sort by workers tht will finish first
            workers.sort(key=lambda w: w.end())

            for worker in workers:
                # worker is going to finish 
                if worker.end() < job.start:
                    self._on_job_ended(worker.active_job, workers)
                    worker.set_job(job)
                    break

            else:
                w = Worker(len(workers))
                workers.append(w)
                w.set_job(job)

        workers.sort(key=lambda w: w.end())
        for worker in workers:
            self._on_job_ended(worker.active_job, workers)
            worker.set_job(None)

        return jobs


def timeline(outputs):
    proc = TimelineProcessor()
    proc(outputs)
    return proc.output




def plot_timeline(jobs):
    with open("data.json", "w") as fp:
        import json
        json.dump(jobs, fp)


    import pandas as pd
    import altair as alt

    df = pd.DataFrame(jobs)
    base = alt.Chart(df).encode(
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
        x2="end:Q",
    )

    bars = base.mark_bar(height=12).properties(
        width=900,
        height=25 * len(set(df["worker"])),
        title="Request Timeline (Gantt)"
    )

    bars.save('chart.png', scale_factor=2)
    return bars


if __name__ == "__main__":
    with open("raw_data.json", "r") as fp:
        import json
        obj = json.load(fp)

    # ab = timeline(obj)
    for p in (timeline(obj)):
        print(p)