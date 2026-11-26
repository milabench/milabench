"""Helpers for benchmarking external diffusion fine-tuning trainers.

Wraps tools that run as subprocesses (ai-toolkit, SimpleTuner, kohya
sd-scripts, ...): vendored repo management, shared dataset materialization,
timed runs and metric reporting through the usual pushers.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

DEFAULT_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
DEFAULT_DATASET = "lambdalabs/naruto-blip-captions"

REPOS = {
    "aitoolkit": "https://github.com/ostris/ai-toolkit",
    "simpletuner": "https://github.com/bghira/SimpleTuner",
    "kohya": "https://github.com/kohya-ss/sd-scripts",
}


def cache_root(cache: str = None) -> Path:
    base = cache or os.environ.get("MILABENCH_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base).expanduser() / "sdiffusion"


def output_dir(tool: str) -> Path:
    """Artifact location: under $MILABENCH_BASE (never inside the repo)."""
    base = os.environ.get("MILABENCH_BASE")
    root = Path(base) if base else Path(tempfile.gettempdir()) / "milabench"
    return root / "outputs" / tool


def vendor_dir(name: str, rev: str = None, cache: str = None) -> Path:
    suffix = f"@{rev}" if rev else ""
    return cache_root(cache) / "vendor" / f"{name}{suffix}"


def dataset_dir(name: str, cache: str = None) -> Path:
    return cache_root(cache) / "datasets" / name


def clone_repo(name: str, rev: str = None, cache: str = None) -> Path:
    assert name in REPOS, f"unknown tool: {name}"
    dest = vendor_dir(name, rev, cache)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if (dest / ".git").exists():
        run = ["git", "-C", str(dest), "fetch", "--all", "--tags"]
    else:
        run = ["git", "clone", REPOS[name], str(dest)]

    subprocess.run(run, check=True)
    if rev:
        subprocess.run(["git", "-C", str(dest), "checkout", rev], check=True)
    return dest


def install_requirements(req: Path):
    uv = shutil.which("uv")
    if uv:
        cmd = [uv, "pip", "install", "--python", sys.executable, "-r", str(req)]
    else:
        cmd = [sys.executable, "-m", "pip", "install", "-r", str(req)]
    subprocess.run(cmd, check=True, cwd=req.parent)


def materialize_dataset(
    dest: Path,
    dataset_id: str = DEFAULT_DATASET,
    train_count: int = 1000,
    val_count: int = 32,
):
    """Materialize an image+caption dataset as <image>.jpg / <image>.txt folders.

    This layout is understood natively by all three trainers:
      * kohya sd-scripts  (standard imagefolder layout)
      * ai-toolkit        (caption_ext: txt)
      * SimpleTuner       (--caption_strategy textfile)
    """
    train_dir = dest / "train" / "images"
    val_dir = dest / "val" / "images"
    marker = dest / ".complete"

    if marker.exists():
        return train_dir, val_dir

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset

    ds = load_dataset(dataset_id)
    split = ds["train"] if "train" in ds else ds[list(ds.keys())[0]]

    total = len(split)
    train_count = min(train_count, max(total - 1, 1))
    val_count = min(val_count, max(total - train_count, 1))

    def caption_of(row):
        for key in ("caption", "text"):
            caption = row.get(key)
            if caption:
                return caption
        captions = row.get("captions") or []
        return captions[0] if isinstance(captions, (list, tuple)) and captions else ""

    def dump(items, folder, start):
        for i in range(start, start + items):
            row = split[i]
            name = folder / f"{i:05d}"
            row["image"].convert("RGB").save(f"{name}.jpg", quality=95)
            Path(f"{name}.txt").write_text(caption_of(row))

    dump(train_count, train_dir, 0)
    dump(val_count, val_dir, train_count)

    marker.write_text(json.dumps({
        "dataset_id": dataset_id,
        "train": train_count,
        "val": val_count,
        "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }))
    return train_dir, val_dir


def report_metrics(**kwargs):
    """Push final metrics the same way benchmate's sumggle pusher does."""
    kwargs.setdefault("task", "train")
    try:
        from .metrics import sumggle_push
        sumggle_push()(**kwargs)
    except Exception:
        print(json.dumps(kwargs), flush=True)


# a trainer's progress bar (tqdm: "  24/200 [00:30<03:40, ...]") and the loss it shows on it or logs:
# kohya "avr_loss=0.123", ai-toolkit "loss: 1.234e-01", SimpleTuner "step_loss=0.123"
_STEP_RE = re.compile(r"(\d+)/(\d+) \[")
_LOSS_RE = re.compile(r"(?:avr_loss|step_loss|train_loss|loss)\s*[=:]\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)")


class StepTracker:
    """Training steps seen in a trainer's output: the time each step completed and the loss it reported.

    Only the progress bar whose total is ``steps`` counts (others: latent / text-embedding caching); without
    ``steps``, the latest bar. The window runs from the bar's first appearance (step 0, or the first step seen)
    to the last step: model load and caching are outside it.
    """

    def __init__(self, steps: int = None):
        self.steps = steps
        self.total = None
        self.window_start = None
        self.last_step = 0
        self.step_times = []   # (step, timestamp)
        self.losses = []       # (step, loss)

    def feed(self, line: str, now: float = None):
        now = time.time() if now is None else now
        match = None
        for match in _STEP_RE.finditer(line):
            pass
        if match is None:
            return
        n, total = int(match.group(1)), int(match.group(2))
        if self.steps and total != self.steps:
            return
        if total != self.total:  # a new bar: start over
            self.total, self.window_start, self.last_step, self.step_times, self.losses = total, None, 0, [], []
        if self.window_start is None:
            self.window_start = now
        if n > self.last_step:
            self.last_step = n
            self.step_times.append((n, now))
            losses = _LOSS_RE.findall(line)
            if losses:
                try:
                    self.losses.append((n, float(losses[-1])))
                except ValueError:
                    pass

    def summary(self) -> dict:
        if not self.step_times:
            return {}
        first_step, first_t = self.step_times[0]
        last_step, last_t = self.step_times[-1]
        # the window opens at the bar's first line; when that was the first completed step itself, from that step
        start = self.window_start if self.window_start < first_t else first_t
        steps_in_window = last_step if self.window_start < first_t else last_step - first_step
        seconds = last_t - start
        deltas = [b[1] - a[1] for a, b in zip(self.step_times, self.step_times[1:]) if b[0] == a[0] + 1]
        out = {
            "window_start": start,
            "window_end": last_t,
            "steps": last_step,
            "window_steps": steps_in_window,
            "window_seconds": round(seconds, 3),
            "window_steps_per_second": round(steps_in_window / seconds, 4) if seconds > 0 else None,
        }
        if deltas:
            ordered = sorted(deltas)
            out["step_time_s_p50"] = round(ordered[len(ordered) // 2], 4)
            out["step_time_s_p90"] = round(ordered[int(0.9 * (len(ordered) - 1))], 4)
        if self.losses:
            values = [loss for _, loss in self.losses]
            k = min(10, max(1, len(values) // 4))
            out.update({
                "loss_first": values[0],
                "loss_last": values[-1],
                "loss_start": round(sum(values[:k]) / k, 6),
                "loss_end": round(sum(values[-k:]) / k, 6),
                "losses": [[step, loss] for step, loss in self.losses],
            })
        return out


# the StepTracker summary of the last timed_run (empty when its output showed no progress bar)
last_run = {}


def timed_run(cmd, cwd, env=None, *, steps: int = None, tool: str = None, summary: bool = True):
    """Run a trainer; returns its wall-clock seconds (process start to exit, unchanged).

    Its output is passed through and read for the training steps (StepTracker): when any were seen, a
    ``finetune_summary`` event (window_start / window_end of the training steps, so energy can be computed over
    the steps only, step time, loss_first / loss_last) is pushed, and kept in ``last_run``.
    """
    global last_run

    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    tracker = StepTracker(steps)
    start = time.time()
    proc = subprocess.Popen(cmd, cwd=str(cwd), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    pending = b""
    out = getattr(sys.stdout, "buffer", None)
    try:
        while True:
            chunk = os.read(proc.stdout.fileno(), 65536)
            if not chunk:
                break
            if out is not None:
                out.write(chunk)
                out.flush()
            else:
                sys.stdout.write(chunk.decode(errors="replace"))
                sys.stdout.flush()
            pending += chunk
            # progress bars redraw with \r: each redraw is a line
            *lines, pending = re.split(rb"[\r\n]", pending)
            for line in lines:
                try:
                    tracker.feed(line.decode(errors="replace"))
                except Exception:  # parsing is best effort: never fail the run over it
                    pass
        if pending:
            try:
                tracker.feed(pending.decode(errors="replace"))
            except Exception:
                pass
        proc.wait()
    except BaseException:
        proc.kill()
        proc.wait()
        raise
    duration = time.time() - start
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)
    last_run = tracker.summary()
    if last_run and summary:
        event = {"tool": tool, "process_start": start, "process_seconds": round(duration, 3), **last_run}
        print("finetune_summary:", json.dumps({k: v for k, v in event.items() if k != "losses"}), flush=True)
        report_metrics(finetune_summary=event)
    return duration


def throughput(steps: int, seconds: float, batch_size: int) -> dict:
    rate = steps * batch_size / seconds if seconds > 0 else 0.0
    return {
        "num_steps": steps,
        "duration": seconds,
        "rate": rate,
        "steps_per_second": steps / seconds if seconds > 0 else 0.0,
        "samples_per_second": rate,
        "batch_size": batch_size,
    }
