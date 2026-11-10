"""Helpers for benchmarking external diffusion fine-tuning trainers.

Wraps tools that run as subprocesses (ai-toolkit, SimpleTuner, kohya
sd-scripts, ...): vendored repo management, shared dataset materialization,
timed runs and metric reporting through the usual pushers.
"""

import json
import os
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


def timed_run(cmd, cwd, env=None):
    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    start = time.time()
    proc = subprocess.run(cmd, cwd=str(cwd), env=env)
    duration = time.time() - start
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)
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
