import csv
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

BENCHMATE = Path(__file__).resolve().parents[2]


def nodemon(out: Path, *args: str) -> subprocess.Popen:
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(BENCHMATE), os.environ.get("PYTHONPATH", "")])}
    return subprocess.Popen(
        [sys.executable, "-m", "benchmate.nodemon", "--out", str(out), *args],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def status(out: Path, timeout: float = 10.0) -> dict:
    """The sampler's status file, once it exists."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        files = list(out.glob("*.status.json"))
        if files:
            return json.loads(files[0].read_text())
        time.sleep(0.05)
    raise TimeoutError("no status file")


def test_runs_once_per_host_writes_pending_rows_on_close_and_refuses_a_used_directory(tmp_path):
    first = nodemon(tmp_path)
    assert status(tmp_path)["state"] == "running"

    second = nodemon(tmp_path)
    assert second.wait(timeout=10) == 0  # the host already has a sampler

    time.sleep(4.5)  # at least two 2 s cpu samples
    first.send_signal(signal.SIGTERM)
    stdout, stderr = first.communicate(timeout=30)
    assert first.returncode == 0, stderr.decode()
    assert stdout == b""
    final = status(tmp_path)
    assert (final["state"], final["reason"], final["pid"]) == ("closed", "signal SIGTERM", first.pid)
    assert final["collectors"]["cpu"]["cpu_s"] > 0  # rewritten at the end, with the CPU used
    # The 10 s flush never came within 4.5 s: every row was written by close().
    for name in ("cpu", "mem"):
        (path,) = (tmp_path / "nodes").glob(f"*.{name}.tsv")
        with path.open() as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
        assert len(rows) >= 2
        assert all(row["time_ms"] != "time_ms" for row in rows)  # one header

    # A later run into the same directory refuses rather than mixing its rows in.
    rerun = nodemon(tmp_path)
    assert rerun.wait(timeout=10) == 2
    assert status(tmp_path)["pid"] == first.pid


def test_sigkill_leaves_the_status_running(tmp_path):
    process = nodemon(tmp_path)
    assert status(tmp_path)["state"] == "running"

    process.kill()
    process.wait(timeout=10)

    assert status(tmp_path)["state"] == "running"
