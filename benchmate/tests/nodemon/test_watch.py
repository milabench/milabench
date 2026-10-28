import json
import time
from pathlib import Path

from benchmate.nodemon.watch import Tail, Window, block, read, statuses


def tsv(path: Path, columns: list, rows: list, partial: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["\t".join(columns)] + ["\t".join(str(cell) for cell in row) for row in rows]
    path.write_text("\n".join(lines) + "\n" + partial)


def test_a_line_still_being_written_is_left_for_the_next_read(tmp_path):
    path = tmp_path / "nodes" / "cn-a.gpu.tsv"
    tsv(path, ["time_ms", "gpu", "sm_clock_mhz"], [[1000, 0, 1410]], partial="2000\t0\t17")
    tail = Tail(path, tail_bytes=1 << 16)

    first = tail.rows()
    tsv(path, ["time_ms", "gpu", "sm_clock_mhz"], [[1000, 0, 1410], [2000, 0, 1755]])
    second = tail.rows()

    assert [row["sm_clock_mhz"] for row in first] == ["1410"]
    assert [row["sm_clock_mhz"] for row in second] == ["1755"]


def test_the_first_read_of_a_long_file_starts_near_its_end_not_at_the_header(tmp_path):
    path = tmp_path / "nodes" / "cn-a.gpu.tsv"
    tsv(path, ["time_ms", "gpu", "sm_clock_mhz"], [[t, 0, 1400 + t] for t in range(2000)])
    tail = Tail(path, tail_bytes=200)

    rows = tail.rows()

    assert tail.columns == ["time_ms", "gpu", "sm_clock_mhz"]  # the header is still read
    assert len(rows) < 2000  # ... but not the whole history behind it
    assert rows[-1]["time_ms"] == "1999"


def test_a_killed_sampler_is_reported_stale_while_a_closed_one_is_not(tmp_path):
    now = time.time()
    old = int((now - 300) * 1000)
    rows = tmp_path / "nodes"
    tsv(rows / "cn-a.gpu.tsv", ["time_ms", "gpu", "sm_clock_mhz", "util_mean"], [[old, 0, 1410, 0.9]])
    tsv(rows / "cn-b.gpu.tsv", ["time_ms", "gpu", "sm_clock_mhz", "util_mean"], [[old, 0, 1755, 0.2]])
    (tmp_path / "cn-a.status.json").write_text(json.dumps({"state": "running", "flush_every": 10.0}))
    (tmp_path / "cn-b.status.json").write_text(json.dumps({"state": "closed", "flush_every": 10.0}))

    window = Window(window_s=600.0)
    read(tmp_path, {}, window)
    window.trim(now)
    text = block(window, statuses(tmp_path), now)

    assert "STALE" in text
    assert "cn-a" in text.split("STALE", 1)[1]
    assert "cn-b" not in text.split("STALE", 1)[1]
    assert "min 1410 MHz on cn-a:0" in text
