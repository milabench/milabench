import json
import math
import time
from pathlib import Path

import pytest

from benchmate.nodemon.watch import (
    OOM_KILL,
    XID,
    Ansi,
    Tail,
    Window,
    block,
    findings,
    main,
    ranked,
    read,
    statuses,
)


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
    window.refresh(statuses(tmp_path), now)
    text = block(window)

    assert "stale sampler: cn-a\n" in text
    assert "min 1410 MHz on cn-a:0" in text


def test_a_counter_an_earlier_job_left_is_quiet_and_only_a_rise_is_red(tmp_path):
    now = time.time()
    start = int((now - 400) * 1000)
    columns = ["time_ms", "mem_used_gib", "oom_kill"]
    # Long enough that the first read skips the opening rows: the baseline must still be the first sample.
    flat = [[start + 200 * t, 80, 61] for t in range(2000)]
    tsv(tmp_path / "nodes" / "cn-a.mem.tsv", columns, flat)
    tsv(tmp_path / "nodes" / "cn-b.mem.tsv", columns, flat[:-1] + [[flat[-1][0], 80, 63]])

    window = Window(window_s=60.0)
    read(tmp_path, {}, window)
    window.refresh({}, now)
    text = block(window, color=True)

    assert f"{Ansi.RED.value}             OOM kill: cn-b oom_kill +2{Ansi.RESET.value}" in text


def test_a_node_whose_cpu_clock_lags_the_others_is_slow_whatever_its_own_start(tmp_path):
    now = time.time()
    stamp = int((now - 5) * 1000)
    columns = ["time_ms", "package", "cpu_mhz_busy_mean"]
    for host, mhz in (("cn-a", 2400), ("cn-b", 2400), ("cn-c", 1500)):
        tsv(tmp_path / "nodes" / f"{host}.cpu.tsv", columns, [[stamp - 60_000, 0, 3700], [stamp, 0, mhz]])

    window = Window(window_s=30.0)
    read(tmp_path, {}, window)
    window.refresh({}, now)
    text = block(window)

    assert "CPU slow: cn-c:0 1500 MHz, -38% from its peers' median\n" in text


def test_follow_stops_once_every_sampler_has_finished(tmp_path, capsys):
    tsv(tmp_path / "nodes" / "cn-a.gpu.tsv", ["time_ms", "gpu", "sm_clock_mhz"], [[1000, 0, 1410]])
    (tmp_path / "cn-a.status.json").write_text(json.dumps({"state": "closed", "flush_every": 10.0}))

    assert main([str(tmp_path), "--follow", "3600"]) == 0
    assert "every sampler has finished: 1 closed" in capsys.readouterr().out


def test_nodes_rank_by_the_summed_severity_of_what_was_found_on_them(tmp_path):
    now = time.time()
    stamp = int((now - 5) * 1000)
    rows = tmp_path / "nodes"
    gpu = ["time_ms", "gpu", "sm_clock_mhz", "xid"]
    tsv(rows / "cn-a.gpu.tsv", gpu, [[stamp, 0, 1980, 79], [stamp, 1, 1980, 79]])
    tsv(rows / "cn-b.gpu.tsv", gpu, [[stamp, 0, 1980, 79], [stamp, 1, 1980, ""]])
    tsv(rows / "cn-b.mem.tsv", ["time_ms", "oom_kill"], [[stamp - 1000, 0], [stamp, 1]])
    tsv(rows / "cn-c.gpu.tsv", gpu, [[stamp, 0, 1980, ""]])

    window = Window(window_s=60.0)
    read(tmp_path, {}, window)
    window.refresh({}, now)
    scored = ranked(findings(window), {"cn-a", "cn-b", "cn-c"})

    # Two GPUs with an XID outrank one XID and an OOM kill, which outranks a clean node.
    assert [(score, host) for score, host, _ in scored] == [
        (2 * XID.severity, "cn-a"), (XID.severity + OOM_KILL.severity, "cn-b"), (0, "cn-c"),
    ]
    assert "Score" in block(window)


def test_a_rise_needs_counters_and_a_misspelt_metric_fails_where_it_is_written():
    from benchmate.nodemon.collectors import GpuCollector, MemCollector
    from benchmate.nodemon.watch import Rise

    with pytest.raises(ValueError, match="not a counter"):
        Rise(OOM_KILL, MemCollector.metric("mem_used_gib"))
    with pytest.raises(KeyError, match="has no metric 'sm_clock'"):
        GpuCollector.metric("sm_clock")


def test_a_node_is_coloured_by_its_total_so_enough_small_findings_turn_it_red(tmp_path):
    from benchmate.nodemon.watch import GPU_THERMAL, RED_FROM

    now = time.time()
    stamp = int((now - 5) * 1000)
    gpus = math.ceil(RED_FROM / GPU_THERMAL.severity)  # a yellow slowdown on enough GPUs to add up to RED_FROM
    tsv(tmp_path / "nodes" / "cn-a.gpu.tsv", ["time_ms", "gpu", "sm_clock_mhz", "clocks_event_reasons"],
        [[stamp, gpu, 1755, "0x20"] for gpu in range(gpus)])

    window = Window(window_s=60.0)
    read(tmp_path, {}, window)
    window.refresh({}, now)
    lines = block(window, color=True).splitlines()

    assert any(line.startswith(Ansi.YELLOW.value) and "thermal slowdown:" in line for line in lines)
    assert any(line.startswith(Ansi.RED.value) and "Score" in line and "cn-a" in line for line in lines)


def test_follow_waits_quietly_for_a_sampler_and_ctrl_c_ends_it_without_a_traceback(tmp_path, capsys, monkeypatch):
    def interrupt(seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(time, "sleep", interrupt)

    assert main([str(tmp_path), "--follow", "10"]) == 130
    out, err = capsys.readouterr()
    assert out == ""  # no block before any status file
    assert "Waiting for a sampler to start" in err


def test_a_file_read_before_its_header_is_written_is_read_once_it_is(tmp_path):
    # The sampler creates the file, then writes to it: a watch can open it in between.
    path = tmp_path / "nodes" / "cn-a.gpu.tsv"
    path.parent.mkdir()
    path.write_text("")
    tail = Tail(path, tail_bytes=1 << 16)

    assert tail.rows() == []
    tsv(path, ["time_ms", "gpu", "sm_clock_mhz"], [[1000, 0, 1410]])

    assert tail.rows() == [{"time_ms": "1000", "gpu": "0", "sm_clock_mhz": "1410"}]


def test_a_stale_sampler_is_reported_but_does_not_end_the_follow(tmp_path):
    from benchmate.nodemon.watch import finished

    now = time.time()
    (tmp_path / "cn-a.status.json").write_text(
        json.dumps({"state": "running", "flush_every": 10.0, "start": int((now - 300) * 1000)})
    )
    window = Window(window_s=60.0)
    window.refresh(statuses(tmp_path), now)

    assert "stale sampler: cn-a" in block(window)
    assert finished(window) is None


def test_a_counter_that_was_reset_keeps_its_rise_and_counts_on_from_zero():
    from benchmate.nodemon.watch import Counters

    counters = Counters()
    for value in (10, 12, 1, 3):  # +2, a reset to 1 (one counted since), +2
        counters.update(("mem", "oom_kill", "cn-a", ""), value)

    assert counters.risen("mem", "oom_kill") == {("cn-a", ""): 5}


def test_one_gpu_whose_clock_lags_its_peers_is_slow_but_an_idle_one_is_not(tmp_path):
    now = time.time()
    stamp = int((now - 5) * 1000)
    columns = ["time_ms", "gpu", "sm_clock_mhz", "clocks_event_reasons"]
    for host in ("cn-a", "cn-b", "cn-c"):
        # No reason is set for cn-c:1, which the driver does not explain: only its peers show it.
        rows = [[stamp, 0, 1980, "0x0"], [stamp, 1, 1500 if host == "cn-c" else 1980, "0x0"]]
        tsv(tmp_path / "nodes" / f"{host}.gpu.tsv", columns, rows + ([[stamp, 2, 345, "0x1"]] if host == "cn-a" else []))

    window = Window(window_s=60.0)
    read(tmp_path, {}, window)
    window.refresh({}, now)
    text = block(window)

    assert "GPU slow: cn-c:1 1500 MHz, -24% from its peers' median\n" in text
    assert "cn-a:2" not in text.split("GPU slow:", 1)[1]  # idling, so its clock says nothing
