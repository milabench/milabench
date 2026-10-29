import json
import math
import os
import re
import shutil
import sys
import time
from pathlib import Path

import pytest

from benchmate.nodemon.watch import (
    RULES,
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

    assert f"{Ansi.RED.value}Findings     OOM kill: cn-b oom_kill +2{Ansi.RESET.value}" in text


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
    severity = {rule.name: rule.severity for rule in RULES}
    assert [(score, host) for score, host, _ in scored] == [
        (2 * severity["XID"], "cn-a"), (severity["XID"] + severity["OOM kill"], "cn-b"), (0, "cn-c"),
    ]
    assert "Score" in block(window)


def test_a_rise_needs_counters_and_a_misspelt_metric_fails_where_it_is_written():
    from benchmate.nodemon.collectors import GpuCollector, MemCollector
    from benchmate.nodemon.watch import Rise

    with pytest.raises(ValueError, match="not a counter"):
        Rise("OOM kill", 6, "", MemCollector.metric("mem_used_gib"))
    with pytest.raises(KeyError, match="has no metric 'sm_clock'"):
        GpuCollector.metric("sm_clock")


def test_a_node_is_coloured_by_its_total_so_enough_small_findings_turn_it_red(tmp_path):
    from benchmate.nodemon.watch import RED_FROM

    now = time.time()
    stamp = int((now - 5) * 1000)
    thermal = next(rule for rule in RULES if rule.name == "GPU thermal slowdown")
    gpus = math.ceil(RED_FROM / thermal.severity)  # a yellow slowdown on enough GPUs to add up to RED_FROM
    tsv(tmp_path / "nodes" / "cn-a.gpu.tsv", ["time_ms", "gpu", "sm_clock_mhz", "clocks_event_reasons"],
        [[stamp, gpu, 1755, "0x20"] for gpu in range(gpus)])

    window = Window(window_s=60.0)
    read(tmp_path, {}, window)
    window.refresh({}, now)
    lines = block(window, color=True).splitlines()

    assert any(line.startswith(Ansi.YELLOW.value) and "GPU thermal slowdown:" in line for line in lines)
    assert any(line.startswith(Ansi.RED.value) and "cn-a" in line for line in lines)  # its row in the table


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


def test_a_file_is_opened_once_and_later_reads_only_seek_to_what_was_appended(tmp_path, monkeypatch):
    path = tmp_path / "nodes" / "cn-a.gpu.tsv"
    tsv(path, ["time_ms", "gpu", "sm_clock_mhz"], [[1000, 0, 1410]])
    opened = []
    original = Path.open
    monkeypatch.setattr(Path, "open", lambda self, *a, **k: opened.append(self) or original(self, *a, **k))
    tail = Tail(path, tail_bytes=1 << 16)

    assert len(tail.rows()) == 1
    with open(path, "a") as stream:  # the sampler's side, not counted
        stream.write("2000\t0\t1755\n")
    assert tail.rows() == [{"time_ms": "2000", "gpu": "0", "sm_clock_mhz": "1755"}]
    tail.close()

    assert opened.count(path) == 1


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


def test_ib_congestion_is_a_port_far_above_the_others_not_one_that_is_merely_busy(tmp_path):
    now = time.time()
    columns = ["time_ms", "hca", "port", "xmit_wait"]
    # Over 10 s: three ports at 2 million a second, one at ten times that, and one near idle.
    for host, ticks in (("cn-a", 20_000_000), ("cn-b", 20_000_000), ("cn-c", 20_000_000), ("cn-d", 200_000_000),
                        ("cn-e", 1_000)):
        tsv(tmp_path / "nodes" / f"{host}.ib.tsv", columns,
            [[int((now - 12) * 1000), "mlx5_0", 1, 7], [int((now - 2) * 1000), "mlx5_0", 1, 7 + ticks]])

    window = Window(window_s=60.0)
    read(tmp_path, {}, window)
    window.refresh({}, now)
    text = block(window)

    assert "IB congestion: cn-d:mlx5_0:1 xmit_wait 20,000,000/s, median 2,000,000/s\n" in text
    flagged = text.split("IB congestion:", 1)[1]
    assert not any(host in flagged for host in ("cn-a", "cn-b", "cn-c", "cn-e"))


def test_several_xids_in_one_tick_are_still_an_xid(tmp_path):
    now = time.time()
    stamp = int((now - 5) * 1000)
    tsv(tmp_path / "nodes" / "cn-a.gpu.tsv", ["time_ms", "gpu", "sm_clock_mhz", "xid"],
        [[stamp, 0, 1980, "79;45"], [stamp, 1, 1980, 0], [stamp, 2, 1980, ""]])

    window = Window(window_s=60.0)
    read(tmp_path, {}, window)
    window.refresh({}, now)

    assert "XID: cn-a:0 xid 79;45\n" in block(window)
    assert "cn-a:1" not in block(window) and "cn-a:2" not in block(window)  # zero and empty are no XID


def test_flagged_hosts_limits_the_names_on_a_line(tmp_path, capsys):
    stamp = int((time.time() - 5) * 1000)
    for host in ("cn-a", "cn-b", "cn-c", "cn-d"):
        tsv(tmp_path / "nodes" / f"{host}.gpu.tsv", ["time_ms", "gpu", "sm_clock_mhz", "xid"], [[stamp, 0, 1980, 79]])
        (tmp_path / f"{host}.status.json").write_text(json.dumps({"state": "closed", "flush_every": 10.0}))

    assert main([str(tmp_path), "--flagged-hosts", "2"]) == 0
    assert "XID: cn-a:0 xid 79, cn-b:0 xid 79 and 2 more" in capsys.readouterr().out


def test_the_ranking_is_a_table_of_score_node_and_reasons(tmp_path):
    now = time.time()
    stamp = int((now - 5) * 1000)
    columns = ["time_ms", "gpu", "sm_clock_mhz", "xid"]
    tsv(tmp_path / "nodes" / "cn-a.gpu.tsv", columns, [[stamp, 0, 1980, 79], [stamp, 1, 1980, 79]])
    tsv(tmp_path / "nodes" / "cn-long.gpu.tsv", columns, [[stamp, 0, 1980, 79]])
    tsv(tmp_path / "nodes" / "cn-ok.gpu.tsv", columns, [[stamp, 0, 1980, ""]])

    window = Window(window_s=60.0)
    read(tmp_path, {}, window)
    window.refresh({}, now)

    assert block(window).splitlines()[-3:] == [
        "Score  Node     Reasons",
        "   20  cn-a     XID",
        "   10  cn-long  XID",
    ]
    assert "\n\n1/3 nodes healthy, " in block(window)


def test_the_header_carries_the_time_as_iso_8601_and_the_span_the_block_covers(tmp_path):
    start = time.time() - 40
    tsv(tmp_path / "nodes" / "cn-a.gpu.tsv", ["time_ms", "gpu", "sm_clock_mhz"],
        [[int(start * 1000), 0, 1410], [int((start + 30) * 1000), 0, 1410]])
    window = Window(window_s=60.0)
    read(tmp_path, {}, window)
    window.refresh({}, time.time())

    clock = lambda seconds: time.strftime("%H:%M:%S", time.localtime(seconds))
    lines = block(window).splitlines()
    assert re.fullmatch(r"== \d{4}-\d\d-\d\dT\d\d:\d\d:\d\d[+-]\d\d:\d\d  1 node", lines[0])
    unstamped = block(window, stamp=False).splitlines()
    assert unstamped[0] == "" and "p50" in unstamped[1]  # without it a blank line and the table's heading come first
    assert any(line.endswith(f" healthy, {clock(int(start))} to {clock(int(start + 30))} (last 60s)") for line in lines)

    # Once the rows are older than the window there is no span, and the line says no more than the window.
    window.refresh({"cn-a": {"state": "running"}}, time.time() + 600)
    assert "0/1 nodes healthy, last 60s" in block(window).splitlines()



def test_the_block_has_blank_lines_between_sections_and_bold_headings(tmp_path):
    now = time.time()
    stamp = int((now - 5) * 1000)
    tsv(tmp_path / "nodes" / "cn-a.gpu.tsv", ["time_ms", "gpu", "sm_clock_mhz", "xid"], [[stamp, 0, 1980, 79]])
    window = Window(window_s=60.0)
    read(tmp_path, {}, window)
    window.refresh({}, now)

    lines = block(window).splitlines()
    findings_at = next(at for at, line in enumerate(lines) if line.startswith("Findings"))
    assert lines[0].startswith("==") and lines[1] == "" and lines[findings_at - 1] == ""
    assert lines.count("") == 4  # header | readings | healthy | findings | ranking

    painted = block(window, color=True).splitlines()
    assert painted[0].startswith(Ansi.BOLD.value)
    assert any(line.startswith(Ansi.BOLD.value) and "Reasons" in line for line in painted)
    assert any(line.startswith(Ansi.PURPLE.value) and "nodes healthy" in line for line in painted)


def test_redraw_overwrites_from_the_top_clears_the_rest_and_cuts_what_the_terminal_cannot_show(monkeypatch):
    from benchmate.nodemon.watch import redraw

    monkeypatch.setattr(shutil, "get_terminal_size", lambda: os.terminal_size((80, 5)))

    assert redraw("a\nb", first=True) == "\033[2J\033[Ha\033[K\nb\033[K\033[J"
    assert redraw("a\nb", first=False).startswith("\033[H")
    cut = redraw("\n".join("abcdefgh"), first=False).split("\n")  # 8 lines on a 5-row terminal: 4 rows to use
    assert len(cut) == 4 and "5 more line(s)" in cut[-1]
    assert redraw("\033[1m" + "x" * 200 + "\033[0m", first=False) == f"\033[H\033[1m{'x' * 80}\033[0m\033[K\033[J"


def test_a_follow_on_a_terminal_redraws_in_place_but_a_pipe_appends(tmp_path, capsys, monkeypatch):
    tsv(tmp_path / "nodes" / "cn-a.gpu.tsv", ["time_ms", "gpu", "sm_clock_mhz"], [[1000, 0, 1410]])
    (tmp_path / "cn-a.status.json").write_text(json.dumps({"state": "closed", "flush_every": 10.0}))

    assert main([str(tmp_path), "--follow", "3600"]) == 0
    assert "\033[2J" not in capsys.readouterr().out  # captured output is not a terminal

    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    assert main([str(tmp_path), "--follow", "3600"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("\033[2J\033[H") and "every sampler has finished" in out
    assert "  1 node\033" not in out  # redrawn in place, the block has no time header


def test_the_readings_are_a_table_of_p50_p90_and_max(tmp_path):
    now = time.time()
    columns = ["time_ms", "gpu", "sm_clock_mhz", "clocks_event_reasons", "util_mean", "power_w", "temp_c"]
    rows = [[int((now - 10 + t) * 1000), 0, clock, "0x0", util, 300, 65] for t, (clock, util) in enumerate(
        [(1980, 0.5), (1980, 0.7), (1980, 0.9), (1800, 1.0)])]
    tsv(tmp_path / "nodes" / "cn-a.gpu.tsv", columns, rows)
    tsv(tmp_path / "nodes" / "cn-a.mem.tsv", ["time_ms", "mem_used_gib"], [[rows[0][0], 20.0], [rows[1][0], 30.0]])
    tsv(tmp_path / "nodes" / "cn-a.gpm.tsv", ["time_ms", "gpu", "sm_util"], [[rows[0][0], 0, 0.4]])
    tsv(tmp_path / "nodes" / "cn-a.cpu.tsv", ["time_ms", "package", "cpu_util"], [[rows[0][0], 0, 0.25]])

    window = Window(window_s=60.0)
    read(tmp_path, {}, window)
    window.refresh({}, now)
    lines = block(window).splitlines()

    table = {line[:14].strip(): line[14:].split() for line in lines[2:13] if line.strip()}
    assert table[""] == ["p50", "p90", "max", "Note"]
    assert table["GPU clock"][:6] == ["1980", "MHz", "1980", "MHz", "1980", "MHz"]
    assert "min 1800 MHz on cn-a:0" in next(line for line in lines if line.startswith("GPU clock"))
    assert table["GPU busy"][:3] == ["0.90", "1.00", "1.00"]
    assert table["GPU SM util"][:3] == ["0.40", "0.40", "0.40"]
    assert "GPU SM active" not in table  # no dcgm samples
    assert table["GPU power"][:2] == ["300", "W"] and table["GPU temp"][:2] == ["65", "C"]
    assert table["GPU temp"][-4:] == ["hottest", "cn-a:0", "65", "C"]
    assert table["CPU busy"][:3] == ["0.25", "0.25", "0.25"]
    assert table["Host memory"][:6] == ["30.0", "GiB", "30.0", "GiB", "30.0", "GiB"]
    assert "NVLink" not in table  # nothing sampled: only the GPU rows are shown without samples
