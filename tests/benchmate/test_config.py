import io
import json
import threading

import pytest

from benchmate import config as cfg
from benchmate.config import (
    Config,
    add_runtime_config,
    load_runtime_config,
    runtime_config_path,
    ConfigChange,
    compute_config_key,
    get_config_key,
    load_multiconfig,
    multiconfig,
    next_config,
    run_multiconfig,
    set_config_key,
    tag_metric,
)


@pytest.fixture(autouse=True)
def clean_key():
    set_config_key(None)
    yield
    set_config_key(None)


def test_key_is_stable_and_distinct():
    a = compute_config_key(["--batch-size", "32"])
    assert a == compute_config_key(["--batch-size", "32"])
    assert a != compute_config_key(["--batch-size", "64"])


def test_set_and_clear_key():
    assert get_config_key() is None
    key = set_config_key(["--a"])
    assert key == get_config_key() == compute_config_key(["--a"])
    assert set_config_key(None) is None
    assert get_config_key() is None


def test_start_and_end_markers(capsys):
    set_config_key(None)
    assert capsys.readouterr().err == ""

    key = set_config_key(["--a"])
    err = capsys.readouterr().err
    assert f"config start ({key})" in err and "--a" in err

    set_config_key(None)
    assert f"config end ({key})" in capsys.readouterr().err


def test_switching_config_ends_the_previous_one(capsys):
    first = set_config_key(["--a"])
    second = set_config_key(["--b"])
    err = capsys.readouterr().err
    assert err.index(f"config end ({first})") < err.index(f"config start ({second})")


def test_markers_around_config_run(capsys):
    run_multiconfig(lambda argv: next_config("enough"), [["--a"]])
    err = capsys.readouterr().err
    assert err.index("config start") < err.index("config change triggered") < err.index("config end")


def test_key_visible_from_other_threads():
    set_config_key(["--a"])
    seen = []
    t = threading.Thread(target=lambda: seen.append(get_config_key()))
    t.start()
    t.join()
    assert seen == [get_config_key()]


def test_tag_metric():
    assert tag_metric({"rate": 1}) == {"rate": 1}

    set_config_key(["--a"])
    assert tag_metric({"rate": 1}) == {"rate": 1, "config_key": get_config_key()}

    # an explicit key is not overwritten
    assert tag_metric({"config_key": "x"})["config_key"] == "x"


def test_load_multiconfig():
    configs = [["--a", "1"], ["--b"]]
    env = {"MILABENCH_CONFIG": json.dumps({"multiconfig": configs})}
    assert load_multiconfig(env) == configs


@pytest.mark.parametrize(
    "env",
    [
        {},
        {"MILABENCH_CONFIG": ""},
        {"MILABENCH_CONFIG": "not json"},
        {"MILABENCH_CONFIG": "[]"},
        {"MILABENCH_CONFIG": "{}"},
        {"MILABENCH_CONFIG": '{"multiconfig": null}'},
    ],
)
def test_load_multiconfig_missing(env):
    assert load_multiconfig(env) == []


def test_multiconfig_from_env(monkeypatch):
    monkeypatch.setenv(
        "MILABENCH_CONFIG", json.dumps({"multiconfig": [["--a"], "--b 2"]})
    )
    assert [list(c) for c in multiconfig()] == [["--a"], ["--b", "2"]]


def test_multiconfig_defaults_to_argv(monkeypatch):
    monkeypatch.delenv("MILABENCH_CONFIG", raising=False)
    monkeypatch.setattr(cfg.sys, "argv", ["prog", "--x", "1"])
    assert [list(c) for c in multiconfig()] == [["--x", "1"]]


def test_config_sets_and_resets_key():
    (config,) = multiconfig([["--a"]])
    assert isinstance(config, Config)

    with config:
        assert get_config_key() == config.key
    assert get_config_key() is None


def test_run_multiconfig_runs_all_with_their_key():
    seen = []
    run_multiconfig(
        lambda argv: seen.append((argv, get_config_key())), [["--a"], ["--b"]]
    )

    assert [argv for argv, _ in seen] == [["--a"], ["--b"]]
    keys = [key for _, key in seen]
    assert None not in keys and keys[0] != keys[1]
    assert get_config_key() is None


def test_crash_moves_on(capsys):
    seen = []

    def run(argv):
        seen.append(argv)
        if argv == ["--a"]:
            raise ValueError("boom")

    run_multiconfig(run, [["--a"], ["--b"]])

    assert seen == [["--a"], ["--b"]]
    err = capsys.readouterr().err
    assert "crashed" in err and "boom" in err
    assert get_config_key() is None


def test_next_config_is_unconditional_jump():
    swallowed = []
    after = []

    def run(argv):
        if argv == ["--a"]:
            try:
                next_config("enough")
            except Exception:  # must not catch it
                swallowed.append(True)
            after.append(True)

    seen = []
    run_multiconfig(lambda argv: (seen.append(argv), run(argv)), [["--a"], ["--b"]])

    assert not swallowed
    assert not after
    assert seen == [["--a"], ["--b"]]


def test_config_change_is_base_exception_only():
    assert issubclass(ConfigChange, BaseException)
    assert not issubclass(ConfigChange, Exception)
    assert not issubclass(ConfigChange, SystemExit)


def test_next_config_not_swallowed_by_system_exit_handler():
    after = []

    def run(argv):
        try:
            next_config("enough")
        except SystemExit:  # e.g. code guarding argparse's sys.exit
            after.append(True)

    run_multiconfig(run, [["--a"]])
    assert not after


def test_config_change_reason_reported(capsys):
    key = compute_config_key(["--a"])
    run_multiconfig(lambda argv: next_config("enough"), [["--a"]])
    err = capsys.readouterr().err
    assert "config change triggered" in err and "enough" in err and key in err


@pytest.mark.parametrize("exc", [KeyboardInterrupt, SystemExit])
def test_fatal_signals_propagate(exc):
    seen = []

    def run(argv):
        seen.append(argv)
        raise exc()

    with pytest.raises(exc):
        run_multiconfig(run, [["--a"], ["--b"]])

    assert seen == [["--a"]]
    assert get_config_key() is None


def test_stop_program_propagates():
    from voir.phase import StopProgram

    seen = []

    def run(argv):
        seen.append(argv)
        raise StopProgram()

    with pytest.raises(StopProgram):
        run_multiconfig(run, [["--a"], ["--b"]])

    assert seen == [["--a"]]
    assert get_config_key() is None


def test_file_push_tags_metrics():
    from benchmate.metrics import file_push

    out = io.StringIO()
    push = file_push(out)

    push(rate=1)
    with Config(["--a"]) as config:
        push(rate=2)
    push(rate=3)

    first, second, third = [json.loads(line) for line in out.getvalue().splitlines()]
    assert "config_key" not in first
    assert second["config_key"] == config.key
    assert "config_key" not in third


def test_give_push_tags_metrics(monkeypatch):
    from benchmate import metrics

    given = []

    class Overseer:
        def give(self, **kwargs):
            given.append(kwargs)

    class Var:
        def get(self):
            return Overseer()

    monkeypatch.setattr(metrics, "current_overseer", Var())
    give = metrics.give_push()

    with Config(["--a"]) as config:
        give(rate=1)
    give(rate=2)

    assert given == [{"rate": 1, "config_key": config.key}, {"rate": 2}]


# Runtime config


@pytest.fixture
def runtime(tmp_path):
    return str(tmp_path / "sub" / "runtime_config.yaml")


def test_runtime_config_path(tmp_path):
    assert runtime_config_path({}) is None
    assert runtime_config_path({"MILABENCH_RUNTIME_CONFIG": "/x.yaml"}) == "/x.yaml"

    env = {
        "MILABENCH_DIR_RUNS": str(tmp_path),
        "MILABENCH_CONFIG": json.dumps({"name": "bench"}),
    }
    assert runtime_config_path(env) == str(tmp_path / "bench" / "runtime_config.yaml")


def test_add_and_load_runtime_config(runtime):
    assert load_runtime_config(runtime) == []

    add_runtime_config(["--a", 1], runtime)
    add_runtime_config("--b 2", runtime)

    assert load_runtime_config(runtime) == [["--a", "1"], ["--b", "2"]]


def _write(path, content):
    import os

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fp:
        fp.write(content)


@pytest.mark.parametrize("content", ["{{{ nope", "multiconfig: 3"])
def test_malformed_runtime_config_is_ignored(runtime, content, capsys):
    _write(runtime, content)
    assert load_runtime_config(runtime) == []
    assert "ignoring" in capsys.readouterr().err


@pytest.mark.parametrize("content", ["", "- a\n- b", "other: 1"])
def test_runtime_config_without_multiconfig_is_empty(runtime, content):
    _write(runtime, content)
    assert load_runtime_config(runtime) == []


def test_runtime_config_locked_is_skipped(runtime, monkeypatch, capsys):
    add_runtime_config(["--a"], runtime)
    monkeypatch.setattr(cfg, "LOCK_TIMEOUT", 0.05)

    from filelock import FileLock

    with FileLock(runtime + ".lock"):
        assert load_runtime_config(runtime) == []
    assert "ignoring" in capsys.readouterr().err

    assert load_runtime_config(runtime) == [["--a"]]


def test_runtime_config_picked_up_between_configs(runtime):
    seen = []

    def run(argv):
        seen.append(argv)
        if argv == ["--a"]:
            add_runtime_config(["--late"], runtime)

    run_multiconfig(run, [["--a"], ["--b"]], runtime)

    # static configs first, then the runtime ones
    assert seen == [["--a"], ["--b"], ["--late"]]


def test_runtime_config_chain(runtime):
    seen = []

    def run(argv):
        seen.append(argv)
        if argv == ["--a"]:
            add_runtime_config(["--x"], runtime)
        if argv == ["--x"]:
            add_runtime_config(["--y"], runtime)

    run_multiconfig(run, [["--a"]], runtime)
    assert seen == [["--a"], ["--x"], ["--y"]]


def test_runtime_config_is_deduplicated(runtime):
    add_runtime_config(["--a"], runtime)  # already a static config
    add_runtime_config(["--new"], runtime)

    seen = []
    run_multiconfig(seen.append, [["--a"]], runtime)

    # the file is reloaded after each config, nothing runs twice
    assert seen == [["--a"], ["--new"]]


def test_runtime_config_crash_does_not_stop(runtime):
    seen = []

    def run(argv):
        seen.append(argv)
        if argv == ["--a"]:
            add_runtime_config(["--late"], runtime)
            raise ValueError("boom")

    run_multiconfig(run, [["--a"]], runtime)
    assert seen == [["--a"], ["--late"]]


def test_runtime_config_from_env(runtime, monkeypatch):
    monkeypatch.setenv("MILABENCH_RUNTIME_CONFIG", runtime)
    monkeypatch.setenv("MILABENCH_CONFIG", json.dumps({"multiconfig": [["--a"]]}))
    add_runtime_config(["--b"])

    assert [list(c) for c in multiconfig()] == [["--a"], ["--b"]]


def test_explicit_configs_ignore_runtime_file(runtime, monkeypatch):
    monkeypatch.setenv("MILABENCH_RUNTIME_CONFIG", runtime)
    add_runtime_config(["--b"])
    assert [list(c) for c in multiconfig([["--a"]])] == [["--a"]]


def test_sync_hook(runtime):
    add_runtime_config(["--b"], runtime)
    add_runtime_config(["--c"], runtime)

    got = [list(c) for c in multiconfig([["--a"]], runtime, sync=lambda cs: cs[:1])]
    assert got == [["--a"], ["--b"]]


# CLI: milabench rtconfig add


def test_cli_add_appends_to_bench_runtime_config(tmp_path):
    from milabench.cli.rtconfig.add import Add, resolve_target
    from argparse import Namespace

    src = tmp_path / "mine.yaml"
    src.write_text("multiconfig:\n  - [--batch-size, '64']\n  - --batch-size 128\n")
    base = tmp_path / "base"

    args = Namespace(bench="bert-fp16", config=str(src), base=str(base), path=None)
    assert Add.execute(args) == 0
    assert Add.execute(args) == 0  # appended, not replaced

    target = resolve_target("bert-fp16", str(base), environ={})
    assert target == str(base / "runs" / "bert-fp16" / "runtime_config.yaml")
    assert load_runtime_config(target) == [["--batch-size", "64"], ["--batch-size", "128"]] * 2


def test_cli_matches_what_the_benchmark_reads(tmp_path):
    from milabench.cli.rtconfig.add import resolve_target

    env = {
        "MILABENCH_DIR_RUNS": str(tmp_path / "runs"),
        "MILABENCH_CONFIG": json.dumps({"name": "bert-fp16"}),
    }
    assert resolve_target("bert-fp16", None, environ=env) == runtime_config_path(env)
    assert resolve_target("bert-fp16", str(tmp_path), environ={}) == runtime_config_path(env)


def test_cli_accepts_plain_list_and_rejects_bad_files(tmp_path, capsys):
    from milabench.cli.rtconfig.add import Add, load_configs
    from argparse import Namespace

    plain = tmp_path / "plain.yaml"
    plain.write_text("- [--a]\n")
    assert load_configs(str(plain)) == [["--a"]]

    bad = tmp_path / "bad.yaml"
    bad.write_text("other: 1\n")
    args = Namespace(bench="b", config=str(bad), base=str(tmp_path), path=None)
    assert Add.execute(args) == 1
    assert "Error" in capsys.readouterr().err
