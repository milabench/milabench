import io
import json
import threading

import pytest

from benchmate import config as cfg
from benchmate.config import (
    Config,
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
