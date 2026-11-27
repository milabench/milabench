"""quality.py: per-task options, the run-wide toggle, and one QualityBench call per group of tasks on the same
server."""

import json
from types import SimpleNamespace as NS

from benchmate.benchserve import quality


def test_a_plain_list_is_one_group_as_before():
    spec = {"tier": "quick", "thinking": ["off"], "tasks": ["gsm8k", "lcb"], "reply_window": 1800}
    resolved, thinking = quality.effective_spec(spec)
    assert thinking == ["off"]
    assert resolved == {"gsm8k": {"tier": "quick", "reply_window": 1800}, "lcb": {"tier": "quick", "reply_window": 1800}}
    assert [tasks for _, tasks in quality.task_groups(resolved)] == [["gsm8k", "lcb"]]


def test_per_task_options_split_into_groups():
    spec = {"thinking": [False, True], "thinking_max_tokens": 32768,
            "tasks": {"gsm8k": "full", "ifeval": None, "lcb": {"tier": "full", "max_tokens": 32768}}}
    resolved, thinking = quality.effective_spec(spec)
    assert thinking == ["off", "on"]  # YAML's bare off / on
    assert resolved["ifeval"]["tier"] == "quick"  # the default tier
    groups = quality.task_groups(resolved)
    assert [tasks for _, tasks in groups] == [["gsm8k"], ["ifeval"], ["lcb"]]
    assert quality.group_tags(groups) == ["full1", "quick", "full2"]


def test_unknown_task_option_is_an_error():
    try:
        quality.effective_spec({"tasks": {"lcb": {"tiers": "full"}}})
    except ValueError as e:
        assert "tiers" in str(e)
    else:
        raise AssertionError("expected a ValueError")


def test_toggle_narrows():
    spec = {"thinking": ["on"], "tasks": {"gsm8k": "full", "lcb": {"tier": "full", "max_tokens": 32768}}}
    # probe = quick everywhere; tasks / thinking only narrow (never add a mode the pass lacks)
    resolved, thinking = quality.effective_spec(spec, {"tier": "probe", "tasks": ["lcb"], "thinking": ["off"]})
    assert resolved == {"lcb": {"tier": "quick", "max_tokens": 32768}}
    assert thinking == []
    resolved, thinking = quality.effective_spec(spec, {"thinking": [True]})  # --override's [on]: a YAML boolean
    assert thinking == ["on"] and resolved["gsm8k"]["tier"] == "full"


def _run(monkeypatch, tmp_path, spec, bench=None):
    """quality.run with QualityBench stubbed: returns the commands and envs it was called with."""
    monkeypatch.setenv("MILABENCH_CONFIG", json.dumps({"server": {"backend": "vllm", "argv": {"--max-logprobs": 64}},
                                                        "max_duration": None, **(bench or {})}))
    monkeypatch.setattr(quality, "_env", lambda: {})
    calls = []

    def fake_run(cmd, env):
        calls.append((cmd, env))
        out = cmd[cmd.index("--out") + 1]
        tasks = cmd[cmd.index("--tasks") + 1].split(",")
        arm = cmd[cmd.index("--thinking") + 1] if "--thinking" in cmd else "off"
        json.dump({"tasks": [{"id": t, arm: {"k": 1, "n": 2, "acc": 50.0}} for t in tasks]}, open(out, "w"))
        return NS(returncode=0)

    monkeypatch.setattr(quality.subprocess, "run", fake_run)
    flag = lambda argv, name, default=None: default if name != "--model" else "m"
    pushed = []
    out = quality.run([], spec, flag=flag, output_dir=str(tmp_path), name="b.quality",
                      push=lambda **k: pushed.append(k["eval_summary"]))
    return calls, out


def test_run_makes_one_call_per_group_and_mode(monkeypatch, tmp_path):
    spec = {"thinking": ["off", "on"], "thinking_max_tokens": 32768,
            "tasks": {"gsm8k": "full", "lcb": {"tier": "full", "max_tokens": 32768}}}
    calls, out = _run(monkeypatch, tmp_path, spec)
    assert len(calls) == 4
    outs = [c[c.index("--out") + 1].rsplit("/", 1)[1] for c, _ in calls]
    assert outs == ["b.quality.qualitybench.full1.thinking-off.json", "b.quality.qualitybench.full2.thinking-off.json",
                    "b.quality.qualitybench.full1.thinking-on.json", "b.quality.qualitybench.full2.thinking-on.json"]
    assert all("--quick" not in c for c, _ in calls)
    assert [e.get("QB_MAX_TOKENS_FLOOR") for _, e in calls] == [None, "32768", None, "32768"]
    assert {(s["task"], s["thinking"], s["tier"]) for s in out} == {
        ("gsm8k", "off", "full"), ("lcb", "off", "full"), ("gsm8k", "on", "full"), ("lcb", "on", "full")}


def test_run_single_group_keeps_the_old_file_name(monkeypatch, tmp_path):
    calls, _ = _run(monkeypatch, tmp_path, {"tier": "quick", "thinking": ["off"], "tasks": ["gsm8k", "lcb"]})
    (cmd, _), = calls
    assert cmd[cmd.index("--out") + 1].endswith("/b.quality.qualitybench.thinking-off.json")
    assert "--quick" in cmd and cmd[cmd.index("--tasks") + 1] == "gsm8k,lcb"


def test_run_applies_the_toggle_from_the_bench_config(monkeypatch, tmp_path):
    spec = {"thinking": ["off", "on"], "tasks": {"gsm8k": "full", "lcb": "full"}}
    calls, _ = _run(monkeypatch, tmp_path, spec, {"quality_toggle": {"tier": "probe", "thinking": ["on"]}})
    (cmd, _), = calls
    assert "--quick" in cmd and cmd[cmd.index("--thinking") + 1] == "on"
