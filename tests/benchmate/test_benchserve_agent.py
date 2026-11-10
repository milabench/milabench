"""benchmate.benchserve scoring passes: QualityBench results, SWE-bench sessions, agent-shaped real text."""

import json
from types import SimpleNamespace as NS

import pytest

from benchmate import benchserve
from benchmate.benchserve import quality, swebench


# --- SWE-bench -----------------------------------------------------------------


def test_subset_is_pinned_and_stratified():
    data = json.load(open(swebench.DATA))
    assert len(data["quick"]) == 50 and len(data["full"]) == 281
    assert set(data["quick"]) <= set(data["full"])
    assert data["quick"] == sorted(data["quick"])


def test_image_names_follow_the_published_naming():
    # fully qualified: podman refuses short names by default
    assert swebench.image("django__django-11099", "arm64") == "docker.io/swebench/sweb.eval.arm64.django_1776_django-11099:latest"
    assert swebench.image("sympy__sympy-20590", "x86_64") == "docker.io/swebench/sweb.eval.x86_64.sympy_1776_sympy-20590:latest"


def _traj(*cmds, status="Submitted"):
    return {
        "info": {"exit_status": status},
        "messages": [{"role": "system", "content": "x"}]
        + [{"role": "assistant", "extra": {"actions": [{"command": c}]}} for c in cmds]
        + [{"role": "user", "content": "output"}],
    }


def test_commands_normalizes_whitespace():
    assert swebench.commands(_traj("ls  -la", "cat\nfile.py")) == ["ls -la", "cat file.py"]


@pytest.mark.parametrize("cmds, hit_limit, looped", [
    (["ls", "cat a", "python t.py", "git diff"], False, False),
    # the same command three times in a row
    (["ls", "python t.py", "python t.py", "python t.py"], False, True),
    # cycling between two commands until the step limit
    (["ls", "cat a"] + ["python t.py", "sed -i x a"] * 6, True, True),
    # the same cycle, but the session submitted: not counted from the tail alone
    (["ls", "cat a"] + ["python t.py", "sed -i x a"] * 6, False, False),
])
def test_loop_detection(cmds, hit_limit, looped):
    assert swebench.loop_stats(cmds, hit_limit)["looped"] is looped


def test_wilson_interval():
    lo, hi = swebench.wilson(25, 50)
    assert lo < 0.5 < hi and 0.35 < lo and hi < 0.65
    assert swebench.wilson(0, 0) == [0.0, 0.0]


def test_swebench_run_summary(tmp_path, monkeypatch):
    ids = ["a__a-1", "b__b-2", "c__c-3", "d__d-4"]
    monkeypatch.setattr(swebench, "instance_ids", lambda tier: ids)
    monkeypatch.setattr(swebench, "dataset_dir", lambda tier: str(tmp_path / "data"))
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "test.jsonl").write_text("{}\n")
    monkeypatch.setattr(swebench.tools, "executable", lambda name, program=None: f"/bin/{name}")
    monkeypatch.setattr(swebench.tools, "container_runtime", lambda: "podman")
    monkeypatch.setattr(swebench.tools, "container_env", lambda runtime=None: {"MSWEA_DOCKER_EXECUTABLE": "podman"})
    served = []

    class FakeApi:
        env = {"DOCKER_HOST": "unix:///tmp/podman.sock"}

        def __init__(self, runtime):
            served.append(runtime)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(swebench.tools, "container_api", FakeApi)
    out = tmp_path / "out"
    trajs = {
        "a__a-1": _traj("ls", "python t.py", status="Submitted"),
        "b__b-2": _traj("ls", "pytest", "pytest", "pytest", status="LimitsExceeded"),
        "c__c-3": _traj("ls", status="Submitted"),
    }  # d__d-4: no trajectory (the agent crashed before saving)
    calls = []

    def fake_run(cmd, check=True, env=None, cwd=None):
        calls.append(cmd)
        assert env["MSWEA_DOCKER_EXECUTABLE"] == "podman"
        if cmd[0] == "/bin/swebench":
            assert env["DOCKER_HOST"] == "unix:///tmp/podman.sock", "the harness talks to podman's API"
        if cmd[0] == "/bin/mini-swe-agent":
            for i, t in trajs.items():
                (out / "trajectories" / i).mkdir(parents=True)
                (out / "trajectories" / i / f"{i}.traj.json").write_text(json.dumps(t))
        else:
            (out / f"hosted_vllm__model.{cmd[cmd.index('-r') + 1]}.json").write_text(json.dumps({"resolved_ids": ["a__a-1"]}))
        return NS(returncode=0)

    monkeypatch.setattr(swebench.subprocess, "run", fake_run)
    pushed = []
    summary = swebench.run(["--served-model-name", "model", "--port", "8123"], {"workers": 2},
                           flag=benchserve._flag, output_dir=str(out), name="bench.swebench", push=lambda **k: pushed.append(k))

    agent = calls[0]
    assert "hosted_vllm/model" in " ".join(agent) and "model.model_kwargs.api_base=http://127.0.0.1:8123/v1" in agent
    assert "agent.step_limit=250" in agent and agent[agent.index("-w") + 1] == "2"
    assert summary["samples"] == 4 and summary["solved"] == 1 and summary["scores"]["resolved"] == 0.25
    assert summary["submitted"] == 2 and summary["step_limit"] == 1 and summary["looped"] == 1 and summary["errors"] == 1
    assert pushed == [{"agent_summary": summary}]
    assert served == ["podman"], "the harness gets podman's API socket"
    records = [json.loads(line) for line in (out / "instances.jsonl").read_text().splitlines()]
    assert [r["resolved"] for r in records] == [True, False, False, False]
    assert records[3]["exit_status"] == "missing"


# --- QualityBench ----------------------------------------------------------------


def test_supported_tasks_follow_the_server():
    tasks = ["gsm8k", "bfcl", "logprobs", "needle"]
    kept, dropped = quality.supported_tasks(tasks, "vllm", {"--max-logprobs": 64, "--tool-call-parser": "x", "--max-model-len": 65536})
    assert kept == tasks and not dropped
    kept, dropped = quality.supported_tasks(tasks, "vllm", {"--max-logprobs": 64, "--max-model-len": 8192,
                                                             "--speculative-config": '{"method": "mtp"}'})
    assert kept == ["gsm8k"] and set(dropped) == {"bfcl", "logprobs", "needle"}
    assert quality.supported_tasks(["logprobs"], "sglang", {})[0] == []


def test_quality_summary_from_results_file():
    results = {
        "tasks": [{"id": "gsm8k", "label": "GSM8K", "off": {"k": 90, "n": 132, "acc": 68.18, "ci": [60.0, 75.4]}},
                  {"id": "ifeval", "label": "IFEval", "on": {"k": 40, "n": 55, "acc": 72.7, "ci": [59.8, 82.7]}}],
        "logprobs": {"ppl": 6.1, "windows": 60},
        "caveats": ["some caveat"],
    }
    pushed = []
    out = quality.summarize(results, name="b.quality", thinking="off", path="r.json", push=lambda **k: pushed.append(k))
    assert [(s["task"], s["thinking"]) for s in out] == [("gsm8k", "off"), ("ifeval", "on"), ("logprobs", "off")]
    assert out[0]["scores"]["acc"] == pytest.approx(0.6818) and out[0]["ci95"] == pytest.approx([0.6, 0.754])
    assert len(pushed) == 3 and all("eval_summary" in p for p in pushed)


def test_scoring_passes_are_not_rate_passes(monkeypatch):
    passes = [{"name": "conc8", "argv": {}}, {"name": "q", "quality": {}}, {"name": "s", "swebench": {}},
              {"name": "e", "eval": {"tasks": []}}]
    monkeypatch.setenv("MILABENCH_CONFIG", json.dumps({"client": {"passes": passes}}))
    assert [p["name"] for p in benchserve.client_passes("perf")] == ["conc8"]
    assert [p["name"] for p in benchserve.client_passes("eval")] == ["q", "s", "e"]


# --- agent-shaped real text --------------------------------------------------------


def test_shared_prefix_real_text(tmp_path):
    from benchmate.benchserve import text_corpus

    corpus = tmp_path / "corpus.json"
    corpus.write_text(json.dumps([{"conversations": [{"from": "human", "value": f"q{i} " * 50},
                                                       {"from": "gpt", "value": f"a{i} " * 50}]} for i in range(3000)]))

    class Tok:
        def __call__(self, text, add_special_tokens=False):
            return NS(input_ids=[hash(w) % 50000 for w in text.split()])

    class Request:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    class Base:
        def __init__(self, dataset_path=None, random_seed=0, **kw):
            self.dataset_path, self.random_seed = dataset_path, random_seed

    def decode(tokenizer, token_sequence, target_token_len, add_special_tokens, rng):
        return " ".join(map(str, token_sequence)), list(token_sequence), 0

    datasets = NS(BenchmarkDataset=Base, SampleRequest=Request, gen_prompt_decode_to_target_len=decode)
    ds = text_corpus.make_real_text_dataset(datasets)(dataset_path=str(corpus), random_seed=0)
    samples = ds.sample_shared_prefix(Tok(), num_requests=8, prefix_len=300, suffix_len=20, num_prefixes=2, output_len=16)
    assert len(samples) == 8 and all(s.prompt_len == 320 and s.expected_output_len == 16 for s in samples)
    prefixes = {tuple(s.prompt.split()[:300]) for s in samples}
    assert len(prefixes) == 2, "two shared contexts"
    assert len({tuple(s.prompt.split()[300:]) for s in samples}) == 8, "each turn its own"


def test_container_env_points_every_tool_at_podman(monkeypatch):
    from benchmate.benchserve import tools

    monkeypatch.setattr(tools, "podman_socket", lambda: "/run/user/1/podman/podman.sock")
    env = tools.container_env("podman")
    assert env == {"MSWEA_DOCKER_EXECUTABLE": "podman", "QUALITYBENCH_CONTAINER": "podman",
                   "DOCKER_HOST": "unix:///run/user/1/podman/podman.sock"}
    assert tools.container_env("docker")["QUALITYBENCH_CONTAINER"] == "docker"
