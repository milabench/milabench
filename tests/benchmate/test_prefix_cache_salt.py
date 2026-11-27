"""--prefix-cache: a fresh cache_salt per request on cold passes, none on warm (prefix_repetition) ones."""

import asyncio
import dataclasses
from types import SimpleNamespace as NS

from benchmate import benchserve


def test_auto_is_cold_except_prefix_repetition_on_vllm():
    r = benchserve.resolve_prefix_cache
    assert r(NS(prefix_cache="auto", dataset_name="random"), "vllm")
    assert not r(NS(prefix_cache="auto", dataset_name="prefix_repetition"), "vllm")
    assert not r(NS(prefix_cache="auto", dataset_name="random"), "sglang")
    assert not r(NS(prefix_cache="auto", dataset_name="random"), "llamacpp")
    assert r(NS(prefix_cache="cold", dataset_name="prefix_repetition"), "vllm")
    assert not r(NS(prefix_cache="warm", dataset_name="random"), "vllm")


@dataclasses.dataclass
class Input:
    prompt: str
    extra_body: dict | None = None


def test_salted_request_func(monkeypatch):
    seen = []

    async def fn(inp, session=None, pbar=None):
        seen.append(inp.extra_body)

    wrapped = benchserve._salted(fn)
    monkeypatch.setattr(benchserve, "_cold_requests", True)
    asyncio.run(wrapped(Input("a", {"temperature": 0}), session=None))
    asyncio.run(wrapped(Input("a", {"temperature": 0}), session=None))
    monkeypatch.setattr(benchserve, "_cold_requests", False)
    asyncio.run(wrapped(Input("a"), session=None))
    assert seen[0]["temperature"] == 0 and seen[0]["cache_salt"] != seen[1]["cache_salt"]
    assert seen[2] is None


def test_passes_beyond_the_context_are_left_out():
    ctx256k = {"name": "ctx256k", "argv": {"--random-input-len": 260096, "--random-output-len": 256}}
    ctx128k = {"name": "ctx128k", "argv": {"--random-input-len": 130048, "--random-output-len": 256}}
    warm = {"name": "ctx60k-4", "argv": {"--dataset-name": "prefix_repetition", "--prefix-repetition-prefix-len": 60000,
                                        "--prefix-repetition-suffix-len": 1000, "--prefix-repetition-output-len": 512}}
    quality = {"name": "quality", "quality": {"tasks": ["needle"]}}
    image = {"name": "img1", "argv": {"--dataset-name": "custom_image", "--custom-output-len": 64}}
    fits = benchserve.fits_context
    assert not fits(ctx256k, 131072) and fits(ctx128k, 131072) and fits(ctx256k, 262144)
    assert fits(warm, 65536) and not fits(warm, 40960)
    assert fits(quality, 8192) and fits(image, 8192) and fits(ctx256k, None)


def test_no_real_text_turns_it_off_again():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    benchserve.add_real_text_options(parser)
    assert parser.parse_args(["--real-text"]).real_text
    assert not parser.parse_args(["--real-text", "--no-real-text"]).real_text
    assert parser.parse_args([]).prefix_cache == "auto"


def test_ruler_drops_lengths_beyond_the_context(monkeypatch):
    import importlib.util
    import json
    import pathlib

    path = pathlib.Path(__file__).parents[2] / "benchmarks/longctx/eval_tasks/longctx_ruler.py"
    spec = importlib.util.spec_from_file_location("longctx_ruler", path)
    ruler = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ruler)
    cfg = {"longctx": {"ruler_lengths": [8192, 32768, 65536, 131072]}, "server": {"argv": {"--max-model-len": 131072}}}
    monkeypatch.setenv("MILABENCH_CONFIG", json.dumps(cfg))
    assert ruler._settings({})[0] == [8192, 32768, 65536]
    cfg["server"]["argv"]["--max-model-len"] = 262144
    monkeypatch.setenv("MILABENCH_CONFIG", json.dumps(cfg))
    assert ruler._settings({})[0] == [8192, 32768, 65536, 131072]
