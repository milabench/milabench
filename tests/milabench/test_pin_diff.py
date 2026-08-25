"""Tests for vLLM vs torch lockfile comparison."""

from __future__ import annotations

from types import SimpleNamespace

from milabench.cli.tools.pin_diff import PinDiff, _resolve_pairs
from milabench.dependencies.compare import (
    compare_files,
    default_pairs,
    display_spec,
    filter_pairs,
    format_diff,
    load_pins,
    matching_torch_lockfile,
    normalize_name,
    pair_for_versions,
    parse_requirement_line,
    specs_conflict,
)


def test_normalize_name():
    assert normalize_name("FlashInfer_Python") == "flashinfer-python"
    assert normalize_name("vllm[bench]") == "vllm"


def test_parse_requirement_line_skips_options():
    assert parse_requirement_line("# comment") is None
    assert parse_requirement_line("-c constraints.common.txt") is None
    assert parse_requirement_line("--index-url https://pypi.org/simple") is None
    assert parse_requirement_line("setuptools==80.10.2") == (
        "setuptools",
        "==80.10.2",
    )


def test_load_pins_follows_include(tmp_path):
    common = tmp_path / "constraints.common.txt"
    common.write_text("numpy==1.26.4\nsetuptools==84.0.0\n")
    unique = tmp_path / "constraints.cuda130.torch2100.txt"
    unique.write_text(
        "# header\n"
        "-c constraints.common.txt\n"
        "\n"
        "setuptools==84.0.0\n"
        "torch==2.10.0+cu130\n"
    )
    pins = load_pins(unique)
    assert pins["numpy"] == "==1.26.4"
    assert pins["setuptools"] == "==84.0.0"
    assert pins["torch"] == "==2.10.0+cu130"
    assert "-c constraints.common.txt" not in pins


def test_specs_conflict_exact():
    assert not specs_conflict("==1.0.0", "==1.0.0")
    assert specs_conflict("==80.10.2", "==84.0.0")
    assert specs_conflict("==2.10.0+cu130", "==2.10.0+cpu")
    assert not specs_conflict("==0.6.6", ">=0.6,<0.7")
    assert specs_conflict("==0.6.13", "==0.6.6")


def test_compare_files_reports_version_mismatches(tmp_path):
    vllm = tmp_path / "constraints.vllm.cuda130.torch2100.txt"
    torch = tmp_path / "constraints.cuda130.torch2100.txt"
    vllm.write_text(
        "flashinfer-cubin==0.6.6\n"
        "setuptools==80.10.2\n"
        "torch==2.10.0+cu130\n"
        "vllm==0.19.1+cu130\n"
    )
    torch.write_text(
        "setuptools==84.0.0\n"
        "torch==2.10.0+cu130\n"
        "numpy==2.3.0\n"
    )
    diff = compare_files(vllm, torch)
    assert [c.package for c in diff.conflicts] == ["setuptools"]
    assert display_spec(diff.conflicts[0].left) == "80.10.2"
    assert display_spec(diff.conflicts[0].right) == "84.0.0"
    assert "vllm" in diff.only_left
    assert "flashinfer-cubin" in diff.only_left
    assert "numpy" in diff.only_right
    assert diff.shared == 2
    assert [m.package for m in diff.same] == ["torch"]
    assert abs(diff.agreement - 0.5) < 1e-9

    text = format_diff(diff)
    assert "2 shared: 1 same" in text
    assert "1 different" in text
    assert "setuptools" in text
    assert "80.10.2" in text
    assert "84.0.0" in text
    assert "vllm==" not in text

    same_text = format_diff(diff, same=True)
    assert "same shared packages:" in same_text
    assert "torch" in same_text
    assert "2.10.0+cu130" in same_text


def test_matching_torch_lockfile(tmp_path):
    vllm = tmp_path / "constraints.vllm.cuda130.torch2100.txt"
    torch = tmp_path / "constraints.cuda130.torch2100.txt"
    vllm.write_text("vllm==0.19.1\n")
    torch.write_text("torch==2.10.0\n")
    assert matching_torch_lockfile(vllm) == torch
    assert default_pairs(tmp_path) == [(vllm, torch)]


def test_resolve_pairs_explicit_files(tmp_path):
    left = tmp_path / "a.txt"
    right = tmp_path / "b.txt"
    left.write_text("x==1\n")
    right.write_text("x==2\n")
    args = SimpleNamespace(
        files=[str(left), str(right)],
        pin_dir=None,
        all=False,
        cuda=None,
        torch=None,
        backend=None,
    )
    assert _resolve_pairs(args) == [(left, right)]


def test_pair_for_cuda_torch_version(tmp_path):
    vllm = tmp_path / "constraints.vllm.cuda130.torch2100.txt"
    torch = tmp_path / "constraints.cuda130.torch2100.txt"
    other = tmp_path / "constraints.vllm.cuda130.torch2110.txt"
    other_t = tmp_path / "constraints.cuda130.torch2110.txt"
    vllm.write_text("vllm==0.19.1\n")
    torch.write_text("torch==2.10.0\n")
    other.write_text("vllm==0.26.0\n")
    other_t.write_text("torch==2.11.0\n")
    assert pair_for_versions(tmp_path, cuda="130", torch="2.10.0") == (vllm, torch)
    pairs = default_pairs(tmp_path)
    assert filter_pairs(pairs, cuda="130", torch="2.10") == [(vllm, torch)]
    args = SimpleNamespace(
        files=[],
        pin_dir=str(tmp_path),
        all=False,
        cuda="130",
        torch="2.10.0",
        backend=None,
    )
    assert _resolve_pairs(args) == [(vllm, torch)]


def test_pin_diff_cli_exit_code(tmp_path, capsys):
    vllm = tmp_path / "constraints.vllm.cuda130.torch2100.txt"
    torch = tmp_path / "constraints.cuda130.torch2100.txt"
    vllm.write_text("setuptools==80.10.2\n")
    torch.write_text("setuptools==84.0.0\n")
    args = SimpleNamespace(
        files=[str(vllm), str(torch)],
        pin_dir=None,
        all=False,
        unique=False,
        same=False,
        cuda=None,
        torch=None,
        backend=None,
    )
    assert PinDiff.execute(args) == 1
    out = capsys.readouterr().out
    assert "setuptools" in out
    assert "80.10.2" in out
