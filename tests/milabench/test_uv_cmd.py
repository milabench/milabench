"""Tests for resolving the uv binary used by install/pin."""

import os
import sys

from milabench.pack import uv_cmd


def test_uv_env_var_wins(monkeypatch, tmp_path):
    custom = tmp_path / "custom-uv"
    custom.write_text("#!/bin/sh\n")
    custom.chmod(0o755)
    monkeypatch.setenv("UV", str(custom))

    assert uv_cmd() == [str(custom)]


def test_prefers_uv_next_to_sys_executable(monkeypatch, tmp_path):
    monkeypatch.delenv("UV", raising=False)
    fake_python = tmp_path / "python"
    fake_python.write_text("")
    venv_uv = tmp_path / "uv"
    venv_uv.write_text("#!/bin/sh\n")
    venv_uv.chmod(0o755)
    monkeypatch.setattr(sys, "executable", str(fake_python))
    monkeypatch.setattr("milabench.pack.shutil.which", lambda name: None)

    assert uv_cmd() == [str(venv_uv)]


def test_falls_back_to_python_module(monkeypatch, tmp_path):
    monkeypatch.delenv("UV", raising=False)
    fake_python = tmp_path / "python"
    fake_python.write_text("")
    monkeypatch.setattr(sys, "executable", str(fake_python))
    monkeypatch.setattr("milabench.pack.shutil.which", lambda name: None)
    monkeypatch.setattr(os.path, "expanduser", lambda p: str(tmp_path / "no-home"))

    assert uv_cmd() == [str(fake_python), "-m", "uv"]
