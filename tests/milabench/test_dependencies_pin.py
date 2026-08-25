"""Tests for milabench.dependencies.pin — pure logic and file-processing functions."""

from __future__ import annotations

from pathlib import Path

import pytest
from packaging.version import Version

from milabench.dependencies.pin import (
    _PinBlock,
    _append_vllm_wheel_note,
    _compat_conditions_match,
    _extract_common_constraints,
    _filter_combinations,
    _is_vllm_requirement,
    _normalize_backend_version,
    _parse_constraint_file,
    _strip_index_urls_from_constraint_file,
    _torch_local_label,
    _torch_minor_pin,
    _torch_pin,
    _validate_pinned_constraint_file,
    _vllm_wheel_note_lines,
    constraint_filename,
    get_constraint_file,
)
from milabench.dependencies.platforms import (
    BackendConfig,
    CompatEntry,
    CompatRule,
    IndexConfig,
    PlatformConfig,
)
from milabench.dependencies.pin import (
    _build_constraints_content,
    _build_index_args,
    _resolve_compat_constraints,
)


# ---------------------------------------------------------------------------
# constraint_filename
# ---------------------------------------------------------------------------
class TestConstraintFilename:
    def test_cuda_with_arch(self):
        assert (
            constraint_filename("cuda", "130", "2.12.0", "x86_64")
            == "constraints.cuda130.torch2120.x86_64.txt"
        )

    def test_rocm_with_arch(self):
        assert (
            constraint_filename("rocm", "7.1", "2.10.0", "aarch64")
            == "constraints.rocm71.torch2100.aarch64.txt"
        )

    def test_cpu_no_backend_version(self):
        assert (
            constraint_filename("cpu", "", "2.12.0", "x86_64")
            == "constraints.cpu.torch2120.x86_64.txt"
        )

    def test_no_arch(self):
        assert (
            constraint_filename("cuda", "130", "2.12.0")
            == "constraints.cuda130.torch2120.txt"
        )

    def test_empty_arch_explicit(self):
        assert (
            constraint_filename("cuda", "126", "2.10.0", "")
            == "constraints.cuda126.torch2100.txt"
        )

    def test_four_part_torch(self):
        assert (
            constraint_filename("cuda", "130", "2.12.1", "x86_64")
            == "constraints.cuda130.torch2121.x86_64.txt"
        )


# ---------------------------------------------------------------------------
# get_constraint_file
# ---------------------------------------------------------------------------
class TestGetConstraintFile:
    def test_returns_path_under_pin_dir(self, tmp_path):
        result = get_constraint_file(tmp_path, "cuda", "130", "2.12.0")
        assert result == tmp_path / "constraints.cuda130.torch2120.txt"

    def test_with_arch(self, tmp_path):
        result = get_constraint_file(tmp_path, "rocm", "7.2", "2.11.0", "x86_64")
        assert result == tmp_path / "constraints.rocm72.torch2110.x86_64.txt"

    def test_is_path_object(self, tmp_path):
        result = get_constraint_file(tmp_path, "cpu", "", "2.10.0")
        assert isinstance(result, Path)


# ---------------------------------------------------------------------------
# _filter_combinations — --set restricts the pin matrix
# ---------------------------------------------------------------------------
_FAKE_COMBOS = [
    ("cpu", "", "2.10.0", "x86_64"),
    ("cpu", "", "2.11.0", "x86_64"),
    ("cuda", "126", "2.10.0", "x86_64"),
    ("cuda", "130", "2.10.0", "x86_64"),
    ("cuda", "130", "2.10.0", "aarch64"),
    ("cuda", "130", "2.11.0", "x86_64"),
    ("rocm", "7.1", "2.10.0", "x86_64"),
]


class TestFilterCombinations:
    def test_no_overrides_keeps_all(self):
        assert _filter_combinations(_FAKE_COMBOS, None) == _FAKE_COMBOS
        assert _filter_combinations(_FAKE_COMBOS, {}) == _FAKE_COMBOS

    def test_cuda_and_torch(self):
        got = _filter_combinations(
            _FAKE_COMBOS, {"cuda": "130", "torch": "2.10.0"}
        )
        assert got == [
            ("cuda", "130", "2.10.0", "x86_64"),
            ("cuda", "130", "2.10.0", "aarch64"),
        ]

    def test_torch_two_part_normalized(self):
        got = _filter_combinations(_FAKE_COMBOS, {"torch": "2.10"})
        assert {c[0:3] for c in got} == {
            ("cpu", "", "2.10.0"),
            ("cuda", "126", "2.10.0"),
            ("cuda", "130", "2.10.0"),
            ("rocm", "7.1", "2.10.0"),
        }

    def test_backend_shorthand(self):
        got = _filter_combinations(_FAKE_COMBOS, {"backend": "cu130", "torch": "2.10.0"})
        assert got == [
            ("cuda", "130", "2.10.0", "x86_64"),
            ("cuda", "130", "2.10.0", "aarch64"),
        ]

    def test_no_match_raises(self):
        with pytest.raises(RuntimeError, match="No pin combinations match"):
            _filter_combinations(_FAKE_COMBOS, {"cuda": "118", "torch": "2.10.0"})


# ---------------------------------------------------------------------------
# _normalize_backend_version
# ---------------------------------------------------------------------------
class TestNormalizeBackendVersion:
    def test_three_digit_compact(self):
        assert _normalize_backend_version("130") == "13.0"

    def test_three_digit_compact_126(self):
        assert _normalize_backend_version("126") == "12.6"

    def test_already_dotted(self):
        assert _normalize_backend_version("7.1") == "7.1"

    def test_already_dotted_long(self):
        assert _normalize_backend_version("13.0.1") == "13.0.1"

    def test_empty_string(self):
        assert _normalize_backend_version("") == ""

    def test_two_digit_compact(self):
        assert _normalize_backend_version("72") == "7.2"

    def test_single_digit(self):
        assert _normalize_backend_version("6") == ".6"


# ---------------------------------------------------------------------------
# _compat_conditions_match
# ---------------------------------------------------------------------------
class TestCompatConditionsMatch:
    def test_single_condition_match(self):
        known = {"torch": Version("2.12.0")}
        assert _compat_conditions_match("torch>=2.11", known) is True

    def test_single_condition_no_match(self):
        known = {"torch": Version("2.9.0")}
        assert _compat_conditions_match("torch>=2.11", known) is False

    def test_multiple_conditions_all_match(self):
        known = {"torch": Version("2.12.0"), "cuda": Version("13.0")}
        assert _compat_conditions_match("torch>=2.11,cuda>=13", known) is True

    def test_multiple_conditions_partial_match(self):
        known = {"torch": Version("2.12.0"), "cuda": Version("12.6")}
        assert _compat_conditions_match("torch>=2.11,cuda>=13", known) is False

    def test_missing_key(self):
        known = {"torch": Version("2.12.0")}
        assert _compat_conditions_match("torch>=2.11,cuda>=13", known) is False

    def test_empty_known(self):
        assert _compat_conditions_match("torch>=2.11", {}) is False

    def test_exact_version(self):
        known = {"torch": Version("2.11.0")}
        assert _compat_conditions_match("torch==2.11.0", known) is True

    def test_less_than(self):
        known = {"torch": Version("2.10.0")}
        assert _compat_conditions_match("torch<2.11", known) is True

    def test_invalid_condition_format(self):
        known = {"torch": Version("2.12.0")}
        assert _compat_conditions_match("", known) is False

    def test_whitespace_in_conditions(self):
        known = {"torch": Version("2.12.0"), "cuda": Version("13.0")}
        assert _compat_conditions_match("torch>=2.11, cuda>=13", known) is True


# ---------------------------------------------------------------------------
# _PinBlock
# ---------------------------------------------------------------------------
class TestPinBlock:
    def test_package_name_simple(self):
        b = _PinBlock(package_line="numpy==1.26.4")
        assert b.package_name == "numpy"

    def test_package_name_hyphenated(self):
        b = _PinBlock(package_line="scikit-learn==1.5.0")
        assert b.package_name == "scikit-learn"

    def test_as_text_no_via(self):
        b = _PinBlock(package_line="numpy==1.26.4")
        assert b.as_text() == "numpy==1.26.4"

    def test_as_text_with_via_comments(self):
        b = _PinBlock(
            package_line="numpy==1.26.4",
            via_comments=["    # via", "    #   pandas", "    #   scipy"],
        )
        expected = "numpy==1.26.4\n    # via\n    #   pandas\n    #   scipy"
        assert b.as_text() == expected

    def test_default_via_comments_empty(self):
        b = _PinBlock(package_line="torch==2.12.0")
        assert b.via_comments == []

    def test_package_name_no_version(self):
        b = _PinBlock(package_line="torch")
        assert b.package_name == "torch"


# ---------------------------------------------------------------------------
# _parse_constraint_file
# ---------------------------------------------------------------------------
class TestParseConstraintFile:
    def test_basic_parsing(self, tmp_path):
        content = (
            "# Pinned with: cuda=130 torch=2.12.0\n"
            "# Generated by: milabench pin\n"
            "\n"
            "numpy==1.26.4\n"
            "    # via\n"
            "    #   pandas\n"
            "scipy==1.14.0\n"
        )
        f = tmp_path / "constraints.txt"
        f.write_text(content)
        header, blocks = _parse_constraint_file(f)
        assert len(header) == 3
        assert header[0] == "# Pinned with: cuda=130 torch=2.12.0"
        assert len(blocks) == 2
        assert blocks[0].package_name == "numpy"
        assert blocks[0].via_comments == ["    # via", "    #   pandas"]
        assert blocks[1].package_name == "scipy"
        assert blocks[1].via_comments == []

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("")
        header, blocks = _parse_constraint_file(f)
        assert header == []
        assert blocks == []

    def test_header_only(self, tmp_path):
        content = "# Just a comment\n# Another comment\n"
        f = tmp_path / "header_only.txt"
        f.write_text(content)
        header, blocks = _parse_constraint_file(f)
        assert len(header) == 2
        assert blocks == []

    def test_no_header(self, tmp_path):
        content = "numpy==1.26.4\nscipy==1.14.0\n"
        f = tmp_path / "no_header.txt"
        f.write_text(content)
        header, blocks = _parse_constraint_file(f)
        assert header == []
        assert len(blocks) == 2

    def test_blank_lines_between_blocks(self, tmp_path):
        content = (
            "numpy==1.26.4\n"
            "\n"
            "scipy==1.14.0\n"
            "    # via\n"
            "    #   something\n"
            "\n"
            "torch==2.12.0\n"
        )
        f = tmp_path / "blanks.txt"
        f.write_text(content)
        header, blocks = _parse_constraint_file(f)
        assert len(blocks) == 3
        assert blocks[0].package_name == "numpy"
        assert blocks[1].package_name == "scipy"
        assert blocks[2].package_name == "torch"

    def test_via_comments_attached_correctly(self, tmp_path):
        content = (
            "aiohttp==3.9.0\n"
            "    # via\n"
            "    #   dep-a\n"
            "    #   dep-b\n"
            "boto3==1.34.0\n"
            "    # via\n"
            "    #   dep-c\n"
        )
        f = tmp_path / "via.txt"
        f.write_text(content)
        _, blocks = _parse_constraint_file(f)
        assert len(blocks) == 2
        assert len(blocks[0].via_comments) == 3
        assert len(blocks[1].via_comments) == 2


# ---------------------------------------------------------------------------
# _strip_index_urls_from_constraint_file
# ---------------------------------------------------------------------------
class TestStripIndexUrls:
    def test_removes_index_lines(self, tmp_path):
        content = (
            "--index-url https://pypi.org/simple\n"
            "--extra-index-url https://download.pytorch.org/whl/cu130\n"
            "--find-links https://example.com/wheels\n"
            "numpy==1.26.4\n"
        )
        f = tmp_path / "c.txt"
        f.write_text(content)
        _strip_index_urls_from_constraint_file(f, "cuda", "130", "2.12.0")
        result = f.read_text()
        assert "--index-url" not in result
        assert "--extra-index-url" not in result
        assert "--find-links" not in result
        assert "numpy==1.26.4" in result

    def test_replaces_autogenerated_header(self, tmp_path):
        content = (
            "# This file was autogenerated by uv\n"
            "numpy==1.26.4\n"
        )
        f = tmp_path / "c.txt"
        f.write_text(content)
        _strip_index_urls_from_constraint_file(f, "cuda", "130", "2.12.0")
        result = f.read_text()
        assert "# Pinned with: cuda=130 torch=2.12.0" in result
        assert "# Generated by: milabench pin" in result
        assert "This file was autogenerated" not in result

    def test_adds_header_if_missing(self, tmp_path):
        content = "numpy==1.26.4\n"
        f = tmp_path / "c.txt"
        f.write_text(content)
        _strip_index_urls_from_constraint_file(f, "cuda", "130", "2.12.0")
        result = f.read_text()
        lines = result.splitlines()
        assert lines[0] == "# Pinned with: cuda=130 torch=2.12.0"
        assert lines[1] == "# Generated by: milabench pin"

    def test_strips_uv_command_comment(self, tmp_path):
        content = (
            "#    uv pip compile --no-build /tmp/toml-deps-abc123.txt\n"
            "numpy==1.26.4\n"
        )
        f = tmp_path / "c.txt"
        f.write_text(content)
        _strip_index_urls_from_constraint_file(f, "cuda", "130", "2.12.0")
        result = f.read_text()
        assert "uv pip compile" not in result

    def test_normalizes_temp_paths(self, tmp_path):
        content = (
            "# via -r /tmp/toml-deps-abc123.txt\n"
            "# -c /tmp/toml-constraints-xyz789.txt\n"
            "numpy==1.26.4\n"
        )
        f = tmp_path / "c.txt"
        f.write_text(content)
        _strip_index_urls_from_constraint_file(f, "cuda", "130", "2.12.0")
        result = f.read_text()
        assert "requirements.in" in result
        assert "constraints.in" in result
        assert "/tmp/toml-deps-" not in result
        assert "/tmp/toml-constraints-" not in result

    def test_cpu_backend_no_version(self, tmp_path):
        content = "numpy==1.26.4\n"
        f = tmp_path / "c.txt"
        f.write_text(content)
        _strip_index_urls_from_constraint_file(f, "cpu", "", "2.12.0")
        result = f.read_text()
        assert "# Pinned with: cpu torch=2.12.0" in result

    def test_with_arch_in_header(self, tmp_path):
        content = "numpy==1.26.4\n"
        f = tmp_path / "c.txt"
        f.write_text(content)
        _strip_index_urls_from_constraint_file(f, "cuda", "130", "2.12.0", "x86_64")
        result = f.read_text()
        assert "arch=x86_64" in result


# ---------------------------------------------------------------------------
# _extract_common_constraints
# ---------------------------------------------------------------------------
class TestExtractCommonConstraints:
    def _write_constraint(self, path, header_lines, packages):
        """Helper: write a fake constraint file with header + package blocks."""
        parts = list(header_lines)
        for pkg_line, via in packages:
            parts.append(pkg_line)
            parts.extend(via)
        path.write_text("\n".join(parts) + "\n")

    def test_basic_extraction(self, tmp_path):
        f1 = tmp_path / "constraints.cuda130.torch2120.txt"
        f2 = tmp_path / "constraints.rocm72.torch2120.txt"

        common_pkgs = [
            ("numpy==1.26.4", ["    # via", "    #   pandas"]),
            ("scipy==1.14.0", []),
        ]
        unique_f1 = [("cupy==13.0", [])]
        unique_f2 = [("rccl==1.0", [])]

        self._write_constraint(
            f1,
            ["# Header cuda"],
            common_pkgs + unique_f1,
        )
        self._write_constraint(
            f2,
            ["# Header rocm"],
            common_pkgs + unique_f2,
        )

        _extract_common_constraints([f1, f2], tmp_path)

        common_file = tmp_path / "constraints.common.txt"
        assert common_file.exists()
        common_text = common_file.read_text()
        assert "numpy==1.26.4" in common_text
        assert "scipy==1.14.0" in common_text
        assert "cupy" not in common_text
        assert "rccl" not in common_text

        # Individual files should now only have unique packages + -c reference
        f1_text = f1.read_text()
        assert "cupy==13.0" in f1_text
        assert "-c constraints.common.txt" in f1_text
        assert "numpy==1.26.4" not in f1_text

        f2_text = f2.read_text()
        assert "rccl==1.0" in f2_text
        assert "-c constraints.common.txt" in f2_text
        assert "numpy==1.26.4" not in f2_text

    def test_no_common_packages(self, tmp_path, capsys):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        self._write_constraint(f1, ["# H1"], [("pkg-a==1.0", [])])
        self._write_constraint(f2, ["# H2"], [("pkg-b==2.0", [])])

        _extract_common_constraints([f1, f2], tmp_path)

        common_file = tmp_path / "constraints.common.txt"
        assert not common_file.exists()
        captured = capsys.readouterr()
        assert "No common packages" in captured.out

    def test_all_packages_common(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        pkgs = [("torch==2.12.0", []), ("numpy==1.26.4", [])]
        self._write_constraint(f1, ["# H1"], pkgs)
        self._write_constraint(f2, ["# H2"], pkgs)

        _extract_common_constraints([f1, f2], tmp_path)

        common_file = tmp_path / "constraints.common.txt"
        assert common_file.exists()
        common_text = common_file.read_text()
        assert "torch==2.12.0" in common_text
        assert "numpy==1.26.4" in common_text

        # Individual files should only have header + -c ref + no packages
        f1_text = f1.read_text()
        assert "-c constraints.common.txt" in f1_text
        assert "torch==2.12.0" not in f1_text

    def test_strips_old_c_references_in_header(self, tmp_path):
        """Header lines starting with '-c ' are stripped before re-adding the new ref."""
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"

        # Manually write files so the -c line lands in the header
        # (parser puts comment/blank lines before first package into the header)
        f1.write_text(
            "# H1\n"
            "numpy==1.0\n"
        )
        f2.write_text(
            "# H2\n"
            "numpy==1.0\n"
        )

        _extract_common_constraints([f1, f2], tmp_path)

        common_file = tmp_path / "constraints.common.txt"
        assert common_file.exists()
        f1_text = f1.read_text()
        assert "-c constraints.common.txt" in f1_text
        assert "# H1" in f1_text


# ---------------------------------------------------------------------------
# _build_index_args (uses PlatformConfig dataclasses directly)
# ---------------------------------------------------------------------------
class TestBuildIndexArgs:
    def test_index_url_only(self):
        config = PlatformConfig(
            backends={
                "cuda": BackendConfig(
                    name="cuda",
                    indexes=IndexConfig(
                        index_url="https://download.pytorch.org/whl/cu{cuda}",
                    ),
                )
            }
        )
        args = _build_index_args(config, "cuda", {"cuda": "130"})
        assert args == [
            "--index-url",
            "https://download.pytorch.org/whl/cu130",
        ]

    def test_extra_index_and_find_links(self):
        config = PlatformConfig(
            backends={
                "cuda": BackendConfig(
                    name="cuda",
                    indexes=IndexConfig(
                        index_url="https://pypi.org/simple",
                        extra_index_url=["https://download.pytorch.org/whl/cu{cuda}"],
                        find_links=["https://example.com/{torch}"],
                    ),
                )
            }
        )
        args = _build_index_args(config, "cuda", {"cuda": "130", "torch": "2.12.0"})
        assert "--index-url" in args
        assert "--extra-index-url" in args
        assert "https://download.pytorch.org/whl/cu130" in args
        assert "--find-links" in args
        assert "https://example.com/2.12.0" in args

    def test_skips_missing_expanded_assets_find_links(self, monkeypatch):
        from milabench.dependencies import pin as pin_mod

        pin_mod._find_links_availability_cache.clear()
        monkeypatch.setattr(pin_mod, "_find_links_url_available", lambda url, timeout=10.0: False)

        missing = (
            "https://github.com/milabench/wheels/releases/expanded_assets/"
            "torch{torch_short}-cu{cuda}"
        )
        config = PlatformConfig(
            vars={"torch": "2.11.0", "cuda": "129"},
            backends={
                "cuda": BackendConfig(
                    name="cuda",
                    indexes=IndexConfig(
                        index_url="https://pypi.org/simple",
                        find_links=[missing, "https://example.com/always"],
                    ),
                )
            },
        )
        args = _build_index_args(
            config, "cuda", {"cuda": "129", "torch": "2.11.0"}
        )
        assert "--find-links" in args
        assert "https://example.com/always" in args
        assert "expanded_assets/torch2.11-cu129" not in " ".join(args)

    def test_keeps_published_expanded_assets_find_links(self, monkeypatch):
        from milabench.dependencies import pin as pin_mod

        pin_mod._find_links_availability_cache.clear()
        monkeypatch.setattr(pin_mod, "_find_links_url_available", lambda url, timeout=10.0: True)

        url = (
            "https://github.com/milabench/wheels/releases/expanded_assets/"
            "torch{torch_short}-rocm{rocm}"
        )
        config = PlatformConfig(
            vars={"torch": "2.12.1", "rocm": "7.2"},
            backends={
                "rocm": BackendConfig(
                    name="rocm",
                    indexes=IndexConfig(
                        index_url="https://pypi.org/simple",
                        find_links=[url],
                    ),
                )
            },
        )
        args = _build_index_args(
            config, "rocm", {"rocm": "7.2", "torch": "2.12.1"}
        )
        assert (
            "https://github.com/milabench/wheels/releases/expanded_assets/"
            "torch2.12-rocm7.2"
        ) in args

    def test_skips_missing_pyg_find_links(self, monkeypatch):
        """PyG ships no cu129 index; the 403 must not fail resolution."""
        from milabench.dependencies import pin as pin_mod

        pin_mod._find_links_availability_cache.clear()
        monkeypatch.setattr(
            pin_mod,
            "_find_links_url_available",
            lambda url, timeout=10.0: "cu129" not in url,
        )

        pyg = "https://data.pyg.org/whl/torch-{torch}+cu{cuda}.html"
        config = PlatformConfig(
            vars={"torch": "2.11.0", "cuda": "129"},
            backends={
                "cuda": BackendConfig(
                    name="cuda",
                    indexes=IndexConfig(
                        index_url="https://pypi.org/simple",
                        find_links=[pyg],
                    ),
                )
            },
        )
        missing = _build_index_args(config, "cuda", {"cuda": "129", "torch": "2.11.0"})
        assert "data.pyg.org" not in " ".join(missing)

        present = _build_index_args(config, "cuda", {"cuda": "130", "torch": "2.11.0"})
        assert "https://data.pyg.org/whl/torch-2.11.0+cu130.html" in present

    def test_unknown_backend_uses_default_pypi(self):
        config = PlatformConfig()
        args = _build_index_args(config, "xpu", {})
        assert args == ["--index-url", "https://pypi.org/simple"]

    def test_no_index_url(self):
        config = PlatformConfig(
            backends={
                "cpu": BackendConfig(
                    name="cpu",
                    indexes=IndexConfig(index_url=""),
                )
            }
        )
        args = _build_index_args(config, "cpu", {})
        assert "--index-url" not in args


# ---------------------------------------------------------------------------
# _build_constraints_content
# ---------------------------------------------------------------------------
class TestBuildConstraintsContent:
    def test_simple_constraints(self):
        config = PlatformConfig(
            backends={
                "cuda": BackendConfig(
                    name="cuda",
                    constraints={"torch": "=={torch}", "torchvision": ">=0.18"},
                )
            }
        )
        lines = _build_constraints_content(config, "cuda", {"torch": "2.12.0"})
        assert "torch==2.12.0" in lines
        assert "torchvision>=0.18" in lines

    def test_empty_constraints(self):
        config = PlatformConfig(
            backends={"cuda": BackendConfig(name="cuda")}
        )
        lines = _build_constraints_content(config, "cuda", {})
        assert lines == []

    def test_includes_compat_constraints(self):
        config = PlatformConfig(
            vars={"torch": "2.12.0", "cuda": "130"},
            backends={"cuda": BackendConfig(name="cuda")},
            compat={
                "torchao": CompatEntry(
                    package="torchao",
                    rules=[
                        CompatRule(conditions="torch>=2.11", constraint="<0.18"),
                    ],
                )
            },
        )
        lines = _build_constraints_content(config, "cuda", {"torch": "2.12.0", "cuda": "130"})
        assert "torchao<0.18" in lines

    def test_compat_scoped_to_active_backend(self):
        """ROCm-conditioned rules must not apply when building CUDA constraints."""
        config = PlatformConfig(
            vars={"torch": "2.12.0", "cuda": "130", "rocm": "7.2"},
            backends={
                "cuda": BackendConfig(name="cuda"),
                "rocm": BackendConfig(name="rocm"),
            },
            compat={
                "torchao": CompatEntry(
                    package="torchao",
                    rules=[
                        CompatRule(
                            conditions="torch>=2.12,rocm>=7",
                            constraint=">=0.18.0.dev0,<0.19",
                        ),
                        CompatRule(conditions="torch>=2.12", constraint="<0.19"),
                    ],
                )
            },
        )
        cuda_lines = _build_constraints_content(
            config, "cuda", {"torch": "2.12.0", "cuda": "130"}
        )
        rocm_lines = _build_constraints_content(
            config, "rocm", {"torch": "2.12.0", "rocm": "7.2"}
        )
        assert "torchao<0.19" in cuda_lines
        assert "torchao>=0.18.0.dev0,<0.19" not in cuda_lines
        assert "torchao>=0.18.0.dev0,<0.19" in rocm_lines


# ---------------------------------------------------------------------------
# _resolve_compat_constraints
# ---------------------------------------------------------------------------
class TestResolveCompatConstraints:
    def test_no_compat_section(self):
        config = PlatformConfig()
        assert _resolve_compat_constraints(config) == []

    def test_first_match_wins(self):
        config = PlatformConfig(
            vars={"torch": "2.12.0"},
            compat={
                "torchao": CompatEntry(
                    package="torchao",
                    rules=[
                        CompatRule(conditions="torch>=2.12", constraint="<0.20"),
                        CompatRule(conditions="torch>=2.11", constraint="<0.18"),
                    ],
                )
            },
        )
        lines = _resolve_compat_constraints(config, {"torch": "2.12.0"})
        assert lines == ["torchao<0.20"]

    def test_no_rules_match(self):
        config = PlatformConfig(
            vars={"torch": "2.9.0"},
            compat={
                "torchao": CompatEntry(
                    package="torchao",
                    rules=[
                        CompatRule(conditions="torch>=2.11", constraint="<0.18"),
                    ],
                )
            },
        )
        lines = _resolve_compat_constraints(config, {"torch": "2.9.0"})
        assert lines == []

    def test_cuda_version_normalized(self):
        config = PlatformConfig(
            vars={},
            compat={
                "torchao": CompatEntry(
                    package="torchao",
                    rules=[
                        CompatRule(conditions="cuda>=13", constraint="<0.20"),
                    ],
                )
            },
        )
        lines = _resolve_compat_constraints(config, {"cuda": "130"})
        assert lines == ["torchao<0.20"]

    def test_invalid_version_skipped(self):
        config = PlatformConfig(
            vars={},
            compat={
                "torchao": CompatEntry(
                    package="torchao",
                    rules=[
                        CompatRule(conditions="torch>=2.11", constraint="<0.18"),
                    ],
                )
            },
        )
        lines = _resolve_compat_constraints(config, {"torch": "not_a_version"})
        assert lines == []

    def test_multiple_packages(self):
        config = PlatformConfig(
            vars={"torch": "2.12.0"},
            compat={
                "torchao": CompatEntry(
                    package="torchao",
                    rules=[
                        CompatRule(conditions="torch>=2.11", constraint="<0.18"),
                    ],
                ),
                "flash-attn": CompatEntry(
                    package="flash-attn",
                    rules=[
                        CompatRule(conditions="torch>=2.10", constraint=">=2.5"),
                    ],
                ),
            },
        )
        lines = _resolve_compat_constraints(config, {"torch": "2.12.0"})
        assert "torchao<0.18" in lines
        assert "flash-attn>=2.5" in lines

    def test_constraint_formats_variables(self):
        config = PlatformConfig(
            vars={
                "cuda": "130",
                "torch": "2.10.0",
            },
            compat={
                "flashinfer-python": CompatEntry(
                    package="flashinfer-python",
                    rules=[
                        CompatRule(
                            conditions="torch>=2.10,cuda>=13",
                            constraint=">=0.6,<0.7",
                        ),
                    ],
                )
            },
        )
        lines = _resolve_compat_constraints(
            config, {"torch": "2.10.0", "cuda": "130"}
        )
        assert lines == ["flashinfer-python>=0.6,<0.7"]

    def test_vllm_exact_mapping_not_via_compat(self):
        from milabench.dependencies.platforms import VllmMapping

        config = PlatformConfig(
            vars={"cuda": "129", "torch": "2.10.0"},
            vllm_maps={
                "cuda": {
                    ("2.10.0", "129"): VllmMapping(
                        version="0.19.1",
                        find_links="https://example.com/v0.19.1",
                    ),
                    ("2.10.0", "130"): VllmMapping(
                        version="0.19.1+cu130",
                        find_links="https://example.com/v0.19.1",
                    ),
                }
            },
        )
        assert config.lookup_vllm("cuda", "129", "2.10.0").as_constraint() == (
            "vllm==0.19.1"
        )
        assert config.lookup_vllm("cuda", "130", "2.10.0").as_constraint() == (
            "vllm==0.19.1+cu130"
        )
        # compat path no longer synthesizes vllm pins
        assert _resolve_compat_constraints(
            config, {"torch": "2.10.0", "cuda": "129"}
        ) == []

    def test_vllm_requirement_includes_extras(self):
        assert _is_vllm_requirement("vllm")
        assert _is_vllm_requirement("vllm[bench]")
        assert _is_vllm_requirement("vllm[audio]==0.19.1")
        assert not _is_vllm_requirement("vllm-benchmark")

    def test_vllm_wheel_note_records_find_links(self):
        from milabench.dependencies.platforms import VllmMapping

        mapping = VllmMapping(
            version="0.19.1+cu130",
            find_links="https://github.com/vllm-project/vllm/releases/expanded_assets/v0.19.1",
        )
        lines = _vllm_wheel_note_lines(mapping)
        assert any("vllm==0.19.1+cu130" in line for line in lines)
        assert any("expanded_assets/v0.19.1" in line for line in lines)
        assert all(line.startswith("#") for line in lines)

    def test_vllm_note_is_header_comment_not_a_pin(self, tmp_path):
        from milabench.dependencies.platforms import VllmMapping

        path = tmp_path / "constraints.txt"
        path.write_text(
            "# Pinned with: cuda=130 torch=2.10.0\n"
            "# Generated by: milabench pin\n"
            "torch==2.10.0+cu130\n"
        )
        _append_vllm_wheel_note(
            path,
            VllmMapping(
                version="0.19.1+cu130",
                find_links="https://example.com/v0.19.1",
            ),
        )
        text = path.read_text()
        assert "vllm==0.19.1+cu130" in text
        assert "--find-links https://example.com/v0.19.1" in text
        assert not any(
            line.strip().startswith("vllm==") for line in text.splitlines()
        )


# ---------------------------------------------------------------------------
# _torch_pin — local version so uv cannot pick untagged PyPI torch
# ---------------------------------------------------------------------------
class TestTorchPin:
    def test_minor_range(self):
        assert _torch_minor_pin("2.12.1") == "torch>=2.12,<2.13"

    def test_cuda_exact_local(self):
        assert _torch_pin("2.10.0", "cuda", "130") == "torch==2.10.0+cu130"
        assert _torch_pin("2.10.0", "cuda", "126") == "torch==2.10.0+cu126"

    def test_rocm_exact_local(self):
        assert _torch_pin("2.10.0", "rocm", "7.1") == "torch==2.10.0+rocm7.1"

    def test_cpu_exact_local(self):
        assert _torch_pin("2.13.0", "cpu", "") == "torch==2.13.0+cpu"

    def test_hpu_falls_back_to_minor_range(self):
        assert _torch_pin("2.4.1", "hpu", "") == "torch>=2.4,<2.5"

    def test_local_labels(self):
        assert _torch_local_label("cuda", "130") == "cu130"
        assert _torch_local_label("rocm", "7.2") == "rocm7.2"
        assert _torch_local_label("cpu", "") == "cpu"
        assert _torch_local_label("hpu", "") is None


class TestValidatePinnedConstraintFile:
    def _write(self, tmp_path, body: str) -> Path:
        path = tmp_path / "constraints.txt"
        path.write_text(body)
        return path

    def test_accepts_matching_cu130(self, tmp_path):
        path = self._write(
            tmp_path,
            "torch==2.10.0+cu130\nnvidia-cuda-runtime==13.0.96\n",
        )
        _validate_pinned_constraint_file(path, "cuda", "130", "2.10.0")

    def test_rejects_untagged_torch(self, tmp_path):
        path = self._write(tmp_path, "torch==2.10.0\n")
        with pytest.raises(RuntimeError, match="expected torch==2.10.0\\+cu130"):
            _validate_pinned_constraint_file(path, "cuda", "130", "2.10.0")

    def test_rejects_cuda13_mixed_with_cu12_runtime(self, tmp_path):
        path = self._write(
            tmp_path,
            "torch==2.10.0+cu130\n"
            "nvidia-cuda-runtime==13.3.29\n"
            "nvidia-cuda-runtime-cu12==12.8.90\n",
        )
        with pytest.raises(RuntimeError, match="nvidia-cuda-runtime-cu12"):
            _validate_pinned_constraint_file(path, "cuda", "130", "2.10.0")

    def test_allows_cutlass_cu12_name_on_cuda13(self, tmp_path):
        path = self._write(
            tmp_path,
            "torch==2.10.0+cu130\n"
            "nvidia-cuda-runtime==13.0.96\n"
            "nvidia-cutlass-dsl-libs-cu12==4.6.2\n",
        )
        _validate_pinned_constraint_file(path, "cuda", "130", "2.10.0")

    def test_cuda126_may_pin_cu12_runtime(self, tmp_path):
        path = self._write(
            tmp_path,
            "torch==2.10.0+cu126\nnvidia-cuda-runtime-cu12==12.6.77\n",
        )
        _validate_pinned_constraint_file(path, "cuda", "126", "2.10.0")


# ---------------------------------------------------------------------------
# _ensure_build_backends
# ---------------------------------------------------------------------------
class TestEnsureBuildBackends:
    """`uv pip compile --no-build-isolation` needs the [build] backends present.

    Guards the fix for `ModuleNotFoundError: No module named 'setuptools'` when
    building source dists (e.g. the torchtitan git dep) during pinning.
    """

    @pytest.fixture(autouse=True)
    def _reset_seed_flag(self):
        import milabench.dependencies.pin as pin

        pin._BUILD_BACKENDS_SEEDED = False
        yield
        pin._BUILD_BACKENDS_SEEDED = False

    def _fake_run(self, calls, returncode=0, stderr=""):
        import subprocess as _sp

        def run(cmd, *args, **kwargs):
            calls.append(cmd)
            return _sp.CompletedProcess(cmd, returncode, stdout="", stderr=stderr)

        return run

    def test_installs_build_requires(self, monkeypatch):
        import milabench.dependencies.pin as pin

        calls = []
        monkeypatch.setattr(pin.subprocess, "run", self._fake_run(calls))
        config = PlatformConfig(build_requires=["setuptools", "wheel", "hatchling"])

        pin._ensure_build_backends(config)

        assert calls == [["uv", "pip", "install", "setuptools", "wheel", "hatchling"]]

    def test_seeds_only_once_per_process(self, monkeypatch):
        import milabench.dependencies.pin as pin

        calls = []
        monkeypatch.setattr(pin.subprocess, "run", self._fake_run(calls))
        config = PlatformConfig(build_requires=["setuptools"])

        pin._ensure_build_backends(config)
        pin._ensure_build_backends(config)

        assert len(calls) == 1

    def test_noop_when_no_build_requires(self, monkeypatch):
        import milabench.dependencies.pin as pin

        calls = []
        monkeypatch.setattr(pin.subprocess, "run", self._fake_run(calls))

        pin._ensure_build_backends(PlatformConfig(build_requires=None))
        pin._ensure_build_backends(PlatformConfig(build_requires=[]))

        assert calls == []

    def test_raises_on_install_failure(self, monkeypatch):
        import milabench.dependencies.pin as pin

        calls = []
        monkeypatch.setattr(
            pin.subprocess,
            "run",
            self._fake_run(calls, returncode=1, stderr="boom"),
        )
        config = PlatformConfig(build_requires=["setuptools"])

        with pytest.raises(RuntimeError, match="Failed to install .build. requires"):
            pin._ensure_build_backends(config)

    def test_repo_toml_declares_setuptools(self):
        """The shipped platforms.toml must seed setuptools for torchtitan builds."""
        from milabench.dependencies.platforms import load_platform_config

        repo_toml = Path(__file__).resolve().parents[2] / "platforms.toml"
        config = load_platform_config(path=repo_toml)
        assert "setuptools" in (config.build_requires or [])